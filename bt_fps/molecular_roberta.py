import math
import re
import sys
import types
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SMI_SYMBOLS = (
    r"Li|Be|Na|Mg|Al|Si|Cl|Ca|Zn|As|Se|se|Br|Rb|Sr|Ag|Sn|Te|te|Cs|Ba|Bi|[\d]|"
    r"[HBCNOFPSKIbcnops#%\)\(\+\-\\\/\.=@\[\]]"
)
SMI_REGEX = re.compile(SMI_SYMBOLS)


def tokenize_smiles(smiles: str) -> List[str]:
    return SMI_REGEX.findall(smiles.strip())


class Vocabulary:
    def __init__(self) -> None:
        self.symbols: List[str] = []
        self.indices: Dict[str, int] = {}
        self.bos_index = self.add_symbol("<s>")
        self.pad_index = self.add_symbol("<pad>")
        self.eos_index = self.add_symbol("</s>")
        self.unk_index = self.add_symbol("<unk>")

    def __len__(self) -> int:
        return len(self.symbols)

    def add_symbol(self, symbol: str) -> int:
        if symbol in self.indices:
            return self.indices[symbol]
        idx = len(self.symbols)
        self.indices[symbol] = idx
        self.symbols.append(symbol)
        return idx

    def index(self, symbol: str) -> int:
        return self.indices.get(symbol, self.unk_index)

    def encode_smiles(self, smiles: str) -> List[int]:
        ids = [self.bos_index]
        ids.extend(self.index(token) for token in tokenize_smiles(smiles))
        ids.append(self.eos_index)
        return ids

    @classmethod
    def load(cls, dict_path: str) -> "Vocabulary":
        vocab = cls()
        with open(dict_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                split_at = line.rfind(" ")
                if split_at == -1:
                    raise ValueError(
                        f"Bad dictionary line {line_number} in {dict_path!r}: {line!r}"
                    )
                vocab.add_symbol(line[:split_at])
        return vocab


def make_positions(tokens: torch.Tensor, padding_idx: int) -> torch.Tensor:
    mask = tokens.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "gelu"):
        return F.gelu(x.float()).type_as(x)
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def activation(name: str):
    if name == "gelu":
        return gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "linear":
        return lambda x: x
    raise ValueError(f"Unsupported activation function: {name}")


class FairseqSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tgt_len, batch_size, embed_dim = x.size()

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim)
        k = k.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim)
        v = v.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(
                batch_size, self.num_heads, tgt_len, tgt_len
            )
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, tgt_len)

        attn_probs = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        return self.out_proj(attn)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = activation(activation_fn)
        self.self_attn = FairseqSelfAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn(x, key_padding_mask=self_attn_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attn_layer_norm(residual + x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.final_layer_norm(residual + x)


class MolecularRobertaEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_positions: int,
        embed_dim: int,
        ffn_dim: int,
        num_layers: int,
        num_heads: int,
        padding_idx: int = 1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.max_positions = max_positions
        self.embed_dim = embed_dim
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx)
        self.embed_positions = nn.Embedding(
            max_positions + padding_idx + 1,
            embed_dim,
            padding_idx,
        )
        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.size(1) > self.max_positions:
            raise ValueError(
                f"Input length {tokens.size(1)} exceeds max_positions={self.max_positions}"
            )

        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        positions = make_positions(tokens, self.padding_idx)
        x = self.embed_tokens(tokens) + self.embed_positions(positions)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, self_attn_padding_mask=padding_mask)
        return x.transpose(0, 1)


@dataclass
class ModelConfig:
    vocab_size: int
    max_positions: int
    embed_dim: int
    ffn_dim: int
    num_layers: int
    num_heads: int
    activation_fn: str


def _torch_load(path: str, map_location: str):
    class AverageMeter(object):
        pass

    class TimeMeter(object):
        pass

    class StopwatchMeter(object):
        pass

    AverageMeter.__module__ = "fairseq.meters"
    TimeMeter.__module__ = "fairseq.meters"
    StopwatchMeter.__module__ = "fairseq.meters"

    old_fairseq = sys.modules.get("fairseq")
    old_meters = sys.modules.get("fairseq.meters")
    if old_fairseq is None:
        sys.modules["fairseq"] = types.ModuleType("fairseq")
    meters_module = types.ModuleType("fairseq.meters")
    meters_module.AverageMeter = AverageMeter
    meters_module.TimeMeter = TimeMeter
    meters_module.StopwatchMeter = StopwatchMeter
    sys.modules["fairseq.meters"] = meters_module

    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals(
                [argparse.Namespace, AverageMeter, TimeMeter, StopwatchMeter]
            )
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        if old_fairseq is None:
            sys.modules.pop("fairseq", None)
        else:
            sys.modules["fairseq"] = old_fairseq
        if old_meters is None:
            sys.modules.pop("fairseq.meters", None)
        else:
            sys.modules["fairseq.meters"] = old_meters


def _get_model_state(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Checkpoint must be a state dict or contain a 'model' state dict")


def _get_arg(checkpoint, name: str, default):
    args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    return getattr(args, name, default)


def _count_layers(state: Dict[str, torch.Tensor]) -> int:
    pattern = re.compile(r"decoder\.sentence_encoder\.layers\.(\d+)\.")
    layer_ids = [int(match.group(1)) for key in state for match in [pattern.match(key)] if match]
    if not layer_ids:
        raise ValueError("Could not infer encoder layer count from checkpoint keys")
    return max(layer_ids) + 1


def infer_config(
    checkpoint,
    state: Dict[str, torch.Tensor],
    num_heads_override: Optional[int] = None,
) -> ModelConfig:
    token_weight = state["decoder.sentence_encoder.embed_tokens.weight"]
    position_weight = state["decoder.sentence_encoder.embed_positions.weight"]
    fc1_weight = state["decoder.sentence_encoder.layers.0.fc1.weight"]

    embed_dim = int(token_weight.shape[1])
    num_heads = num_heads_override or _get_arg(
        checkpoint,
        "encoder_attention_heads",
        None,
    ) or 8
    activation_fn = _get_arg(checkpoint, "activation_fn", None) or "gelu"
    return ModelConfig(
        vocab_size=int(token_weight.shape[0]),
        max_positions=int(position_weight.shape[0]) - 2,
        embed_dim=embed_dim,
        ffn_dim=int(fc1_weight.shape[0]),
        num_layers=_count_layers(state),
        num_heads=int(num_heads),
        activation_fn=str(activation_fn),
    )


def build_model_from_checkpoint(
    checkpoint_path: str,
    dict_path: str,
    device: torch.device,
    num_heads: Optional[int] = None,
) -> Tuple[MolecularRobertaEncoder, Vocabulary, ModelConfig]:
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    state = _get_model_state(checkpoint)
    vocab = Vocabulary.load(dict_path)
    config = infer_config(checkpoint, state, num_heads_override=num_heads)

    if len(vocab) > config.vocab_size:
        raise ValueError(
            f"Dictionary has {len(vocab)} symbols, but checkpoint vocab size is "
            f"{config.vocab_size}"
        )

    model = MolecularRobertaEncoder(
        vocab_size=config.vocab_size,
        max_positions=config.max_positions,
        embed_dim=config.embed_dim,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=float(_get_arg(checkpoint, "dropout", 0.1)),
        attention_dropout=float(_get_arg(checkpoint, "attention_dropout", 0.1)),
        activation_dropout=float(_get_arg(checkpoint, "activation_dropout", 0.0)),
        activation_fn=config.activation_fn,
    )

    prefix = "decoder.sentence_encoder."
    encoder_state = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    load_result = model.load_state_dict(encoder_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint did not match minimal encoder. "
            f"Missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    return model, vocab, config


def encode_batch(
    smiles_batch: Iterable[str],
    vocab: Vocabulary,
    max_positions: int,
    device: torch.device,
) -> torch.Tensor:
    encoded: List[List[int]] = []
    for smiles in smiles_batch:
        ids = vocab.encode_smiles(smiles)
        if len(ids) > max_positions:
            ids = ids[: max_positions - 1] + [ids[-1]]
        encoded.append(ids)

    max_len = max(len(ids) for ids in encoded)
    batch = torch.full(
        (len(encoded), max_len),
        vocab.pad_index,
        dtype=torch.long,
        device=device,
    )
    for row, ids in enumerate(encoded):
        batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return batch


def pool_features(
    features: torch.Tensor,
    tokens: torch.Tensor,
    padding_idx: int,
    feature_type: str,
) -> torch.Tensor:
    if feature_type == "bos":
        return features[:, 0, :]
    if feature_type == "avg":
        mask = tokens.ne(padding_idx).unsqueeze(-1).type_as(features)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (features * mask).sum(dim=1) / denom
    raise ValueError(f"Unknown feature_type: {feature_type}")


def default_dict_path(model_dir: str) -> str:
    return str(Path(model_dir) / "dict.txt")


def default_checkpoint_path(model_dir: str, checkpoint_file: str) -> str:
    return str(Path(model_dir) / checkpoint_file)
