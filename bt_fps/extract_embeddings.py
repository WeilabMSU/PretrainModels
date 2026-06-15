#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List


def read_smiles(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_runtime():
    try:
        import numpy as np
        import torch
        try:
            from .molecular_roberta import (
                build_model_from_checkpoint,
                default_checkpoint_path,
                default_dict_path,
                encode_batch,
                pool_features,
            )
        except ImportError:
            from molecular_roberta import (
                build_model_from_checkpoint,
                default_checkpoint_path,
                default_dict_path,
                encode_batch,
                pool_features,
            )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script needs only PyTorch and NumPy. Install them with: "
            "pip install torch numpy"
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "build_model_from_checkpoint": build_model_from_checkpoint,
        "default_checkpoint_path": default_checkpoint_path,
        "default_dict_path": default_dict_path,
        "encode_batch": encode_batch,
        "pool_features": pool_features,
    }


def choose_device(name: str, torch_module):
    if name == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    return torch_module.device(name)


def extract_embeddings(args: argparse.Namespace):
    runtime = load_runtime()
    np = runtime["np"]
    torch = runtime["torch"]

    model_dir = Path(args.model_dir)
    checkpoint_path = args.checkpoint or runtime["default_checkpoint_path"](
        str(model_dir),
        args.checkpoint_file,
    )
    dict_path = args.dict_file or runtime["default_dict_path"](str(model_dir))
    device = choose_device(args.device, torch)

    model, vocab, config = runtime["build_model_from_checkpoint"](
        checkpoint_path=checkpoint_path,
        dict_path=dict_path,
        device=device,
        num_heads=args.num_heads,
    )

    smiles = read_smiles(args.smi_file)
    if not smiles:
        raise ValueError(f"No non-empty SMILES found in {args.smi_file}")

    pooled_batches = []
    with torch.no_grad():
        for start in range(0, len(smiles), args.batch_size):
            batch_smiles = smiles[start: start + args.batch_size]
            tokens = runtime["encode_batch"](
                batch_smiles,
                vocab=vocab,
                max_positions=config.max_positions,
                device=device,
            )
            features = model(tokens)
            pooled = runtime["pool_features"](
                features=features,
                tokens=tokens,
                padding_idx=vocab.pad_index,
                feature_type=args.feature_type,
            )
            pooled_batches.append(pooled.detach().cpu().numpy())

    return np.concatenate(pooled_batches, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract molecular embeddings from this repository's pretrained checkpoints."
    )
    parser.add_argument(
        "--model-dir",
        default="chembl27_512",
        help="Folder containing checkpoint_best.pt and dict.txt.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="checkpoint_best.pt",
        help="Checkpoint filename inside --model-dir.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Overrides --model-dir/--checkpoint-file.",
    )
    parser.add_argument(
        "--dict-file",
        default=None,
        help="Optional explicit dictionary path. Defaults to --model-dir/dict.txt.",
    )
    parser.add_argument(
        "--smi-file",
        required=True,
        help="Input .smi file. One SMILES string per line.",
    )
    parser.add_argument(
        "--output",
        default="embeddings.npy",
        help="Output .npy file.",
    )
    parser.add_argument(
        "--feature-type",
        choices=["bos", "avg"],
        default="bos",
        help="Pooling method: 'bos' uses the first-token BT-FP; 'avg' averages tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of molecules per inference batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cpu', 'cuda', or a device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Optional override if a checkpoint does not store encoder_attention_heads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = extract_embeddings(args)
    runtime = load_runtime()
    np = runtime["np"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    print(f"Saved {output}")
    print(f"Features shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
