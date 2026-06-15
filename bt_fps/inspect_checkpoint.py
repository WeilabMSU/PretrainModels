#!/usr/bin/env python3
import argparse
import sys
import types


def torch_load(path: str, torch_module):
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
        if hasattr(torch_module.serialization, "add_safe_globals"):
            torch_module.serialization.add_safe_globals(
                [argparse.Namespace, AverageMeter, TimeMeter, StopwatchMeter]
            )
        try:
            return torch_module.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch_module.load(path, map_location="cpu")
        except Exception:
            return torch_module.load(path, map_location="cpu", weights_only=False)
    finally:
        if old_fairseq is None:
            sys.modules.pop("fairseq", None)
        else:
            sys.modules["fairseq"] = old_fairseq
        if old_meters is None:
            sys.modules.pop("fairseq.meters", None)
        else:
            sys.modules["fairseq.meters"] = old_meters


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a pretrained checkpoint.")
    parser.add_argument("checkpoint", help="Path to checkpoint_best.pt")
    parser.add_argument("--limit", type=int, default=30, help="Number of keys to print")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Install PyTorch first: pip install torch") from exc

    checkpoint = torch_load(args.checkpoint, torch)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    print(f"checkpoint type: {type(checkpoint).__name__}")
    print(f"state dict keys: {len(state)}")

    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if saved_args is not None:
        for name in [
            "encoder_layers",
            "encoder_embed_dim",
            "encoder_ffn_embed_dim",
            "encoder_attention_heads",
            "max_positions",
            "activation_fn",
        ]:
            print(f"{name}: {getattr(saved_args, name, None)}")

    for key in list(state.keys())[: args.limit]:
        value = state[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else ""
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
