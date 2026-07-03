from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_repo_paths(nanotron_src: Path | None, lerobot_src: Path | None) -> None:
    for path in (PROJECT_ROOT, nanotron_src, lerobot_src):
        if path is None:
            continue
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SmolVLA module topology for TP planning.")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--nanotron-src", type=Path, default=None)
    parser.add_argument("--lerobot-src", type=Path, default=None)
    parser.add_argument("--max-lines", type=int, default=300)
    return parser.parse_args()


def requires_grad(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters(recurse=False))


def print_linear_modules(root_name: str, module: nn.Module, max_lines: int) -> None:
    print(f"\n[{root_name}] linear modules")
    count = 0
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        count += 1
        if count <= max_lines:
            bias = child.bias is not None
            trainable = requires_grad(child)
            print(
                f"{count:04d} {name} "
                f"in={child.in_features} out={child.out_features} "
                f"bias={bias} trainable={trainable} dtype={child.weight.dtype}"
            )
    print(f"total_linear_modules={count}")
    if count > max_lines:
        print(f"truncated_after={max_lines}")


def print_param_summary(root_name: str, module: nn.Module) -> None:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    print(f"{root_name}: params={total:,} trainable={trainable:,}")


def main() -> None:
    args = parse_args()
    add_repo_paths(args.nanotron_src, args.lerobot_src)

    from nanotron_smolvla.config import load_config
    from nanotron_smolvla.model import SmolVLALossModule

    cfg = load_config(args.config_file)
    torch.manual_seed(cfg.seed)
    module = SmolVLALossModule(cfg.model)
    module.eval()

    model = module.model
    print("Nanotron-SmVLA topology audit")
    print(f"config={args.config_file}")
    print(f"vlm_model_name={cfg.model.vlm_model_name}")
    print(f"num_vlm_layers={cfg.model.num_vlm_layers}")
    print(f"num_expert_layers={cfg.model.num_expert_layers}")
    print(f"train_expert_only={cfg.model.train_expert_only}")
    print(f"freeze_vision_encoder={cfg.model.freeze_vision_encoder}")
    print(f"attention_mode={cfg.model.attention_mode}")

    print_param_summary("loss_module", module)
    print_param_summary("vla_flow_matching", model)
    print_param_summary("vlm_with_expert", model.vlm_with_expert)
    print_param_summary("vlm_text_model", model.vlm_with_expert.get_vlm_model().text_model)
    print_param_summary("lm_expert", model.vlm_with_expert.lm_expert)

    print("\n[top-level projections]")
    for name in (
        "state_proj",
        "action_in_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "action_out_proj",
    ):
        child = getattr(model, name)
        print(
            f"{name}: in={child.in_features} out={child.out_features} "
            f"bias={child.bias is not None} trainable={requires_grad(child)} dtype={child.weight.dtype}"
        )

    print_linear_modules("vlm_text_model", model.vlm_with_expert.get_vlm_model().text_model, args.max_lines)
    print_linear_modules("lm_expert", model.vlm_with_expert.lm_expert, args.max_lines)


if __name__ == "__main__":
    main()
