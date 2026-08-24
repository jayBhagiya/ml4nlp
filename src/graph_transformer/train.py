import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from .model import GraphTransformer, count_parameters


VARIANTS = {
    "paper": {
        "use_laplacian_pe": True,
        "full_attention": False,
        "normalization": "batch",
    },
    "no-pe": {
        "use_laplacian_pe": False,
        "full_attention": False,
        "normalization": "batch",
    },
    "full": {
        "use_laplacian_pe": True,
        "full_attention": True,
        "normalization": "batch",
    },
    "layer-norm": {
        "use_laplacian_pe": True,
        "full_attention": False,
        "normalization": "layer",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Graph Transformer on PATTERN")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/pattern"))
    parser.add_argument("--variant", choices=VARIANTS, default="paper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=26)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", default="graph-transformer-showcase")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(root: Path):
    transform = AddLaplacianEigenvectorPE(
        k=2,
        attr_name="laplacian_pe",
        is_undirected=True,
        tol=1e-2,
    )
    return tuple(
        GNNBenchmarkDataset(
            root=str(root),
            name="PATTERN",
            split=split,
            pre_transform=transform,
        )
        for split in ("train", "val", "test")
    )


def balanced_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    recalls = []
    for label in range(2):
        selected = labels == label
        if selected.any():
            recalls.append((predictions[selected] == label).float().mean())
    return float(torch.stack(recalls).mean() * 100)


def weighted_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2)
    weights = (labels.numel() - counts).float() / labels.numel()
    return F.cross_entropy(logits, labels, weight=weights)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_nodes = 0
    predictions = []
    labels = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for data in loader:
            data = data.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(
                data.x,
                data.edge_index,
                data.batch,
                data.laplacian_pe,
            )
            loss = weighted_loss(logits, data.y)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.detach().item() * data.num_nodes
            total_nodes += data.num_nodes
            predictions.append(logits.detach().argmax(dim=1).cpu())
            labels.append(data.y.detach().cpu())

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    return total_loss / total_nodes, balanced_accuracy(predictions, labels)


def main() -> None:
    args = parse_args()
    datasets = load_datasets(args.data)
    if args.prepare_only:
        print(
            "Prepared PATTERN splits:",
            ", ".join(
                f"{split}={len(data)}"
                for split, data in zip(("train", "val", "test"), datasets)
            ),
        )
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    import wandb

    seed_everything(args.seed)
    device = torch.device(args.device)
    train_data, validation_data, test_data = datasets
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader = DataLoader(test_data, shuffle=False, **loader_options)

    settings = VARIANTS[args.variant]
    model = GraphTransformer(input_dim=train_data.num_node_features, **settings).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        group="pattern-ablation",
        name=f"pattern-{args.variant}-seed-{args.seed}",
        mode=args.wandb_mode,
        config={**wandb_config, **settings, "parameters": parameter_count},
        tags=["PATTERN", args.variant],
    )

    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    history = []
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
            validation_loss, validation_accuracy = run_epoch(model, validation_loader, device)
            scheduler.step(validation_loss)
            learning_rate = optimizer.param_groups[0]["lr"]
            elapsed_seconds = time.monotonic() - started
            metrics = {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/balanced_accuracy": train_accuracy,
                "validation/loss": validation_loss,
                "validation/balanced_accuracy": validation_accuracy,
                "learning_rate": learning_rate,
                "elapsed_seconds": elapsed_seconds,
            }
            history.append(metrics)
            run.log(metrics, step=epoch)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.2f} "
                f"lr={learning_rate:.2e}"
            )
            if learning_rate <= args.min_learning_rate:
                break
            if elapsed_seconds >= args.max_hours * 3600:
                break

        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        model.load_state_dict(best_state)
        test_loss, test_accuracy = run_epoch(model, test_loader, device)
        elapsed_seconds = time.monotonic() - started
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        result = {
            "dataset": "PATTERN",
            "variant": args.variant,
            "seed": args.seed,
            "parameters": parameter_count,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "test_loss": test_loss,
            "test_balanced_accuracy": test_accuracy,
            "elapsed_seconds": elapsed_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "settings": settings,
            "history": history,
        }
        checkpoint = args.output / "model.pt"
        torch.save(
            {
                "model": best_state,
                "input_dim": train_data.num_node_features,
                "settings": settings,
                "variant": args.variant,
                "seed": args.seed,
                "best_epoch": best_epoch,
            },
            checkpoint,
        )
        (args.output / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        run.summary.update({key: value for key, value in result.items() if key != "history"})
        run.log(
            {"test/loss": test_loss, "test/balanced_accuracy": test_accuracy},
            step=len(history) + 1,
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "history"},
                indent=2,
            )
        )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
