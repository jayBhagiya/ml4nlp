# Graph Transformer Showcase

Focused PyTorch Geometric reimplementation of Dwivedi and Bresson's *A Generalization of Transformer Networks to Graphs*. The experiment tests sparse attention, Laplacian positional encoding, and normalization choice on the PATTERN node-classification benchmark.

This is a modern focused ablation, not a claim of bit-for-bit reproduction. It keeps the paper's 10 layers, 8 heads, hidden size 80, two Laplacian eigenvectors, approximately 500K parameters, optimizer settings, and PATTERN splits. Test data is evaluated only after selecting the checkpoint with the lowest validation loss.

## Local setup

```console
$ uv venv .venv
$ uv pip install --python .venv/bin/python --torch-backend cpu -e .
$ .venv/bin/python -m graph_transformer.train --data data --prepare-only --wandb-mode disabled
```

Use `--torch-backend auto` on a GPU machine.

## Tests

```console
$ .venv/bin/python -m unittest discover -s tests -v
```

## Training

```console
$ .venv/bin/python -m graph_transformer.train \
    --data data \
    --output runs/pattern/paper/seed-42 \
    --variant paper \
    --seed 42
```

Available variants are `paper`, `no-pe`, `full`, and `layer-norm`. W&B logging is online by default; use `--wandb-mode offline` when the machine cannot reach W&B or `disabled` for smoke tests.

Each run writes the best validation checkpoint to `model.pt` and full training history plus final test metrics to `metrics.json`.

Cluster instructions live in [`submit_jobs/README.md`](submit_jobs/README.md). Submit files are provided for manual use and are never submitted by this project.
