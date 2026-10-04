# Graph Transformers on PATTERN

Seminar project for *Machine Learning for Natural Language Processing* (Saarland University). It reimplements the Graph Transformer from Dwivedi and Bresson, [*A Generalization of Transformer Networks to Graphs*](https://arxiv.org/abs/2012.09699), in PyTorch Geometric and tests three of its design choices on the PATTERN node-classification benchmark:

| Variant | Attention | Laplacian positional encoding | Normalization |
|---|---|---|---|
| `paper` | Graph neighbours | Yes | BatchNorm |
| `no-pe` | Graph neighbours | No | BatchNorm |
| `full` | Every node in the same graph | Yes | BatchNorm |
| `layer-norm` | Graph neighbours | Yes | LayerNorm |

This is a focused ablation, not a bit-for-bit reproduction. It keeps the paper's 10 layers, 8 heads, hidden size 80, two Laplacian eigenvectors, about 500K parameters, optimizer settings, and PATTERN splits. Test data is evaluated only once, on the checkpoint with the lowest validation loss. The findings are in the [project write-up](https://jaybhagiya.me/projects/graph-transformers-on-pattern/).

## Repository layout

| Path | Contents |
|---|---|
| `src/graph_transformer/model.py` | Graph attention layer and the Graph Transformer model |
| `src/graph_transformer/train.py` | Dataset preparation, training, and test evaluation on PATTERN |
| `tests/` | Unit tests for attention, full-attention edges, permutation equivariance, and parameter count |
| `condor/` | HTCondor jobs for the full campaign on a GPU cluster |

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 or newer works.

```bash
git clone https://github.com/jayBhagiya/ml4nlp.git
cd ml4nlp
uv venv .venv

# CPU only (any machine)
uv pip install --python .venv/bin/python --torch-backend cpu -e .

# or pick the CUDA build that matches your GPU driver
uv pip install --python .venv/bin/python --torch-backend auto -e .
```

Then download PATTERN (88 MB) and precompute the Laplacian eigenvectors once. This takes more than 10 minutes on a laptop CPU and about 3 GB of disk:

```bash
.venv/bin/python -m graph_transformer.train --data data --prepare-only
```

## Training

The script picks the GPU automatically when one is available and takes `--help` for every option:

```bash
.venv/bin/python -m graph_transformer.train --data data --variant paper --seed 42 \
  --output runs/pattern/paper/seed-42
```

Training uses Adam at `5e-4`, halves the learning rate when validation loss stops improving for 10 epochs, and stops once it falls below `1e-6` or after `--max-hours` (24 by default). On the project's cluster GPUs a run took between 30 minutes and 5.5 hours, with a peak of about 2.4 GB GPU memory for the sparse variants and 5.5 GB for `full`.

PATTERN has 10,000 training graphs, so a full run isn't practical on a laptop CPU. For a quick end-to-end check, add `--epochs 1`. W&B logging is off by default; add `--wandb-mode online` after `wandb login` to enable it.

### Outputs

Each run folder gets `model.pt`, the best validation checkpoint with its settings, and `metrics.json`, which holds the full training history, the parameter count, peak GPU memory, and the test loss and balanced accuracy. PATTERN is imbalanced (about 18% of nodes belong to the pattern), so both training loss and accuracy are class-balanced.

## Running on an HTCondor cluster

`condor/` runs the full campaign as cluster jobs inside the `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` Docker image: 12 training jobs (4 variants × seeds 21, 42, and 87). A setup job installs uv and a CUDA environment once into shared storage and prepares PATTERN; every training job reuses it.

**1. Edit the variables at the top of each `.sub` file:**

| Variable | Set it to |
|---|---|
| `project_dir` | Where this repository is cloned, on a path the worker nodes can read |
| `data_dir` | Shared storage for the environment, dataset, caches, logs, and results (plan for about 10 GB) |
| `campaign` | Optional: the folder name for this set of runs under `data_dir/runs/` |

**2. Adapt the resource lines to your cluster:**
- `requirements` selects GPUs by memory (`GPUs_GlobalMemoryMb`). Add any extra constraints your cluster needs, such as a `UidDomain` or machine pool.
- `+WantGPUHomeMounted = true` is a site-specific attribute that mounts the home directory in the container. Remove it if your cluster doesn't define it.
- Your cluster must support the Docker universe. If it doesn't, switch to `universe = vanilla` and make sure the workers have Python available for `run.sh setup`.
- The training jobs log to W&B (`--wandb-mode online`). Run `wandb login` on a machine that shares your home directory with the workers, or change the flag to `offline` or `disabled` in `train.sub`.

**3. Create the log folder and submit:**

```bash
mkdir -p /path/to/large-storage/ml4nlp/logs

condor_submit -batch-name gt-setup condor/setup.sub    # once: environment and PATTERN
condor_submit -batch-name gt-pattern condor/train.sub
```

Before the full campaign, you can test one job: copy `train.sub`, keep a single `queue` row, and add `--epochs 2` to `arguments`. If a GPU runs out of memory, pass a smaller `--batch-size` (the default, 26, matches the paper); it changes runtime comparisons.

Results land in `data_dir/runs/<campaign>/pattern/<variant>/seed-<seed>/` and job logs in `data_dir/logs/`. Use `condor_q -better-analyze <job-id>` to inspect a waiting job.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
