# Running the PATTERN campaign on HTCondor

These files prepare and run the Graph Transformer experiment on an HTCondor cluster. They do not submit jobs automatically.

## 1. Check paths

The submit files assume:

- project checkout: `/path/to/ml4nlp`
- persistent data: `/path/to/large-storage/ml4nlp`

Edit `project_dir` or `data_dir` in both `.sub` files if your checkout differs.

## 2. Prepare directories and W&B

Run on `your submit machine`:

```console
$ mkdir -p /path/to/large-storage/ml4nlp/logs
$ export WANDB_API_KEY="..."
```

`train.sub` forwards only `HOME` and `WANDB_API_KEY`; the key is not stored in the repository. If W&B login is already persisted in your mounted home directory, the environment variable may be unnecessary.

## 3. Create the uv environment and dataset cache

```console
$ condor_submit -batch-name gt-setup submit_jobs/setup.sub
$ condor_q
```

Wait for setup to finish successfully. Check its `.out`, `.err`, and `.log` files under the data directory. Setup installs Python 3.11 and the CUDA environment with uv, then downloads and preprocesses PATTERN with two Laplacian eigenvectors.

## 4. Run a single smoke experiment

Before the campaign, copy `train.sub`, keep one queue row, and add `--epochs 2` to `arguments`. Submit that temporary file and verify CUDA, W&B logging, dataset loading, and output creation.

## 5. Submit the campaign

```console
$ condor_submit -batch-name gt-pattern submit_jobs/train.sub
$ condor_q
```

The array contains four variants and three seeds, for 12 jobs total:

- `paper`: sparse attention, Laplacian PE, BatchNorm
- `no-pe`: sparse attention, no positional encoding, BatchNorm
- `full`: full within-graph attention, Laplacian PE, BatchNorm
- `layer-norm`: sparse attention, Laplacian PE, LayerNorm

Each job writes `model.pt` and `metrics.json` under:

```text
/path/to/large-storage/ml4nlp/runs/gt-v1/pattern/<variant>/seed-<seed>/
```

W&B runs use group `pattern-ablation` and names such as `pattern-paper-seed-42`.

## 6. Diagnose failures

```console
$ condor_q -hold
$ condor_q -better-analyze <job-id>
```

Batch size defaults to 26, matching the published PATTERN configuration. If a GPU runs out of memory, add a smaller `--batch-size` to `arguments`; record that change because it affects runtime comparisons.
