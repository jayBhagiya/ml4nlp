#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"

UV_VERSION=0.11.30
UV_BIN="${DATA_DIR}/tools/uv-${UV_VERSION}/bin/uv"
VENV="${VENV:-${DATA_DIR}/venvs/graph-transformer-showcase}"
export UV_CACHE_DIR="${DATA_DIR}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${DATA_DIR}/python"
export WANDB_DIR="${WANDB_DIR:-${DATA_DIR}/wandb}"
export WANDB_CACHE_DIR="${DATA_DIR}/cache/wandb"
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == "setup" ]]; then
    mkdir -p \
        "${DATA_DIR}/cache/uv" \
        "${DATA_DIR}/cache/wandb" \
        "${DATA_DIR}/datasets" \
        "${DATA_DIR}/python" \
        "${DATA_DIR}/runs" \
        "${DATA_DIR}/wandb" \
        "$(dirname "${UV_BIN}")" \
        "$(dirname "${VENV}")"
    if [[ ! -x "${UV_BIN}" ]]; then
        python -m pip install --disable-pip-version-check --no-deps \
            --prefix "${DATA_DIR}/tools/uv-${UV_VERSION}" "uv==${UV_VERSION}"
    fi
    "${UV_BIN}" python install 3.11.15
    "${UV_BIN}" venv --clear --python 3.11.15 "${VENV}"
    "${UV_BIN}" pip install --python "${VENV}/bin/python" \
        --torch-backend cu118 -e "${PROJECT_DIR}"
    "${VENV}/bin/python" -c \
        'import torch; assert torch.version.cuda == "11.8", torch.version.cuda; print(torch.__version__, torch.version.cuda)'
    cd "${PROJECT_DIR}"
    "${VENV}/bin/python" -m graph_transformer.train \
        --data "${DATA_DIR}/datasets" --prepare-only --wandb-mode disabled
    exit
fi

PYTHON="${VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Environment missing at ${VENV}; submit condor/setup.sub first." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
fi
exec "${PYTHON}" -u "$@"

