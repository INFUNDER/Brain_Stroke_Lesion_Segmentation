# Source this file to use the revision environment:  source revision/env/activate.sh
_REV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
source "$_REV_ROOT/.venv/bin/activate"
export nnUNet_raw="$_REV_ROOT/nnunet/raw"
export nnUNet_preprocessed="$_REV_ROOT/nnunet/preprocessed"
export nnUNet_results="$_REV_ROOT/nnunet/results"
export nnUNet_compile=f          # torch.compile off for every model: comparable timings, no tracing of custom operators
export PYTHONPATH="$_REV_ROOT/revision${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
