# Shared setup for every PBS job of the revision. Source it from the job script.
# Expects the job to be submitted from the repository root.
cd "$PBS_O_WORKDIR" || exit 1
source revision/env/activate.sh

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=8
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-10}"
export nnUNet_def_n_proc=8
export PYTHONUNBUFFERED=1

echo "=== job ${PBS_JOBID:-none} on $(hostname) at $(date '+%F %T')"
echo "CUDA_VISIBLE_DEVICES from scheduler: '${CUDA_VISIBLE_DEVICES-<unset>}'"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# pick_gpus N NEED_GB: indices of N GPUs with at least NEED_GB of free memory.
# This cluster does not restrict a job to "its" GPUs and the GPUs are shared between
# users, so a fixed index (earlier scripts hard-coded GPU 1) can land on a full device,
# which is what produced the CUDNN_STATUS_NOT_INITIALIZED crashes. If the scheduler
# does set CUDA_VISIBLE_DEVICES, that choice is respected. Otherwise the GPUs with
# the most free memory are taken. Prints nothing if not enough GPUs qualify.
pick_gpus() {
  local n="$1" need_mib=$(( ${2:-12} * 1024 )) picked
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | head -n "$n" | paste -sd, -
    return
  fi
  picked="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v need="$need_mib" '{free = $3 - $2; if (free >= need) print free, $1}' \
    | sort -k1,1nr | head -n "$n" | awk '{print $2}' | paste -sd, -)"
  [ "$(echo "$picked" | tr ',' '\n' | grep -c .)" -eq "$n" ] && echo "$picked"
}

# wait_for_gpus N NEED_GB MAX_MINUTES: like pick_gpus, but retries every 2 minutes.
wait_for_gpus() {
  local n="$1" need="$2" max="${3:-120}" waited=0 g
  while :; do
    g="$(pick_gpus "$n" "$need")"
    [ -n "$g" ] && { echo "$g"; return 0; }
    [ "$waited" -ge "$max" ] && return 1
    sleep 120; waited=$(( waited + 2 ))
  done
}
