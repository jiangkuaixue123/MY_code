#!/bin/bash

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
# nic_name="eth0"
# local_ip="33.215.116.168"
data_parallel_size="2"
data_parallel_size_local="2"
data_parallel_start_rank="0"
data_parallel_head_address="0.0.0.0"
tensor_parallel_size="8"
batch_size="16"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --data-parallel-size)
            data_parallel_size="$2"
            shift 2
            ;;
        --data-parallel-size-local)
            data_parallel_size_local="$2"
            shift 2
            ;;
        --data-parallel-start-rank)
            data_parallel_start_rank="$2"
            shift 2
            ;;
        --data-parallel-head-address)
            data_parallel_head_address="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            tensor_parallel_size="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_normal.sh [options]

Options:
  --batch-size VALUE             Default: 16
  --data-parallel-size VALUE     Default: 2
  --data-parallel-size-local VALUE  Default: 2
  --data-parallel-start-rank VALUE  Default: 0
  --data-parallel-head-address VALUE Default: 0.0.0.0
  --tensor-parallel-size VALUE   Default: 8
  -h, --help                     Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export PYTHONPATH=/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm:/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm-ascend:$PYTHONPATH

# export HCCL_IF_IP=$local_ip
# export GLOO_SOCKET_IFNAME=$nic_name
# export TP_SOCKET_IFNAME=$nic_name
# export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

HEADLESS_FLAG=""
if [[ "${data_parallel_start_rank}" != "0" ]]; then
    HEADLESS_FLAG="--headless"
fi

vllm serve /home/admin/model-csi/models/modelhub_35500009_deepseek-v3-2-w8a8-106300046_20260130104046/model \
--host 0.0.0.0 \
--port 8033 \
--data-parallel-size "$data_parallel_size" \
--data-parallel-size-local "$data_parallel_size_local" \
--data-parallel-address "$data_parallel_head_address" \
--data-parallel-rpc-port 14435 \
--data-parallel-start-rank "$data_parallel_start_rank" \
${HEADLESS_FLAG} \
--tensor-parallel-size "$tensor_parallel_size" \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3_2 \
--enable-expert-parallel \
--max-num-seqs "$batch_size" \
--max-model-len 8192 \
--max-num-batched-tokens "$batch_size" \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": ['$batch_size']}' \
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}'
# --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'
