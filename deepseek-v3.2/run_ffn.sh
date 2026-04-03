#!/bin/bash

export BATCH_SIZE=48
export ATTN_DP_SIZE=16
export FFN_DP_SIZE=16
export FFN_DP_SIZE_LOCAL=16
export FFN_DP_START_RANK=0
export FFN_DP_HEAD_ADDRESS=0.0.0.0
export AFD_HOST=33.215.116.168
export AFD_PORT=29531

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-size)
            export BATCH_SIZE="$2"
            shift 2
            ;;
        --attn-dp-size)
            export ATTN_DP_SIZE="$2"
            shift 2
            ;;
        --ffn-dp-size)
            export FFN_DP_SIZE="$2"
            shift 2
            ;;
        --ffn-dp-size-local)
            export FFN_DP_SIZE_LOCAL="$2"
            shift 2
            ;;
        --ffn-dp-start-rank)
            export FFN_DP_START_RANK="$2"
            shift 2
            ;;
        --ffn-dp-head-address)
            export FFN_DP_HEAD_ADDRESS="$2"
            shift 2
            ;;
        --afd-host)
            export AFD_HOST="$2"
            shift 2
            ;;
        --afd-port)
            export AFD_PORT="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./run_ffn.sh [options]

Options:
  --batch-size VALUE           Default: 48
  --attn-dp-size VALUE         Default: 16
  --ffn-dp-size VALUE          Default: 16
  --ffn-dp-size-local VALUE    Default: 16
  --ffn-dp-start-rank VALUE    Default: 0
  --ffn-dp-head-address VALUE  Default: 0.0.0.0
  --afd-host VALUE             Default: 33.215.116.168
  --afd-port VALUE             Default: 29531
  -h, --help                   Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$local_ip" ]]; then
    local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

source ~/.bashrc
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash
export PYTHONPATH=/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm:/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm-ascend:$PYTHONPATH

nic_name="eth0"
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export VLLM_VERSION=0.13.0
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT=2
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE=10
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST=1500
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR="/a3_inference/itask/workdir/shared/jcz/profile/attention"
export VLLM_ASCEND_FFN_PROFILER_ENABLE=1
export VLLM_ASCEND_FFN_PROFILER_WAIT=2
export VLLM_ASCEND_FFN_PROFILER_WARMUP=1
export VLLM_ASCEND_FFN_PROFILER_ACTIVE=20
export VLLM_ASCEND_FFN_PROFILER_REPEAT=1
export VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST=1500
export VLLM_ASCEND_FFN_PROFILER_DIR="/a3_inference/itask/workdir/shared/jcz/profile/ffn"

HEADLESS_FLAG=""
if [[ "${FFN_DP_START_RANK}" != "0" ]]; then
    HEADLESS_FLAG="--headless"
fi

vllm serve "/home/admin/model-csi/models/modelhub_35500009_deepseek-v3-2-w8a8-106300046_20260130104046/model" \
    --max-num-batched-tokens $BATCH_SIZE \
    --data-parallel-size ${FFN_DP_SIZE} \
    --data-parallel-size-local ${FFN_DP_SIZE_LOCAL} \
    --data-parallel-address ${FFN_DP_HEAD_ADDRESS} \
    --data-parallel-rpc-port 14435 \
    --data-parallel-start-rank ${FFN_DP_START_RANK} \
    ${HEADLESS_FLAG} \
    --enable_expert_parallel \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --no-enable-prefix-caching \
    --quantization ascend \
    --additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
    --compilation-config '{
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": ['$BATCH_SIZE']
        }' \
    --kv-transfer-config '{
        "kv_connector": "DecodeBenchConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "fill_mean":0.015,
            "fill_std": 0.0
        }
    }' \
    --async-scheduling \
    --afd-config '{
        "afd_connector":"camp2pconnector",
        "afd_role": "ffn",
        "afd_host": "'"${AFD_HOST}"'",
        "afd_port":"'"${AFD_PORT}"'",
        "num_afd_stages":"2",
        "compute_gate_on_attention": "False",
        "afd_extra_config":{
            "afd_size":"'"${ATTN_DP_SIZE}A${FFN_DP_SIZE}F"'"
        }
    }' > ffn.log 2>&1 &
