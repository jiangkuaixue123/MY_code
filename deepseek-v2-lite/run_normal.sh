#!/bin/bash

source ~/.bashrc

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash

export PYTHONPATH=/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm:/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm-ascend:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=$1
export VLLM_VERSION=0.13.0
export BATCH_SIZE=48
export HCCL_BUFFSIZE=2028
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WAIT=2
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_WARMUP=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_ACTIVE=10
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_REPEAT=1
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_SKIP_FIRST=1500
export VLLM_ASCEND_MODEL_RUNNER_PROFILER_DIR="/mnt/deepseek/jcz/profile/attention"
export VLLM_ASCEND_FFN_PROFILER_ENABLE=1
export VLLM_ASCEND_FFN_PROFILER_WAIT=2
export VLLM_ASCEND_FFN_PROFILER_WARMUP=1
export VLLM_ASCEND_FFN_PROFILER_ACTIVE=20
export VLLM_ASCEND_FFN_PROFILER_REPEAT=1
export VLLM_ASCEND_FFN_PROFILER_SKIP_FIRST=1500
export VLLM_ASCEND_FFN_PROFILER_DIR="/mnt/deepseek/jcz/profile/ffn"

vllm serve "/home/admin/model-csi/models/modelhub_97542_deepseek-v2-lite-36500041_20260318110950/model" \
    --max-num-seqs $BATCH_SIZE \
    --max-model-len 4096 \
    --max-num-batched-tokens $BATCH_SIZE \
    --data-parallel-size=2 \
    --enable_expert_parallel \
    --port 8022 \
    --no-enable-prefix-caching \
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
    > normal.log 2>&1 &