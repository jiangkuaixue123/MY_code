#!/bin/bash

source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/bin/set_env.bash
# export ASCEND_RT_VISIBLE_DEVICES=$1
export VLLM_VERSION=0.13.0
export BATCH_SIZE=48
export HCCL_BUFFSIZE=2048
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
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

vllm serve "/home/admin/model-csi/models/modelhub_35500009_deepseek-v3-2-w8a8-106300046_20260130104046/model" \
    --max-num-seqs $BATCH_SIZE \
    --max-model-len 4096 \
    --max-num-batched-tokens $BATCH_SIZE \
    --data-parallel-size=16 \
    --enable-dbo \
    --dbo-prefill-token-threshold 12 \
    --dbo-decode-token-threshold 2 \
    --enable_expert_parallel \
    --port 8033 \
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
        "afd_role": "attention",
        "afd_host": "33.215.116.168",
        "afd_port":"29531",
        "num_afd_stages":"2",
        "compute_gate_on_attention": "False",
        "afd_extra_config":{
            "afd_size":"16A16F"
        }
    }' > attn.log 2>&1 &