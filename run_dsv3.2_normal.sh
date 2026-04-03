#!/bin/sh

# this obtained through ifconfig
# nic_name is the network interface name corresponding to local_ip of the current node
nic_name="eth0"
local_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "$local_ip"
# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is install on your machine, you can turn it on.
# export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# AIV
export PYTHONPATH=/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm:/a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm-ascend:$PYTHONPATH

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=200
# export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /home/admin/model-csi/models/modelhub_35500009_deepseek-v3-2-w8a8-106300046_20260130104046/model \
--host 0.0.0.0 \
--port 8033 \
--data-parallel-size 4 \
--data-parallel-size-local 4 \
--data-parallel-address 0.0.0.0 \
--data-parallel-start-rank 0 \
--data-parallel-rpc-port 14435 \
--tensor-parallel-size 4 \
--quantization ascend \
--seed 1024 \
--served-model-name deepseek_v3_2 \
--enable-expert-parallel \
--max-num-seqs 16 \
--max-model-len 8192 \
--max-num-batched-tokens 4096 \
--trust-remote-code \
--no-enable-prefix-caching \
--gpu-memory-utilization 0.92 \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"layer_sharding": ["q_b_proj", "o_proj"]}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}'