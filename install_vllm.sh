#!/bin/bash
#

pushd .
cd /a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm
git config --global --add safe.directory /a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=v0.13.0 pip install -v -e .
popd

pushd .
cd /a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm-ascend
git config --global --add safe.directory /a3_inference/itask/workdir/hk02335263/jcz_afd_100/code/vllm
pip install -v -e .
popd

bash build_umdk.sh
cp /a3_inference/itask/workdir/shared/jcz/obsutil /usr/bin