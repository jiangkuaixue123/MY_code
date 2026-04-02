#!/bin/bash
#

pushd .
cd /a3_inference/itask/workdir/shared/jcz/code/umdk1/output/cam/comm_operator/run
./CAM_ascend910_93_openEuler_aarch64.run --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp
cd /a3_inference/itask/workdir/shared/jcz/code/umdk1/output/cam/comm_operator/dist
pip install --force-reinstall umdk_cam_op_lib-208.1.0b1-cp311-cp311-linux_aarch64.whl
popd
cp /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/CAM/op_api/lib/libcust_opapi.so \
    ../vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib