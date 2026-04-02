name=$1
user=wentiange
#image=hcr.met:a-wulan01.hw-wulan.local/antsys/vllm:release_0.18.0_0415_202603221557_aarch64
image=hcr.meta-wulan01.hw-wulan.local/antsys/vllm:CANN85RC1_202602011815_aarch64
#itask create --name ${name} --user ${user} --image ${image} --hostnet --4card --skip-sync --type a3
itask create --name ${name} --user ${user} --image ${image} --hostnet --16card --skip-sync --type a3