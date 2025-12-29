echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
# 设置PYTHONPATH

cd /home/lws/sglang_eagle3
export PYTHONPATH=${PWD}/python:$PYTHONPATH

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash

#export ENABLE_PROFILING=1
#export PROFILING_BS=96

#export TASK_QUEUE_ENABLE=2

export SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

#export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export HCCL_OP_EXPANSION_MODE="AIV"
#
# tp2
python3 -m sglang.launch_server \
--model-path /home/weights/Qwen3-30B-A3B-Instruct-2507 \
--trust-remote-code \
--attention-backend ascend \
--mem-fraction-static 0.9 \
--disable-radix-cache \
--chunked-prefill-size 32768 \
--cuda-graph-bs 1 12 36 72 96 132 \
--max-running-requests 132 \
--disable-radix-cache \
--warmup 10 \
--tp-size 2 \
--dp-size 1 \
--nnodes 1 \
--node-rank 0 \
--base-gpu-id 0 \
--sampling-backend ascend \
--host 141.61.41.151 \
--speculative-algorithm SUFFIX \
--speculative-num-draft-tokens 15 \
--port 6698

exit 1
vllm bench serve --backend openai-chat --model Qwen3-30B-A3B-Instruct-2507 --tokenizer /home/weights/Qwen3-30B-A3B-Instruct-2507 --served-model-name Qwen3-30B-A3B-Instruct-2507 --dataset-name random  --random-input-len 1024 --random-output-len 100 --request-rate 13 --num-prompts 1000 --endpoint /v1/chat/completions --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --host 141.61.41.151 --port 6698 --seed 1000 --temperature 0.01
