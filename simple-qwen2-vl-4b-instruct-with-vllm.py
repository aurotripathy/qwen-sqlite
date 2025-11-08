
# from https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#consume-the-openai-api-compatible-server


# GPU memory usage:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce GTX TITAN X     Off |   00000000:06:00.0  On |                  N/A |
# | 22%   53C    P0             75W /  250W |     676MiB /  12288MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+
# |   1  NVIDIA TITAN RTX               Off |   00000000:0A:00.0 Off |                  N/A |
# | 41%   42C    P8             14W /  280W |   21062MiB /  24576MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+
# |   2  NVIDIA TITAN RTX               Off |   00000000:0B:00.0 Off |                  N/A |
# | 41%   33C    P8              9W /  280W |   21062MiB /  24576MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+


# expected output:
# (qwen-sqlite) (base) auro@auro:~/qwen-sqlite$ python simple-vll.py 
# Response costs: 20.85s
# Generated text: AuntieAnne's
# CINNAMON SUGAR
# 1 x 17,000
# 17,000
# SUB TOTAL
# 17,000
# GRAND TOTAL
# 17,000
# CASH IDR
# 20,000
# CHANGE DUE
# 3,000


# vLLM server invocation:
# CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-VL-4B-Instruct   --tensor-parallel-size 2   --max-model-len 60000   --async-scheduling --limit-mm-per-prompt.video 0 


import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
    timeout=3600
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
                }
            },
            {
                "type": "text",
                "text": "Read all the text in the image."
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-4B-Instruct",
    messages=messages,
    max_tokens=2048
)
print(f"Response costs: {time.time() - start:.2f}s")
print(f"Generated text: {response.choices[0].message.content}")