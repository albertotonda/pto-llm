# PTO+LLMs
Preliminary experiments using Program Trace Optimization (PTO) + Large Language Models (LLMs).

## Installation
This project requires a few tweaks, to install and use a small LLM locally. We recommend creating a new Python environment from scratch. For example, using Anaconda under Windows 10:
```
conda create -n ptollm python=3.10 -y
conda activate ptollm
```
Several packages are needed; pytorch is one, but choose the appropriate version (check the pytorch website).
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Then, all of HuggingFace utilities:
```
pip install transformers accelerate safetensors sentencepiece bitsandbytes huggingface_hub
```
Now, you will need to create a HuggingFace account. Then, go to https://huggingface.co/settings/tokens to generate a new token. Type:
```
hf auth login
```
and cut/paste the token. This will allow you to download the weights of the models.

### CodeLlama-7B
From the root of the repository, execute the following to download the model:
```
huggingface-cli download codellama/CodeLlama-7b-Python-hf --local-dir ./models/CodeLlama-7b-Python
```

### DeepSeek-Coder-V2
From the root of the repository, execute the following to download the model:
```
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-16B-base --local-dir ./models/DeepSeek-Coder-V2-16B
```