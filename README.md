# PTO+LLMs
Preliminary experiments using Program Trace Optimization (PTO) + Large Language Models (LLMs).

## Installing LLMs from HuggingFace
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
hf download codellama/CodeLlama-7b-Python-hf --local-dir ./models/CodeLlama-7b-Python
```

### DeepSeek-Coder-V2
From the root of the repository, execute the following to download the model:
```
hf download deepseek-ai/DeepSeek-Coder-V2-Lite-Base --local-dir ./models/DeepSeek-Coder-V2-Lite-Base
```

### Checking the models
Once you complete the installation of the models, you can check them by running the script
```
python3 src/test_llm_models.py
```

## Installing ARC files
First create a subdirectory called `data/` from the root. You can then clone the following GitHub repositories inside the `data/` subdirectory
```
git clone https://github.com/michaelhodel/arc-dsl data/arc-dsl
git clone https://github.com/fchollet/ARC-AGI data/arc-agi-1
```
The first repository contains all examples from the ARC-AGI-1 benchmark, used up to the 2024 competition and later replaced by ARC-AGI-2.