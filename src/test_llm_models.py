import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    # TODO: check if model path exists
    #model_path = "models/CodeLlama-7b-Python"
    model_path = "models/DeepSeek-Coder-V2-Lite-Base" # this one takes more VRAM

    print("Looking for model weights and details in: \"" + model_path + "\"...")

    # TODO: maybe add some decision here on whether to use 8-bit quantization or float16
    # Default configuration
    quantization_config = None
    device_map = "auto"

    if "DeepSeek" in model_path:
        print("Loading DeepSeek model, using 8-bit quantization and forcing everything on GPU...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            #llm_int8_enable_fp32_cpu_offload=True, # I could also try to force everything on GPU
            )
        device_map = {"": 0}  # force everything on GPU 0

    # Load tokenizer and model (use float16 for GPU)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,   # puts it on your GPU automatically
        quantization_config=quantization_config,
    )

    # Example prompt
    prompt = """# Write a Python function that computes the Fibonacci sequence up to n."""
    
    # Tokenize input
    t_start_tokenization = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t_end_tokenization = time.time()
    print(f"Tokenization took {t_end_tokenization - t_start_tokenization:.4f} seconds")

    # Generate
    t_start_generation = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=20000,
        do_sample=True,
        temperature=0.2,
        top_p=0.9
    )
    t_end_generation = time.time()
    print(f"Generation took {t_end_generation - t_start_generation:.2f} seconds")

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
