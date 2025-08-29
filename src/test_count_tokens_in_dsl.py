"""
This is another simple test script, to count how many tokens are in Hodel's DSL.
"""
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if __name__ == "__main__":
    # pick a model and load it
    model_path = "models/CodeLlama-7b-Python"
    #model_path = "models/DeepSeek-Coder-V2-Lite-Base" # this one takes more VRAM and is slower

    # TODO: maybe add some decision here on whether to use 8-bit quantization or float16
    # Default configuration
    quantization_config = None
    device_map = "auto"

    # Load tokenizer and model (use float16 for GPU)
    print("Looking for model weights and details in: \"" + model_path + "\"...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device_map,   # puts it on your GPU or CPU, automatically
        quantization_config=quantization_config,
    )

    # read the DSL file
    dsl_file_name = "data/arc-dsl/dsl.py"
    dsl_text = None
    with open(dsl_file_name) as fp:
        dsl_text = fp.read()
    
    # Tokenize input
    inputs = tokenizer(dsl_text, return_tensors="pt").to(model.device)
    print("The DSL file \"" + dsl_file_name + "\" has " + str(inputs.input_ids.shape[1]) + " tokens.")

    # also get the same information for the solvers
    solvers_file_name = "data/arc-dsl/solvers.py"
    solvers_text = None
    with open(solvers_file_name) as fp:
        solvers_text = fp.read()
    inputs = tokenizer(solvers_text, return_tensors="pt").to(model.device)
    print("The solvers file \"" + solvers_file_name + "\" has " + str(inputs.input_ids.shape[1]) + " tokens.")