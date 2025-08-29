"""
Another test script, similar to test_llm_models.py, but generates output token by token.
"""
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def generate_token(model, context_ids, prng, temperature=1.0):
    with torch.no_grad(): # don't compute gradients
        outputs = model(context_ids)
        logits = outputs.logits[:, -1, :]  # [batch, vocab_size]

        # convert logits to probabilities
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1).squeeze(0).cpu().numpy()

        # sort tokens by (descending) probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # some printout debugging here
        #print("There are " + str(len(sorted_probs)) + " possible tokens for the next position; probabilities of the top 10 are:"
        #      + str(sorted_probs[:10]))

        return prng.choice(sorted_indices, p=sorted_probs)

if __name__ == "__main__":
    
    # seed the random number generator
    prng = np.random.default_rng(seed=42)

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

    # Example prompt
    prompt = """# Write a Python function that computes the Fibonacci sequence up to n."""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get ids of all tokens in the input window
    context_ids = inputs['input_ids']

    next_token = None
    
    while next_token != tokenizer.eos_token_id:
        next_token = generate_token(model, context_ids, prng)
        next_token_tensor = torch.tensor([[next_token]], device=model.device)
        context_ids = torch.cat([context_ids, next_token_tensor], dim=1)
        print("Generated token: " + tokenizer.decode(next_token) + " (id=" + str(next_token) + ")")
        print("Current output: " + tokenizer.decode(context_ids[0], skip_special_tokens=True))
        print("---")

    # seed the random number generator
    prng = np.random.default_rng(seed=42)
