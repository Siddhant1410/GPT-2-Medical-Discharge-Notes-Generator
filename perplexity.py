import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def calculate_perplexity(text, model, tokenizer):
    """
    Calculate perplexity of text using a given model and tokenizer.
    
    :param text: Input text string.
    :param model: Pre-trained GPT-2 model.
    :param tokenizer: GPT-2 tokenizer.
    :return: Perplexity value.
    """
    
    # Tokenize input and obtain log likelihood
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    log_likelihood = outputs[0].item()
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(log_likelihood / len(input_ids[0])))
    
    return perplexity.item()

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example usage
text = "The Quick Brown Fox Jumps Over The Lazy Dog."
perplexity_value = calculate_perplexity(text, model, tokenizer)

print(f"Perplexity: {perplexity_value}")

