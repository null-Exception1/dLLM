import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def sparsify_activations(hidden_states, keep_ratio=1.0):
    tensor = hidden_states.clone()
    
    magnitudes = torch.abs(tensor)
    
    total_elements = tensor.numel()
    k = int(total_elements * keep_ratio)
    
    topk_values, _ = torch.topk(magnitudes.flatten(), k)
    
    threshold = topk_values[-1]
    
    mask = magnitudes >= threshold
    
    return tensor * mask
def calculate_topk_entropy(logits, k=10):
    last_token_logits = logits[0, -1, :]
    
    topk_logits, _ = torch.topk(last_token_logits, k, dim=-1)
    
    probs = torch.softmax(topk_logits, dim=-1)
    
    entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)
    
    return entropy.item()
def make_hook(layer_idx):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            processed_states = sparsify_activations(hidden_states, keep_ratio=0.25)
            return (processed_states,) + output[1:]
        
        return sparsify_activations(output, keep_ratio=0.25)
    return hook

model_id = "meta-llama/Llama-3.2-3B-Instruct"

print(f"📦 Initializing {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_4bit=True, # Compresses the model weights to fit your 3-4GB VRAM
    device_map="cpu"
)

num_layers = len(model.model.layers)
start_layer = num_layers // 4
end_layer = (3 * num_layers) // 4

for i in range(start_layer, end_layer):
    model.model.layers[i].register_forward_hook(make_hook(i))
    
prompt = "whats your opinion on the future of decentralized computing architecture"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

print(f"Prompt: \"{prompt}\"\n" + "-" * 60)

input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

with torch.no_grad():
    outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            max_new_tokens=60, 
            do_sample=True, 
            
            # The Anti-Repetition Stack
            temperature=0.85,          # Slightly higher creativity scale
            top_k=50,                  # Look at top 50 choices
            top_p=0.95,                # Nucleus sampling boundary
            repetition_penalty=1.25,   # Penalize repeating tokens mathematically
            no_repeat_ngram_size=3     # Hard ban on repeating 3-word combinations
        )
    
generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Engine Output:\n{generated_text}")
print("-" * 60)


