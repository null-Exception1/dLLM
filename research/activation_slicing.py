import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def sparsify_activations(hidden_states, keep_ratio=0.25):
    tensor = hidden_states.clone()
    
    magnitudes = torch.abs(tensor)
    
    total_elements = tensor.numel()
    k = int(total_elements * keep_ratio)
    
    topk_values, _ = torch.topk(magnitudes.flatten(), k)
    
    threshold = topk_values[-1]
    
    mask = magnitudes >= threshold
    
    return tensor * mask

def make_hook(layer_idx):
    """
    Task 1.2: PyTorch Injection Hook.
    Intercepts and manipulates the data flowing between hidden layers.
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            processed_states = sparsify_activations(hidden_states, keep_ratio=0.25)
            return (processed_states,) + output[1:]
        
        return sparsify_activations(output, keep_ratio=0.25)
    return hook

model_id = "meta-llama/Llama-3.2-1B-Instruct"

print(f"📦 Initializing {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map={"": "cpu"}
)

num_layers = len(model.model.layers)
start_layer = num_layers // 4
end_layer = (3 * num_layers) // 4

print(f"⚡ Injecting Sparsification Hooks into layers {start_layer} through {end_layer}...")
for i in range(start_layer, end_layer):
    model.model.layers[i].register_forward_hook(make_hook(i))
    
prompt = "The future of decentralized computing architecture is based on"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

print("\n🚀 Executing dLLM Activation-Dropped Inference...")
print(f"Prompt: \"{prompt}\"\n" + "-" * 60)

input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask", None)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask, 
        max_new_tokens=50, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9
    )
    
generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
print(f"Engine Output:\n{generated_text}")
print("-" * 60)


