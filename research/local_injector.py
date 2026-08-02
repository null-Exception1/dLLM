import grpc
import time
import numpy as np
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import activation_stream_pb2 as pb2
import activation_stream_pb2_grpc as pb2_grpc

def run_real_token_onramp():
    print("🛰️ Connecting client injector to Go Entry Shard at localhost:50051...")
    channel = grpc.insecure_channel("localhost:50051")
    client = pb2_grpc.ActivationStreamStub(channel)

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"📦 Loading local tokenizer and context on-ramp embeddings for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model weights in ultra-low memory mode just to use layers 0-5 locally
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # =========================================================
    # PRODUCTION INPUT: THE REAL REASONING SENTENCE
    # =========================================================
    prompt = "what are your thoughts on modern feminism?"
    print(f"\n📝 Processing Input Sentence: '{prompt}'")
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # 1. STEP ONE: Run the local on-ramp layers (0 through 5) to generate true context
    print("🎬 Computing local on-ramp context matrices (Layers 0 -> 5)...")
    with torch.no_grad():
        # Pass raw input IDs through the embedding projection layer
        hidden_states = model.model.embed_tokens(inputs)
        
        # Manually run the first 6 layers locally to establish the linguistic trajectory foundation
        position_ids = torch.arange(0, hidden_states.shape[1], dtype=torch.long, device="cuda").unsqueeze(0)
        for i in range(0, 6):
            layer_outputs = model.model.layers[i](hidden_states, position_ids=position_ids)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

    # 2. STEP TWO: Extract the real, high-precision Float16 vector bytes
    print(f"⚡ Extracted True Linguistic Matrix! Shape: {list(hidden_states.shape)} | Channels: {hidden_states.shape[-1]}")
    real_tensor_bytes = hidden_states.detach().cpu().to(torch.float16).numpy().tobytes()
    mock_bitmask = bytes([0xFF] * 384) # 3,072 channels bounded

    # 3. STEP THREE: Blast the true semantic state directly into the Go data plane!
    payload = pb2.ActivationPayload(
        prompt_hash=77777,
        task_serial_number=1, # Branch Trajectory 1
        layer_index=6,        # Targeting Shard entry at Layer 6 gateway!
        quantized_tensor=real_tensor_bytes,
        sparse_bitmask=mock_bitmask,
        token_ids=inputs[0].cpu().numpy().tolist(),
        cumulative_logprob=-1.2
    )

    try:
        print("🚀 Blasting genuine linguistic tensor matrix into Go Port :50051...")
        response = client.StreamLayerActivations(payload)
        if response.success:
            print("\n" + "="*60)
            print("🎉 SUCCESS! Cluster processed real text matrix!")
            if response.generated_token:
                print(f"👉 Returned Word from Mesh: '{response.generated_token}'")
            print("="*60)
    except Exception as e:
        print(f"❌ Cluster Transport Drop: {e}")

if __name__ == "__main__":
    run_real_token_onramp()
