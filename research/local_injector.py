import grpc
import time
import numpy as np
import sys
import os

# Ensure Python can see your compiled protobuf drivers inside the research directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    import activation_stream_pb2 as pb2
    import activation_stream_pb2_grpc as pb2_grpc
except ImportError:
    print("❌ Error: Missing compiled proto files. Run your protoc command from the repository root first!")
    sys.exit(1)

def run_injection_test():
    print("🛰️ Connecting local client injector to Go Entry Shard at localhost:50051...")
    channel = grpc.insecure_channel("localhost:50051")
    client = pb2_grpc.ActivationStreamStub(channel)

    # Simulating sequential layers (6 through 11) to watch the Go Hash Ring deflect targets
    for layer in range(6, 12):
        print(f"\n🎬 Generating synthetic layer activations for Target Layer {layer}...")
        
        # Simulating a 35% sliced Float16 token vector (1,075 non-zero elements -> ~2,150 bytes)
        # FIX: Change from 1075 to 3072 so the matrix dimensions match Llama's brain exactly
        mock_floats = np.random.randn(3072).astype(np.float16)
        raw_tensor_bytes = mock_floats.tobytes()

        # Squeezing a dense 384-byte bitmask layer matching Llama's 3,072-channel dimension bounds
        mock_bitmask = bytes([0b10101010] * 384)

        # Constructing the structural payload matching your compiled Go data contracts
        payload = pb2.ActivationPayload(
            prompt_hash=99999,       # Fixed prompt identifier key to test ring routing stability
            task_serial_number=2,    # Simulating Speculative Branch Trajectory #2
            layer_index=layer,
            global_min=-1.0,
            global_max=1.0,
            scale_factor=1.0,
            quantized_tensor=raw_tensor_bytes,
            sparse_bitmask=mock_bitmask
        )

        try:
            print(f"🚀 Blasting network packet for Layer {layer} into port :50051...")
            response = client.StreamLayerActivations(payload)
            if response.success:
                print(f"✅ Packet for Layer {layer} accepted smoothly by the cluster data plane.")
        except Exception as e:
            print(f"❌ Transport Drop on Layer {layer}: {e}")
            
        time.sleep(1.2) # Adding a tiny breathing gap so you can watch logs flash cross your tabs!

if __name__ == "__main__":
    run_injection_test()
