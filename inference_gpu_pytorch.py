import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")


# Function to measure inference time with PyTorch
def inference_pytorch(model, input_ids):
    model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time


# Measure inference time
inference_time_pytorch = inference_pytorch(model, input_ids)
print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")

# Save the results to a file
with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{inference_time_pytorch}")
