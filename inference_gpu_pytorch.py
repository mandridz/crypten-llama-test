import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use the Llama-2-7b-hf model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")  # Move the model to GPU

# Read the input data from a file
with open('prompt.txt', 'r') as file:
    input_text = file.read()

# Prepare the input data
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # Move input IDs to GPU

# Function to measure inference time with PyTorch
def inference_pytorch(model, input_ids):
    model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=50)  # Generate output text
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time, outputs

# Measure inference time
inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")
print(f"Generated text: {output_text}")

# Save the results to a file
with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{inference_time_pytorch}\n{output_text}")
