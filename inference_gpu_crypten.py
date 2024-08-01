import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaForCausalLM

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use the Llama-2-7b-hf model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")  # Move the model to GPU

# Encrypt the whole model
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits

crypten_model = CrypTenLlamaModel(model).encrypt()

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # Move input IDs to GPU

# Function to measure inference time with CrypTen
def inference_crypten(crypten_model, input_ids):
    crypten_model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        outputs = crypten_model(input_ids)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Measure inference time
inference_time_crypten = inference_crypten(crypten_model, input_ids)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
