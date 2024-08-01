import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaModel

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use the Llama-2-7b-hf model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaModel.from_pretrained(model_name).to("cuda")  # Move the model to GPU

# Define a wrapper for the CrypTen model
class CrypTenLlamaRest(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaRest, self).__init__()
        self.model = model

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds)
        return outputs.last_hidden_state

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # Move input IDs to GPU

# Compute the embeddings before encryption
with torch.no_grad():
    inputs_embeds = model.embed_tokens(input_ids)

# Encrypt the embeddings
inputs_embeds_enc = crypten.cryptensor(inputs_embeds)

# Encrypt the rest of the model
crypten_rest_model = CrypTenLlamaRest(model).encrypt()

# Function to measure inference time with CrypTen (partial encryption)
def inference_crypten(crypten_rest_model, inputs_embeds_enc):
    crypten_rest_model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        # Forward pass through the encrypted part of the model
        outputs_enc = crypten_rest_model(inputs_embeds_enc)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Measure inference time
inference_time_crypten = inference_crypten(crypten_rest_model, inputs_embeds_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
