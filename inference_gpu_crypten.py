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
class CrypTenLlama(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlama, self).__init__()
        self.model = model

    def forward(self, input_ids):
        # Get embeddings
        embeddings = self.model.embed_tokens(input_ids)

        # Encrypt embeddings
        embeddings_enc = crypten.cryptensor(embeddings)

        # Decrypt embeddings before passing to model
        embeddings_plain = embeddings_enc.get_plain_text()

        # Forward pass through the rest of the model
        outputs = self.model(inputs_embeds=embeddings_plain)
        return outputs.last_hidden_state

# Encrypt the entire model
crypten_model = CrypTenLlama(model).encrypt()

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # Move input IDs to GPU
input_ids_enc = crypten.cryptensor(input_ids)

# Function to measure inference time with CrypTen
def inference_crypten(crypten_model, input_ids_enc):
    crypten_model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        outputs_enc = crypten_model(input_ids_enc)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Measure inference time
inference_time_crypten = inference_crypten(crypten_model, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
