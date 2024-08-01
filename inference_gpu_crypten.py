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
class CrypTenLlamaEmbedding(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaEmbedding, self).__init__()
        self.embed_tokens = model.embed_tokens

    def forward(self, input_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        return inputs_embeds

# Encrypt the embedding part of the model
crypten_embedding = CrypTenLlamaEmbedding(model).encrypt()

# Define a wrapper for the rest of the model
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
input_ids_enc = crypten.cryptensor(input_ids)

# Function to measure inference time with CrypTen (partial encryption)
def inference_crypten(crypten_embedding, decoder, input_ids_enc):
    crypten_embedding.eval()
    decoder.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        # Forward pass through the encrypted embedding part
        inputs_embeds_enc = crypten_embedding(input_ids_enc)
        # Decrypt inputs_embeds for the rest of the model
        inputs_embeds = inputs_embeds_enc.get_plain_text().to("cuda")  # Move inputs_embeds to GPU
        # Forward pass through the decoder
        outputs = decoder(inputs_embeds)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Load the decoder separately and keep it unencrypted
decoder = CrypTenLlamaRest(model).to("cuda")  # Move the decoder to GPU

# Measure inference time
inference_time_crypten = inference_crypten(crypten_embedding, decoder, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
