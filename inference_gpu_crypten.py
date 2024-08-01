import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"  # Use a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaModel.from_pretrained(model_name).to("cuda").half()  # Use FP16 for model

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
        return outputs.logits

# Load the decoder separately and keep it unencrypted
decoder = CrypTenLlamaRest(model).to("cuda").half()  # Use FP16 for decoder

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda").half()  # Use FP16 for input_ids
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
        inputs_embeds = inputs_embeds_enc.get_plain_text().half()  # Use FP16 for inputs_embeds
        # Forward pass through the decoder
        outputs = decoder(inputs_embeds)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Measure inference time
inference_time_crypten = inference_crypten(crypten_embedding, decoder, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
