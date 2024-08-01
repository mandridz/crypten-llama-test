import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaModel.from_pretrained(model_name).to("cuda")  # Use LlamaModel instead of AutoModelForCausalLM

# Define a wrapper for the CrypTen model
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        hidden_states = self.model(input_ids=input_ids)[0]
        return hidden_states

# Encrypt the model
crypten_model = CrypTenLlamaModel(model).encrypt()

# Load the decoder separately and keep it unencrypted
decoder = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids)

# Function to measure inference time with CrypTen (partial encryption)
def inference_crypten(crypten_model, decoder, input_ids_enc):
    crypten_model.eval()
    decoder.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        # Forward pass through the encrypted model part
        hidden_states_enc = crypten_model(input_ids_enc)
        # Decrypt hidden states for the decoder
        hidden_states = hidden_states_enc.get_plain_text()
        # Forward pass through the decoder
        outputs = decoder(inputs_embeds=hidden_states).logits
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time

# Measure inference time
inference_time_crypten = inference_crypten(crypten_model, decoder, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
