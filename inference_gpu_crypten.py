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


# Define a wrapper for the CrypTen model
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        # Decrypt input_ids before passing to model
        input_ids_plain = input_ids.get_plain_text().long()  # Ensure the tensor is of type Long

        # Forward pass through the embedding layer
        embeddings = self.model.get_input_embeddings()(input_ids_plain)

        # Encrypt the embeddings
        embeddings_enc = crypten.cryptensor(embeddings)

        # Forward pass through the rest of the model
        outputs = self.model(inputs_embeds=embeddings_enc.get_plain_text())

        # Encrypt the outputs
        outputs_enc = crypten.cryptensor(outputs.logits)
        return outputs_enc


crypten_model = CrypTenLlamaModel(model).encrypt()

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # Move input IDs to GPU
input_ids_enc = crypten.cryptensor(input_ids)

# Function to measure inference
