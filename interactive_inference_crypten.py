import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaForCausalLM

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Define a wrapper for the CrypTen model
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        input_ids_plain = input_ids.get_plain_text().long()
        embeddings = self.model.get_input_embeddings()(input_ids_plain)
        embeddings_enc = crypten.cryptensor(embeddings)
        outputs = self.model(inputs_embeds=embeddings_enc.get_plain_text())
        return outputs.logits

crypten_model = CrypTenLlamaModel(model)

def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits_enc = model(input_ids_enc)
        end_time = time.time()
    return end_time - start_time, logits_enc

while True:
    input_text = input("Enter your prompt: ")
    if input_text.lower() in ['exit', 'quit']:
        break

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    input_ids_enc = crypten.cryptensor(input_ids)

    inference_time_crypten, logits_enc = inference_crypten(crypten_model, input_ids_enc)
    outputs_enc_plain = logits_enc.get_plain_text()
    outputs = torch.argmax(outputs_enc_plain, dim=-1)

    generated_text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)

    print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")
    print(f"Generated text: {generated_text}")
