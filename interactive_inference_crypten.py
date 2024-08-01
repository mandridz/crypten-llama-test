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
        outputs_enc = crypten.cryptensor(outputs.logits)
        return outputs_enc

crypten_model = CrypTenLlamaModel(model).encrypt()

def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs_enc = model(input_ids_enc)
        end_time = time.time()
    return end_time - start_time, outputs_enc.get_plain_text()

while True:
    input_text = input("Enter your prompt: ")
    if input_text.lower() in ['exit', 'quit']:
        break

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    input_ids_enc = crypten.cryptensor(input_ids)

    inference_time_crypten, outputs_enc_plain = inference_crypten(crypten_model, input_ids_enc)
    outputs_enc_plain_list = outputs_enc_plain.tolist()

    generated_text = tokenizer.decode(outputs_enc_plain_list[0], skip_special_tokens=True)

    print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")
    print(f"Generated text: {generated_text}")
