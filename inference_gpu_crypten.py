import time
import torch
import crypten
import crypten.nn as cnn
from transformers import LlamaForCausalLM, LlamaTokenizer

crypten.init()

# Загрузка модели и токенизатора
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Зашифрование модели
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids=input_ids).logits

crypten_model = CrypTenLlamaModel(model).encrypt()

# Подготовка данных
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids)

# Инференс и замер времени
def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids_enc)
        end_time = time.time()
    return end_time - start_time

inference_time_crypten = inference_crypten(crypten_model, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
