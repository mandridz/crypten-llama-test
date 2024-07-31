import time
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Загрузка модели и токенизатора
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Подготовка данных
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")


# Инференс и замер времени
def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model(input_ids)

        print(outputs)

        end_time = time.time()
    return end_time - start_time


inference_time_pytorch = inference_pytorch(model, input_ids)
print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")

with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{inference_time_pytorch}")
