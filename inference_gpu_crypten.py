import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaForCausalLM

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids_enc.get_plain_text().to("cuda")).logits
        end_time = time.time()
    return end_time - start_time, logits

# Load prompt from file
with open("prompt.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

if not input_text:
    raise ValueError('No input text provided')

input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids)

inference_time_crypten, logits_enc = inference_crypten(model, input_ids_enc)
logits_plain = logits_enc.argmax(dim=-1).tolist()
generated_text = tokenizer.decode(logits_plain[0], skip_special_tokens=True)

# Save results to file
with open('results_gpu_crypten.txt', 'w', encoding="utf-8") as f:
    f.write(f"Inference time: {inference_time_crypten}\n")
    f.write(f"Generated text: {generated_text}\n")

print("Results saved to results_gpu_crypten.txt")
