import time
import torch
import crypten
from transformers import AutoTokenizer, AutoModelForCausalLM

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Prepare the input data
input_text = "This is a test input."
input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids, src=0)


# Function to measure inference time with CrypTen (data encrypted)
def inference_crypten(model, input_ids_enc):
    model.eval()
    torch.cuda.synchronize()  # Synchronize GPU before starting the timer
    start_time = time.time()
    with torch.no_grad():
        # Decrypt input_ids for the model (model itself is not encrypted)
        input_ids_plain = input_ids_enc.get_plain_text().long()
        outputs = model(input_ids_plain)
    torch.cuda.synchronize()  # Synchronize GPU after finishing the timer
    end_time = time.time()
    return end_time - start_time


# Measure inference time
inference_time_crypten = inference_crypten(model, input_ids_enc)
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")

# Save the results to a file
with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}")
