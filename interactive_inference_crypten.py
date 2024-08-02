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

# Function to measure inference time
def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        input_ids_plain = input_ids_enc.get_plain_text().long()  # Decrypt input_ids
        output = model.generate(input_ids_plain, max_length=500, num_beams=5, early_stopping=True, temperature=0.5)
        end_time = time.time()
    return end_time - start_time, output

# Interactive loop for inference
while True:
    prompt = input("Enter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break

    # Prepare the input data
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    input_ids_enc = crypten.cryptensor(input_ids)

    # Perform inference with CrypTen
    inference_time_crypten, outputs = inference_crypten(model, input_ids_enc)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")
    print(f"Generated text: {generated_text}")
