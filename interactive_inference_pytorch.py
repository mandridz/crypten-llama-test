import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True)
        end_time = time.time()
    return end_time - start_time, outputs

while True:
    input_text = input("Enter your prompt: ")
    if input_text.lower() in ['exit', 'quit']:
        break

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

    inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")
    print(f"Generated text: {generated_text}")
