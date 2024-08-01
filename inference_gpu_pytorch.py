import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Load the input prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")


# Function to measure inference time and generate output
def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_new_tokens=50, num_beams=5, early_stopping=True)  # Generate output text
        end_time = time.time()
    return end_time - start_time, outputs


# Perform inference
inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")
print(f"Generated text: {generated_text}")

with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{inference_time_pytorch}\n")
    f.write(generated_text)
