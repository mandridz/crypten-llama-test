import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")


def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True, temperature=0.5)
        end_time = time.time()
    return end_time - start_time, outputs


# Load prompt from file
with open("prompt.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

if not input_text:
    raise ValueError('No input text provided')

input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save results to file
with open('results_gpu_pytorch.txt', 'w', encoding="utf-8") as f:
    f.write(f"Inference time: {inference_time_pytorch}\n")
    f.write(f"Generated text: {generated_text}\n")

print("Results saved to results_gpu_pytorch.txt")
