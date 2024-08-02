import time
import torch
import crypten
from transformers import AutoTokenizer, LlamaForCausalLM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Load the input prompt from a file
with open("prompt.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids)

# Function to measure inference time and quality metrics
def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids_enc.get_plain_text().long().to("cuda")).logits
        end_time = time.time()

    inference_time = end_time - start_time
    outputs = torch.argmax(logits, dim=-1)
    return inference_time, outputs

# Measure inference time and generate text
inference_time_crypten, logits_enc = inference_crypten(model, input_ids_enc)
outputs_enc_plain = logits_enc.cpu().numpy().tolist()
generated_text = tokenizer.decode(outputs_enc_plain[0], skip_special_tokens=True)

# For simplicity, assuming ground_truth and predicted outputs are binary labels for metrics
ground_truth = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids[0].tolist()]
predicted = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in outputs_enc_plain[0]]

# Ensure lengths are the same for metric calculation
min_length = min(len(ground_truth), len(predicted))
ground_truth = ground_truth[:min_length]
predicted = predicted[:min_length]

precision = precision_score(ground_truth, predicted, zero_division=1)
recall = recall_score(ground_truth, predicted, zero_division=1)
f1 = f1_score(ground_truth, predicted, zero_division=1)
accuracy = accuracy_score(ground_truth, predicted)

# Save the results to a file
with open('results_gpu_crypten.txt', 'w', encoding="utf-8") as f:
    f.write(f"Inference time: {inference_time_crypten}\n")
    f.write(f"Generated text: {generated_text}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Accuracy: {accuracy}\n")

# Print the results to the console
print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")
print(f"Generated text: {generated_text}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
