import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Function to perform inference
def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True)
        end_time = time.time()
    return end_time - start_time, outputs

# Dialogue loop
while True:
    prompt = input("Enter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Simulated ground truth and predicted labels for testing (for the purpose of metrics demonstration)
    ground_truth = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    predicted = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]

    # Calculate metrics
    precision = precision_score(ground_truth, predicted, zero_division=1)
    recall = recall_score(ground_truth, predicted, zero_division=1)
    f1 = f1_score(ground_truth, predicted, zero_division=1)
    accuracy = accuracy_score(ground_truth, predicted)

    # Output results
    print(f"Generated text: {generated_text}")
    print(f"PyTorch GPU Inference time: {inference_time_pytorch} seconds")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    # Save results to file
    with open("results_gpu_pytorch.txt", "a") as f:
        f.write(f"Inference time: {inference_time_pytorch}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Generated text: {generated_text}\n")
        f.write(f"Prompt: {prompt}\n\n")
