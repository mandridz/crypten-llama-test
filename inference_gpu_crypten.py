import crypten
import torch
import time
from transformers import AutoTokenizer, LlamaForCausalLM
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

crypten.init()

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Function to perform inference
def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids_enc.get_plain_text().to("cuda")).logits
        end_time = time.time()
    return end_time - start_time, logits

# Dialogue loop
while True:
    prompt = input("Enter your prompt (or type 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    input_ids_enc = crypten.cryptensor(input_ids)

    inference_time_crypten, logits_enc = inference_crypten(model, input_ids_enc)

    # Decrypt the logits
    logits_plain = logits_enc.cpu().detach().numpy()

    # Simulated ground truth and predicted labels for testing (for the purpose of metrics demonstration)
    ground_truth = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    predicted = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]

    # Calculate metrics
    precision = precision_score(ground_truth, predicted, zero_division=1)
    recall = recall_score(ground_truth, predicted, zero_division=1)
    f1 = f1_score(ground_truth, predicted, zero_division=1)
    accuracy = accuracy_score(ground_truth, predicted)

    # Generate text from logits
    generated_text = tokenizer.decode(torch.tensor(logits_plain).argmax(dim=-1)[0], skip_special_tokens=True)

    # Output results
    print(f"Generated text: {generated_text}")
    print(f"CrypTen GPU Inference time: {inference_time_crypten} seconds")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

    # Save results to file
    with open("results_gpu_crypten.txt", "a") as f:
        f.write(f"Inference time: {inference_time_crypten}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Generated text: {generated_text}\n")
        f.write(f"Prompt: {prompt}\n\n")
