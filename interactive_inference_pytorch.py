import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Function to measure inference time and generate output
def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True, temperature=0.5)
        end_time = time.time()
    return end_time - start_time, outputs

# Function to calculate metrics
def calculate_metrics(logits, labels, reference, hypothesis):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
    perplexity = torch.exp(loss.mean()).item()

    reference = [reference.split()]
    hypothesis = hypothesis.split()
    bleu_score = sentence_bleu(reference, hypothesis)

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference[0])

    return perplexity, bleu_score, rouge_scores

prompt = "Please write a short story about an adventurous journey."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

reference_text = prompt  # For simplicity, using the prompt as reference
perplexity, bleu_score, rouge_scores = calculate_metrics(outputs, input_ids, reference_text, generated_text)

with open('results_gpu_pytorch.txt', 'w') as f:
    f.write(f"{inference_time_pytorch}\n")
    f.write(f"{perplexity}\n")
    f.write(f"{bleu_score}\n")
    f.write(f"{rouge_scores}\n")
    f.write(f"{generated_text}\n")
