import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from flask import Flask, render_template_string
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import torch.nn.functional as F

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Function to measure inference time and generate output
def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True)
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

# Flask app to display results
app = Flask(__name__)

@app.route('/')
def index():
    prompt = "Please write a short story about an adventurous journey."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reference_text = prompt  # For simplicity, using the prompt as reference
    perplexity, bleu_score, rouge_scores = calculate_metrics(outputs, input_ids, reference_text, generated_text)

    html = f"""
    <h1>PyTorch Inference Results</h1>
    <p><strong>Prompt:</strong> {prompt}</p>
    <p><strong>Generated Text:</strong> {generated_text}</p>
    <p><strong>Inference Time:</strong> {inference_time_pytorch} seconds</p>
    <p><strong>Perplexity:</strong> {perplexity}</p>
    <p><strong>BLEU Score:</strong> {bleu_score}</p>
    <p><strong>ROUGE Scores:</strong> {rouge_scores}</p>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True)
