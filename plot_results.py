import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template_string

# Read results
results = {}
files = ['results_gpu_pytorch.txt', 'results_gpu_crypten.txt']
labels = ['PyTorch', 'CrypTen']

for label, file in zip(labels, files):
    with open(file, 'r') as f:
        data = f.readlines()
        inference_time = float(data[0].strip())
        perplexity = float(data[1].strip())
        bleu_score = float(data[2].strip())
        rouge_scores = eval(data[3].strip())
        generated_text = data[4].strip()

        results[label] = {
            'Inference Time': inference_time,
            'Perplexity': perplexity,
            'BLEU Score': bleu_score,
            'ROUGE Scores': rouge_scores,
            'Generated Text': generated_text
        }

# Flask app to display results
app = Flask(__name__)

@app.route('/')
def index():
    html = """
    <h1>Model Inference Results</h1>
    {% for label, metrics in results.items() %}
        <h2>{{ label }}</h2>
        <p><strong>Inference Time:</strong> {{ metrics['Inference Time'] }} seconds</p>
        <p><strong>Perplexity:</strong> {{ metrics['Perplexity'] }}</p>
        <p><strong>BLEU Score:</strong> {{ metrics['BLEU Score'] }}</p>
        <p><strong>ROUGE Scores:</strong> {{ metrics['ROUGE Scores'] }}</p>
        <p><strong>Generated Text:</strong> {{ metrics['Generated Text'] }}</p>
        <hr>
    {% endfor %}
    """
    return render_template_string(html, results=results)

if __name__ == '__main__':
    app.run(debug=True)
