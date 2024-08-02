import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template_string

app = Flask(__name__)

# Create the static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Load results from files
results = {}
with open('results_gpu_pytorch.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    results["PyTorch"] = {
        "inference_time": float(lines[0].strip().split(": ")[1]),
        "generated_text": lines[1].strip().split(": ")[1]
    }

with open('results_gpu_crypten.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    results["CrypTen"] = {
        "inference_time": float(lines[0].strip().split(": ")[1]),
        "generated_text": lines[1].strip().split(": ")[1]
    }


# Create a bar plot
def create_bar_plot():
    labels = list(results.keys())
    inference_times = [results[label]["inference_time"] for label in labels]

    fig, ax = plt.subplots()
    ax.bar(labels, inference_times, color=['blue', 'green'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Inference Time Comparison')

    # Save plot to a file
    plt.savefig('static/inference_time_comparison.png')


# Route for displaying the results
@app.route('/')
def display_results():
    create_bar_plot()
    results_html = ""
    for model, data in results.items():
        results_html += f"<h2>{model}</h2>"
        results_html += f"<p><b>Inference Time:</b> {data['inference_time']} seconds</p>"
        results_html += f"<p><b>Generated Text:</b> {data['generated_text']}</p>"

    html = f"""
    <html>
    <head><title>Inference Results</title></head>
    <body>
        <h1>Inference Results</h1>
        {results_html}
        <h2>Inference Time Comparison</h2>
        <img src="/static/inference_time_comparison.png" alt="Inference Time Comparison">
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
