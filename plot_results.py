from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def display_results():
    results_pytorch = pd.read_csv('results_gpu_pytorch.txt', sep='\n', header=None, names=["Result"])
    results_crypten = pd.read_csv('results_gpu_crypten.txt', sep='\n', header=None, names=["Result"])

    # Parsing the results
    def parse_results(results):
        metrics = {
            "Inference time": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "Accuracy": [],
            "Generated text": [],
            "Prompt": []
        }
        for i in range(0, len(results), 7):
            metrics["Inference time"].append(float(results[i].split(": ")[1]))
            metrics["Precision"].append(float(results[i + 1].split(": ")[1]))
            metrics["Recall"].append(float(results[i + 2].split(": ")[1]))
            metrics["F1 Score"].append(float(results[i + 3].split(": ")[1]))
            metrics["Accuracy"].append(float(results[i + 4].split(": ")[1]))
            metrics["Generated text"].append(results[i + 5].split(": ")[1])
            metrics["Prompt"].append(results[i + 6].split(": ")[1])
        return metrics

    metrics_pytorch = parse_results(results_pytorch["Result"])
    metrics_crypten = parse_results(results_crypten["Result"])

    # Create bar plot
    def create_bar_plot():
        labels = ['PyTorch', 'CrypTen']
        inference_times = [sum(metrics_pytorch["Inference time"]) / len(metrics_pytorch["Inference time"]),
                           sum(metrics_crypten["Inference time"]) / len(metrics_crypten["Inference time"])]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, inference_times, color=['blue', 'green'])
        plt.xlabel('Method')
        plt.ylabel('Average Inference Time (s)')
        plt.title('Inference Time Comparison')
        plt.savefig('static/inference_time_comparison.png')

    create_bar_plot()

    # HTML content
    html_content = '''
    <html>
        <body>
            <h1>Inference Time Comparison</h1>
            <img src="/static/inference_time_comparison.png" alt="Inference Time Comparison">
            <h2>Results for PyTorch</h2>
            <pre>{{ pytorch_results }}</pre>
            <h2>Results for CrypTen</h2>
            <pre>{{ crypten_results }}</pre>
        </body>
    </html>
    '''
    return render_template_string(html_content,
                                  pytorch_results=results_pytorch.to_string(index=False),
                                  crypten_results=results_crypten.to_string(index=False))

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0')
