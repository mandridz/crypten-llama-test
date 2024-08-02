from flask import Flask, render_template_string
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def display_results():
    def parse_results(file_path):
        metrics = {
            "Inference time": [],
            "Precision": [],
            "Recall": [],
            "F1 Score": [],
            "Accuracy": [],
            "Generated text": [],
            "Prompt": []
        }
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if "Inference time:" in line:
                    metrics["Inference time"].append(float(line.split(": ")[1].strip()))
                elif "Precision:" in line:
                    metrics["Precision"].append(float(line.split(": ")[1].strip()))
                elif "Recall:" in line:
                    metrics["Recall"].append(float(line.split(": ")[1].strip()))
                elif "F1 Score:" in line:
                    metrics["F1 Score"].append(float(line.split(": ")[1].strip()))
                elif "Accuracy:" in line:
                    metrics["Accuracy"].append(float(line.split(": ")[1].strip()))
                elif "Generated text:" in line:
                    metrics["Generated text"].append(line.split(": ")[1].strip())
                elif "Prompt:" in line:
                    metrics["Prompt"].append(line.split(": ")[1].strip())
        return metrics

    metrics_pytorch = parse_results('results_gpu_pytorch.txt')
    metrics_crypten = parse_results('results_gpu_crypten.txt')

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
    pytorch_results_str = "\n".join([f"{key}: {value}" for key, values in metrics_pytorch.items() for value in values])
    crypten_results_str = "\n".join([f"{key}: {value}" for key, values in metrics_crypten.items() for value in values])

    return render_template_string(html_content,
                                  pytorch_results=pytorch_results_str,
                                  crypten_results=crypten_results_str)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0')
