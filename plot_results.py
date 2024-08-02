import os
import matplotlib.pyplot as plt
from flask import Flask, send_file, render_template_string
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    return render_template_string("""
        <h1>Inference Time Comparison</h1>
        <img src="/plot.png" alt="Inference Time Comparison">
    """)


@app.route('/plot.png')
def plot_png():
    # Read results from files
    pytorch_times = []
    with open('results_pytorch.txt', 'r') as f:
        pytorch_times = [float(line.strip()) for line in f.readlines()]

    crypten_times = []
    with open('results_crypten.txt', 'r') as f:
        crypten_times = [float(line.strip()) for line in f.readlines()]

    # Ensure we have the same number of entries
    min_len = min(len(pytorch_times), len(crypten_times))
    pytorch_times = pytorch_times[:min_len]
    crypten_times = crypten_times[:min_len]

    # Plotting the results
    plt.figure(figsize=(10, 5))
    labels = [f'Run {i + 1}' for i in range(min_len)]
    x = range(min_len)

    plt.bar(x, pytorch_times, width=0.4, label='PyTorch', align='center')
    plt.bar(x, crypten_times, width=0.4, label='CrypTen', align='edge')
    plt.xlabel('Run')
    plt.ylabel('Inference Time (s)')
    plt.title('Inference Time Comparison: PyTorch vs CrypTen')
    plt.legend()

    # Adding values on top of the bars
    for i in x:
        plt.text(i, pytorch_times[i] + 0.01, f'{pytorch_times[i]:.2f}', ha='center')
        plt.text(i + 0.4, crypten_times[i] + 0.01, f'{crypten_times[i]:.2f}', ha='center')

    # Save plot to a PNG file
    plot_path = '/tmp/plot.png'
    plt.savefig(plot_path)
    plt.close()

    return send_file(plot_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
