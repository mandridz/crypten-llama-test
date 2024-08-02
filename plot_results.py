import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import pandas as pd

app = Flask(__name__)


def plot_results():
    # Load results
    results_pytorch = pd.read_csv("results_gpu_pytorch.txt", sep=": ", header=None, engine='python').set_index(0).T
    results_crypten = pd.read_csv("results_gpu_crypten.txt", sep=": ", header=None, engine='python').set_index(0).T

    fig, ax = plt.subplots()

    # Plot inference times
    models = ['PyTorch', 'CrypTen']
    times = [float(results_pytorch['Inference time']), float(results_crypten['Inference time'])]
    ax.bar(models, times, label='Inference Time')
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Inference Time Comparison')
    ax.legend()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img


@app.route('/')
def display_plot():
    img = plot_results()
    return send_file(img, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
