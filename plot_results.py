import matplotlib.pyplot as plt
from flask import Flask, send_file
import io
import pandas as pd

app = Flask(__name__)


def plot_results():
    # Load results
    results = pd.read_csv("results.csv")

    fig, ax = plt.subplots()

    # Plot inference times
    ax.bar(results['Model'], results['Inference Time'], label='Inference Time')
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
