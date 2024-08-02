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
        "generated_text": lines[1].strip().split(": ", 1)[1],
        "precision": float(lines[2].strip().split(": ")[1]),
        "recall": float(lines[3].strip().split(": ")[1]),
        "f1_score": float(lines[4].strip().split(": ")[1]),
        "accuracy": float(lines[5].strip().split(": ")[1]),
    }

with open('results_gpu_crypten.txt', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    results["CrypTen"] = {
        "inference_time": float(lines[0].strip().split(": ")[1]),
        "generated_text": lines[1].strip().split(": ", 1)[1],
        "precision": float(lines[2].strip().split(": ")[1]),
        "recall": float(lines[3].strip().split(": ")[1]),
        "f1_score": float(lines[4].strip().split(": ")[1]),
        "accuracy": float(lines[5].strip().split(": ")[1]),
    }

# Print results to console
for model, data in results.items():
    print(f"{model} Inference Time: {data['inference_time']} seconds")
    print(f"{model} Generated Text: {data['generated_text']}")
    print(f"{model} Precision: {data['precision']}")
    print(f"{model} Recall: {data['recall']}")
    print(f"{model} F1 Score: {data['f1_score']}")
    print(f"{model} Accuracy: {data['accuracy']}")


# Create bar plots
def create_bar_plots():
    labels = list(results.keys())

    # Inference Time
    inference_times = [results[label]["inference_time"] for label in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, inference_times, color=['blue', 'green'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Inference Time Comparison')
    plt.savefig('static/inference_time_comparison.png')

    # Precision
    precision_scores = [results[label]["precision"] for label in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, precision_scores, color=['blue', 'green'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Comparison')
    plt.savefig('static/precision_comparison.png')

    # Recall
    recall_scores = [results[label]["recall"] for label in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, recall_scores, color=['blue', 'green'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Recall')
    ax.set_title('Recall Comparison')
    plt.savefig('static/recall_comparison.png')

    # F1 Score
    f1_scores = [results[label]["f1_score"] for label in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, f1_scores, color=['blue', 'green'])
    ax.set_xlabel('Model')
    ax.set_ylabel
