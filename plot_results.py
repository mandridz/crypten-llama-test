import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, send_file, render_template_string

app = Flask(__name__)

# Загрузка данных
results = {}
files = ['results_gpu_pytorch.txt', 'results_gpu_crypten.txt']
labels = ['PyTorch GPU', 'CrypTen GPU']

for file, label in zip(files, labels):
    with open(file, 'r') as f:
        data = f.read().strip()
        results[label] = float(data)

# Данные для графиков
categories = ['Inference Time']
n_categories = len(categories)
index = np.arange(n_categories)

# Построение графиков
fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

bars = []
for i, label in enumerate(labels):
    bar = plt.bar(index + i * bar_width, [results[label]], bar_width, alpha=opacity, label=label)
    bars.append(bar)

plt.xlabel('Metrics')
plt.ylabel('Time (seconds)')
plt.title('Inference Time Comparison')
plt.xticks(index + bar_width / 2, categories)
plt.legend()

# Добавление значений на барграфы
for bar in bars:
    for rect in bar:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points',
                     ha='center', va='bottom')

plt.tight_layout()

# Сохранение графика в файл
output_image = 'results_plot.png'
plt.savefig(output_image)
plt.close(fig)


# Flask routes
@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Inference Time Comparison</title>
    </head>
    <body>
        <h1>Inference Time Comparison</h1>
        <img src="/plot.png" alt="Inference Time Comparison">
    </body>
    </html>
    ''')


@app.route('/plot.png')
def plot_png():
    return send_file(output_image, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
