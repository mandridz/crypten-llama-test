import matplotlib.pyplot as plt

# Initialize dictionaries to store results
results = {}

# Read PyTorch results
with open('results_pytorch.txt', 'r') as f:
    times = f.readlines()
    results['PyTorch'] = [float(time.strip()) for time in times]

# Read CrypTen results
with open('results_crypten.txt', 'r') as f:
    times = f.readlines()
    results['CrypTen'] = [float(time.strip()) for time in times]

# Plot results
labels = list(results.keys())
times = [results[label] for label in labels]

# Plotting
plt.figure(figsize=(10, 5))
plt.boxplot(times, labels=labels)
plt.title("Inference Time Comparison")
plt.ylabel("Time (seconds)")
plt.show()
