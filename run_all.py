import subprocess

scripts = ["inference_gpu_pytorch.py", "inference_gpu_crypten.py"]

for script in scripts:
    try:
        result = subprocess.run(["python3", script], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running {script}:\n\n{result.stderr}")
        else:
            print(f"Successfully ran {script}")
    except Exception as e:
        print(f"Exception occurred while running {script}: {e}")

# Plot results
try:
    result = subprocess.run(["python3", "plot_results.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running plot_results.py:\n{result.stderr}")
    else:
        print("Successfully ran plot_results.py")
except Exception as e:
    print(f"Exception occurred while running plot_results.py: {e}")
