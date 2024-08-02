import subprocess


def run_script(script_name):
    result = subprocess.run(['python3', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Output of {script_name}:\n{result.stdout}")


# Run inference_gpu_pytorch.py
print("Running inference_gpu_pytorch.py")
run_script('inference_gpu_pytorch.py')

# Run inference_gpu_crypten.py
print("Running inference_gpu_crypten.py")
run_script('inference_gpu_crypten.py')

# Run plot_results.py
print("Running plot_results.py")
run_script('plot_results.py')
