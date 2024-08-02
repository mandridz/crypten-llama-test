import subprocess


def run_script(script_name):
    result = subprocess.run(['python3', script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
    else:
        print(f"Output of {script_name}:\n{result.stdout}")


# Run interactive_inference_pytorch.py
print("Running inference_inference_pytorch.py")
run_script('inference_inference_pytorch.py')

# Run interactive_inference_crypten.py
print("Running inference_inference_crypten.py")
run_script('inference_inference_crypten.py')

# Run plot_results.py
print("Running plot_results.py")
run_script('plot_results.py')
