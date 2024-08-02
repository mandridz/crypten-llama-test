import subprocess

scripts = ['inference_gpu_pytorch.py', 'inference_gpu_crypten.py', 'plot_results.py']

for script in scripts:
    print(f"Running {script}")
    result = subprocess.run(['python3', script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}:\n")
        print(result.stderr)
    else:
        print(f"Successfully ran {script}")
