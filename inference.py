import transformers
import torch
import time
import psutil

# Model identifier
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Create a text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Input messages
messages = [
    {"role": "system", "content": "You are a helpful and honest programming assistant."},
    {"role": "user", "content": "What is Llama 3.1?"}
]

# Measure the start time of inference
start_time = time.time()

# Generate text
outputs = pipeline(
    messages,
    max_new_tokens=256,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_beams=1,
    early_stopping=True,
    pad_token_id=pipeline.tokenizer.eos_token_id
)

# Measure the end time of inference
end_time = time.time()
inference_time = end_time - start_time

# Get the generated text
generated_text = outputs[0]["generated_text"]

# Count the number of generated tokens
num_generated_tokens = len(pipeline.tokenizer.encode(generated_text))

# Get memory usage
memory_info = psutil.virtual_memory()
memory_used = memory_info.used / (1024 ** 2)  # Convert to megabytes

# Print results
print("Generated Text:")
print(generated_text)
print("\nInference Metrics:")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Number of Generated Tokens: {num_generated_tokens}")
print(f"Memory Used: {memory_used:.2f} MB")
