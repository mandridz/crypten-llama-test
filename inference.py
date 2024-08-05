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

# Ensure generated_text is a string
if isinstance(generated_text, list):
    # Join the list of dictionaries into a single string
    generated_text = ' '.join([item['content'] for item in generated_text])

# Count the number of generated tokens using the tokenizer
num_generated_tokens = len(pipeline.tokenizer(generated_text)["input_ids"])

# Get memory usage
memory_info = psutil.virtual_memory()
memory_used = memory_info.used / (1024 ** 2)  # Convert to megabytes

# Define the output file name
output_file = "inference.py"

# Print and write results to file
with open(output_file, "w") as file:
    file.write("Inference Time\tNumber of Generated Tokens\tMemory Used\n")
    file.write(f"{inference_time:.2f}\t{num_generated_tokens}\t{memory_used:.2f}\n")

print(f"Results saved to {output_file}")
