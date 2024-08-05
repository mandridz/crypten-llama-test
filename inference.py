import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful and honest programming assistant."},
    {"role": "user", "content": "Что такое Llama 3.1?"}
]

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

print(outputs[0]["generated_text"])