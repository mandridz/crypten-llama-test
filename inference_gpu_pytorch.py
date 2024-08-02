import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from flask import Flask, jsonify

app = Flask(__name__)

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")


def inference_pytorch(model, input_ids):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(input_ids, max_length=500, num_beams=5, early_stopping=True, temperature=0.5)
        end_time = time.time()
    return end_time - start_time, outputs


@app.route('/inference', methods=['POST'])
def run_inference():
    with open("prompt.txt", "r", encoding="utf-8") as file:
        input_text = file.read()

    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    inference_time_pytorch, outputs = inference_pytorch(model, input_ids)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        'inference_time': inference_time_pytorch,
        'generated_text': generated_text
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
