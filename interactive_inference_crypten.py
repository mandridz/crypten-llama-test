import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaForCausalLM
from flask import Flask, jsonify

app = Flask(__name__)

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")


def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        logits = model(input_ids_enc.get_plain_text().to("cuda")).logits
        end_time = time.time()
    return end_time - start_time, logits


@app.route('/inference', methods=['POST'])
def run_inference():
    with open("prompt.txt", "r", encoding="utf-8") as file:
        input_text = file.read()

    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    input_ids_enc = crypten.cryptensor(input_ids)

    inference_time_crypten, logits_enc = inference_crypten(model, input_ids_enc)
    logits_plain = logits_enc.argmax(dim=-1).tolist()
    generated_text = tokenizer.decode(logits_plain, skip_special_tokens=True)

    result = {
        'inference_time': inference_time_crypten,
        'generated_text': generated_text
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
