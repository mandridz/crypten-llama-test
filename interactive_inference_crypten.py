import time
import torch
import crypten
import crypten.nn as cnn
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

crypten.init()

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name).to("cuda")

# Define a wrapper for the CrypTen model
class CrypTenLlamaModel(cnn.Module):
    def __init__(self, model):
        super(CrypTenLlamaModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        input_ids_plain = input_ids.get_plain_text().long()
        embeddings = self.model.get_input_embeddings()(input_ids_plain)
        outputs = self.model(inputs_embeds=embeddings)
        return outputs.logits

crypten_model = CrypTenLlamaModel(model)

# Function to measure inference time and generate output
def inference_crypten(model, input_ids_enc):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs_enc = model(input_ids_enc)
        end_time = time.time()
    return end_time - start_time, outputs_enc

# Function to calculate metrics
def calculate_metrics(logits, labels, reference, hypothesis):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
    perplexity = torch.exp(loss.mean()).item()

    reference = [reference.split()]
    hypothesis = hypothesis.split()
    bleu_score = sentence_bleu(reference, hypothesis)

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypothesis, reference[0])

    return perplexity, bleu_score, rouge_scores

prompt = "Please write a short story about an adventurous journey."
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
input_ids_enc = crypten.cryptensor(input_ids)

inference_time_crypten, outputs_enc = inference_crypten(crypten_model, input_ids_enc)
outputs_enc_plain = outputs_enc.get_plain_text()
generated_text = tokenizer.decode(outputs_enc_plain[0], skip_special_tokens=True)

reference_text = prompt  # For simplicity, using the prompt as reference
perplexity, bleu_score, rouge_scores = calculate_metrics(outputs_enc, input_ids, reference_text, generated_text)

with open('results_gpu_crypten.txt', 'w') as f:
    f.write(f"{inference_time_crypten}\n")
    f.write(f"{perplexity}\n")
    f.write(f"{bleu_score}\n")
    f.write(f"{rouge_scores}\n")
    f.write(f"{generated_text}\n")
