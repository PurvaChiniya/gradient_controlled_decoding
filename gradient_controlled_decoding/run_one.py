import copy
import pickle
import torch
import numpy as np
import pandas as pd
import os
import json
import pickle
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import itertools
from find_critical_parameters import find_critical_para, load_model

import json
import pandas as pd
import json


def process_query(prompt, label):
    system_prompt = generate_system_prompt()
    if label == 0:
        first_token = random.choice(first_tokens_positive)
    else:
        first_token = random.choice(first_tokens_negative)

    full_prompt = f"{system_prompt} USER: {prompt} ASSISTANT:{first_token}"

    input_ids = tokenizer(full_prompt, return_tensors="pt")['input_ids'].to(model.device)
    logits_processors = LogitsProcessorList([
        TemperatureLogitsWarper(temperature=0.7),
        TopPLogitsWarper(top_p=0.9)
    ])

    while len(input_ids[0]) < max_length:
        logits = model(input_ids).logits[:, -1, :]
        logits = logits_processors(input_ids, logits)
        probabilities = torch.softmax(logits, dim=-1)
        top_token = torch.multinomial(probabilities, num_samples=1)
        input_ids = torch.cat([input_ids, top_token], dim=-1)
        if top_token.item() in termination_tokens:
            break

    return tokenizer.decode(input_ids[0])

def save_gradients(gradients, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(gradients, f)

def load_gradients(file_name):
    with open(file_name, 'rb') as f:
        gradients = pickle.load(f)
    return gradients


def cos_sim(model_id, df ,gradient_norms_compare,  minus_row, minus_col,response):
    columns = df[['prompt']]
    model, tokenizer = load_model(model_id,device='cuda:6')

    def build_model_inputs(sample):
        prefix_text = (
            f'<s>[INST] <<SYS>> {{system_prompt}} <</SYS>> {{content}} [/INST]'
        ).format(
            system_prompt='You are a helpful assistant. Help me with the following query: ',
            content=sample['source'],
        )
        target_text = f" {sample['target']} {tokenizer.eos_token}"
        prefix_ids = tokenizer(prefix_text).input_ids
        target_ids = tokenizer(target_text, add_special_tokens=False).input_ids
        input_ids = torch.tensor(np.array([prefix_ids + target_ids]))
        labels = input_ids.clone()
        labels[:, :len(prefix_ids)] = -100
        return input_ids, labels


    #  Calculate cosine similarity based score 
    cos_all = []
    label_all = []
    from tqdm import tqdm 
    for index, row in tqdm(columns.iterrows(), total=len(columns), desc="Processing rows"):
        # Ground truth label
        
        label_all.append(1)
        
        
        # Gradient for prompt paired with complaince response Sure
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        basic_sample = {}
        basic_sample["source"] = row["prompt"]
        basic_sample["target"] = response
        input_ids, target_ids = build_model_inputs(basic_sample)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        neg_log_likelihood.backward()
        
        # Cosine similarities for safety-critical parameters
        cos = []
        import torch.nn.functional as F
        for name, param in model.named_parameters():
            if  param.grad is not None and ("mlp" in name or "self" in name):
                grad_norm = param.grad.to(gradient_norms_compare[name].device)
                row_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=1))
                col_cos = torch.nan_to_num(F.cosine_similarity(grad_norm, (gradient_norms_compare[name]), dim=0))
                ref_row = minus_row[name]
                ref_col = minus_col[name]
                cos.extend(row_cos[ref_row>1].cpu().tolist())
                cos.extend(col_cos[ref_col>1].cpu().tolist())

        cos_all.append(cos)
    # For GradSafe-Zero, average all cosine similarites as one score
    cos_all = [sum(feature)/len(feature) for feature in cos_all]
    return cos_all
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, TopPLogitsWarper, TemperatureLogitsWarper
import random

# Load the model and tokenizer
def load_model(model_id="meta-llama/Llama-2-7b-chat-hf", device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def generate_system_prompt():
    return (
        "<INST>You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF. "
        "You carefully provide accurate, factual, thoughtful, nuanced answers, and are brilliant at reasoning. "
        "If you think there might not be a correct answer, you say so. Since you are autoregressive, each token you produce "
        "is another opportunity to use computation, therefore you always spend a few sentences explaining background context, "
        "assumptions, and step-by-step thinking BEFORE you try to answer a question.</INST>"
    )

model, tokenizer = load_model()


if __name__ == "__main__":
    for model_id in ['meta-llama/Llama-2-7b-chat-hf']:
        
        gradient_norms_compare_sure, minus_row_cos_sure, minus_col_cos_sure =  find_critical_para(model_id,"Sure")
        #save_gradients(gradient_norms_compare_sure, 'gradient_norms_compare_sure.pkl')  # Save gradients
        #save_gradients(minus_row_cos_sure, 'minus_row_cos_sure.pkl')  # Save gradients
        #save_gradients(minus_col_cos_sure, 'minus_col_cos_sure.pkl')  # Save gradients
        

        gradient_norms_compare_sorry, minus_row_cos_sorry, minus_col_cos_sorry =  find_critical_para(model_id,"Sorry")
        #save_gradients(gradient_norms_compare_sorry, 'gradient_norms_compare_sorry.pkl')  # Save gradients
        #save_gradients(minus_row_cos_sorry, 'minus_row_cos_sorry.pkl')  # Save gradients
        #save_gradients(minus_col_cos_sorry, 'minus_col_cos_sorry.pkl')  # Save gradients
        
        
    for _ in range(10):
        prompt = input("Enter your prompt: ")
        df = pd.DataFrame({'prompt': [prompt]})

        cos_sure = cos_sim(model_id, df,gradient_norms_compare_sure, minus_row_cos_sure, minus_col_cos_sure,"Sure")
        cos_sorry = cos_sim(model_id, df,gradient_norms_compare_sorry, minus_row_cos_sorry, minus_col_cos_sorry,"Sorry")
        import pdb 
        pdb.set_trace()
        label = (cos_sure[0]>0.25) & (cos_sorry[0]>0.25)
        df["predicted_label"]= label
        # Define token lists based on label
        first_tokens_positive = ["Sure,", "Absolutely,", "Certainly,"]
        first_tokens_negative = ["Sorry,", "Unfortunately,", "I cannot,"]

        # Define maximum length and termination tokens
        max_length = 512
        termination_tokens = {tokenizer.encode(token)[0] for token in ['.', '?', '!']}

        # Process each row and save the response
        generated_response= df.apply(lambda row: process_query(row['prompt'], row['predicted_label']), axis=1)
        
        for response in generated_response:
            assistant_response = response.split("ASSISTANT:")[1].strip()  # Extract and clean up response
            print(assistant_response)
        

        
