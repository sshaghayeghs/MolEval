import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from angle_emb import Prompts

class Embedding:
    def __init__(self):
        self.models = {
            "simcse": self.simcse_embedding,
            "chatgpt": self.chatgpt_embedding,
            "sbert": self.sbert_embedding,
            "anglebert": self.anglebert_embedding,  # Add AngleBERT here
            "anglellama": self.angle_llama_embedding  # Add AnglE here

        }
        self.model_name = "text-embedding-3-small"  # Replace with the actual model name

    def chatgpt_embedding(self, texts, api_key):
        openai.api_key = api_key
        embeddings = []
        for text in tqdm(texts):
            try:
                response = openai.Embedding.create(
                    model=self.model_name,
                    input=text)
                embedding = [item["embedding"] for item in response["data"]]
                embeddings.append(np.array(embedding[0]))
            except Exception as e:
                print(f"Error in processing text: {text}. Error: {e}")
                embeddings.append(np.zeros(768))  # Assuming embedding size is 768, adjust if different
        return embeddings
    
    def simcse_embedding(self, texts):
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

        embeddings_cpu = embeddings.cpu()
        numpy_embeddings = embeddings_cpu.numpy()
        return numpy_embeddings
    def anglebert_embedding(self, texts):
        model_id = 'SeanLee97/angle-bert-base-uncased-nli-en-v1'
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).cuda()

        embeddings = []
        for text in tqdm(texts):
            tok = tokenizer([text], return_tensors='pt')
            for k, v in tok.items():
                tok[k] = v.cuda()

            with torch.no_grad():
                hidden_state = model(**tok).last_hidden_state
                vec = (hidden_state[:, 0] + torch.mean(hidden_state, dim=1)) / 2.0
                embeddings.append(vec.cpu().numpy())

        return np.vstack(embeddings)
    def angle_llama_embedding(self, texts):
        peft_model_id = 'SeanLee97/angle-llama-7b-nli-v2'
        config = PeftConfig.from_pretrained(peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path).bfloat16().cuda()
        model = PeftModel.from_pretrained(base_model, peft_model_id).cuda()

        embeddings = []
        for text in tqdm(texts):
            decorated_text = Prompts.A.format(text=text)
            tok = tokenizer([decorated_text], return_tensors='pt')
            for k, v in tok.items():
                tok[k] = v.cuda()

            with torch.no_grad():
                vec = model(output_hidden_states=True, **tok).hidden_states[-1][:, -1].float().detach().cpu().numpy()
                embeddings.append(vec)

        return np.vstack(embeddings)

    def sbert_embedding(self, texts):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
        return embeddings

    def get_embeddings(self, model_name, texts, api_key=None):
        if model_name in self.models:
            if model_name == "chatgpt" and api_key:
                return self.models[model_name](texts, api_key)
            return self.models[model_name](texts)
        else:
            raise ValueError(f"Model {model_name} not supported.")

