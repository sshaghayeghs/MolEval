import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel, AutoTokenizer, LlamaModel, LlamaTokenizer, RobertaModel, RobertaTokenizer,
    BertModel, BertTokenizer, GPT2TokenizerFast, GPT2LMHeadModel, AutoModelForCausalLM
)
from tqdm import tqdm
import openai
from sentence_transformers import SentenceTransformer
import deepchem as dc
from rdkit import Chem
from huggingface_hub import HfFolder
class EmbeddingExtractor:
    def __init__(self, hf_token=None, api_key=None):
        self.hf_token = hf_token
        self.api_key = api_key
        self.models = {
            "llama2": ("meta-llama/Llama-2-7b-hf", LlamaModel, LlamaTokenizer),
            "molformer": ("ibm/MoLFormer-XL-both-10pct", AutoModel, AutoTokenizer, True),  # Added a flag for trust_remote_code
            "chemberta": ("DeepChem/ChemBERTa-10M-MLM", RobertaModel, RobertaTokenizer),
            "bert": ("bert-base-uncased", BertModel, BertTokenizer),
            "roberta_zinc": ("entropy/roberta_zinc_480m", RobertaModel, RobertaTokenizer),
            "gpt2": ("entropy/gpt2_zinc_87m", GPT2LMHeadModel, GPT2TokenizerFast),
            "roberta": ("FacebookAI/roberta-base", RobertaModel, RobertaTokenizer),
            "simcse": ("princeton-nlp/sup-simcse-bert-base-uncased", AutoModel, AutoTokenizer),
            "anglebert": ("SeanLee97/angle-bert-base-uncased-nli-en-v1", AutoModel, AutoTokenizer),
            "sbert": SentenceTransformer("all-MiniLM-L6-v2"),
            "mol2vec": dc.feat.Mol2VecFingerprint(),
            "morgan": dc.feat.CircularFingerprint(size=1024, radius=2)
        }
        if hf_token:
            self.authenticate_huggingface(hf_token)

    def authenticate_huggingface(self, token):
        HfFolder.save_token(token)

    def load_model_tokenizer(self, model_key):
        if model_key in self.models:
            item = self.models[model_key]
            if isinstance(item, tuple):
                if len(item) == 4:
                    path, model_cls, tokenizer_cls, trust_remote = item
                    model = model_cls.from_pretrained(path, trust_remote_code=trust_remote)
                    tokenizer = tokenizer_cls.from_pretrained(path, trust_remote_code=trust_remote)
                else:
                    path, model_cls, tokenizer_cls = item
                    model = model_cls.from_pretrained(path)
                    tokenizer = tokenizer_cls.from_pretrained(path)
                if model_key == "llama2":
                    tokenizer.pad_token = tokenizer.eos_token
                return model, tokenizer
            else:
                return item, None  # For feature extractors like 'morgan'
        else:
            raise ValueError(f"Unsupported model key: {model_key}")

    def compute_embeddings(self, texts, model, tokenizer):
        MAX_LENGTH = 512
        embeddings = []
        if tokenizer:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                full_embeddings = outputs.hidden_states[-1]
                mask = inputs['attention_mask']
                embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))
            return embeddings.detach().cpu().numpy()
        else:  # Handle non-language models
            return model.featurize(texts)  # Assuming texts are suitable for the model (e.g., SMILES strings for 'morgan')

    def sbert_embedding(self, texts):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
        return embeddings

    def extract_features(self, texts, model_key):
        if model_key in self.models:
            if model_key == 'sbert':
                embeddings = self.sbert_embedding(texts)
            elif 'openai' in model_key:
                embeddings = self.get_embeddings_from_openai(texts)
            else:
                model, tokenizer = self.load_model_tokenizer(model_key)
                if tokenizer is None:  # Indicates a feature extractor like 'morgan'
                    embeddings = model.featurize(texts)  # Assuming texts are SMILES strings for chemical features
                else:
                    embeddings = self.compute_embeddings(texts, model, tokenizer)
            return pd.DataFrame(embeddings)
        else:
            raise ValueError(f"Unsupported model key: {model_key}")
