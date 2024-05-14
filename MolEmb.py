from transformers import (
    AutoModel, AutoTokenizer, LlamaModel, LlamaTokenizer, RobertaModel, RobertaTokenizer,
    BertModel, BertTokenizer, AutoModelForCausalLM
)
import openai
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem

class EmbeddingExtractor:
    def __init__(self, model_name, df, openai_api_key=None, huggingface_token=None):
        self.model_name = model_name
        self.df = df
        self.smiles = df['SMILES'].to_list()
        self.target = df.drop(columns=['SMILES']).to_numpy()
        self.openai_api_key = openai_api_key
        self.huggingface_token = huggingface_token
        self.tokenizer = None
        self.model = None
        self.all_embeddings = []
        self.errored_indices = []
        self.select_model()
        self.process_smiles()

    def select_model(self):
        if self.model_name == 'SBERT':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        elif self.model_name == 'LLaMA2':
            self.login_to_huggingface()
            self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.model_name == 'Molformer':
            self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct")
            self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct")
        elif self.model_name == 'ChemBERTa':
            self.tokenizer = RobertaTokenizer.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
            self.model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-10M-MLM")
        elif self.model_name == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
            self.model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        elif self.model_name == 'RoBERTa_ZINC':
            self.tokenizer = RobertaTokenizer.from_pretrained("entropy/roberta_zinc_480m")
            self.model = RobertaModel.from_pretrained("entropy/roberta_zinc_480m")
        elif self.model_name == 'RoBERTa':
            self.tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
            self.model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        elif self.model_name == 'SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
            self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        elif self.model_name == 'AngleBERT':
            self.tokenizer = AutoTokenizer.from_pretrained("SeanLee97/angle-bert-base-uncased-nli-en-v1")
            self.model = AutoModel.from_pretrained("SeanLee97/angle-bert-base-uncased-nli-en-v1")
        elif self.model_name == 'GPT':
            openai.api_key = self.openai_api_key
            self.client = openai.OpenAI(api_key=openai.api_key)
        elif self.model_name == 'Mol2Vec':
            self.featurizer = dc.feat.Mol2VecFingerprint()
        elif self.model_name == 'Morgan':
            pass
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def login_to_huggingface(self):
        if self.huggingface_token:
            !huggingface-cli login --token {self.huggingface_token}
        else:
            raise ValueError("Huggingface token is required for LLaMA2 model")

    def process_smiles(self):
        if self.model_name in ['SBERT', 'LLaMA2', 'Molformer', 'ChemBERTa', 'BERT', 'RoBERTa_ZINC', 'RoBERTa', 'SimCSE', 'AngleBERT']:
            self._process_transformers()
        elif self.model_name == 'GPT':
            self._process_gpt()
        elif self.model_name == 'Mol2Vec':
            self._process_mol2vec()
        elif self.model_name == 'Morgan':
            self._process_morgan()

    def _process_transformers(self):
        for i, smile in enumerate(self.smiles):
            try:
                inputs = self.tokenizer(smile, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
                outputs = self.model(**inputs, output_hidden_states=True)
                full_embeddings = outputs.hidden_states[-1]
                mask = inputs['attention_mask']
                embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1))
                self.all_embeddings.append(embeddings.detach().cpu().numpy())
            except Exception as e:
                self.errored_indices.append(i)
                print(f"Error processing SMILES at index {i}: {smile}")
                print(f"Error message: {str(e)}")

    def _process_gpt(self):
        for i, s in enumerate(self.smiles):
            try:
                response = self.client.embeddings.create(input=s, model="text-embedding-ada-002")
                self.all_embeddings.append(response['data'][0]['embedding'])
            except Exception as e:
                self.errored_indices.append(i)
                print(f"Error processing SMILES at index {i}: {s}")
                print(f"Error message: {str(e)}")

    def _process_mol2vec(self):
        for i, s in enumerate(self.smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                self.errored_indices.append(i)
                print(f"Failed to create molecule from SMILES at index {i}: {s}")
            else:
                try:
                    features = self.featurizer.featurize([mol])
                    self.all_embeddings.append(features)
                except Exception as e:
                    self.errored_indices.append(i)
                    print(f"Error processing SMILES at index {i}: {s}")
                    print(f"Error message: {str(e)}")

    def _process_morgan(self):
        for i, s in enumerate(self.smiles):
            try:
                fp = self.get_morgan_fingerprint(s)
                self.all_embeddings.append(fp)
            except Exception as e:
                self.errored_indices.append(i)
                print(f"Error processing SMILES at index {i}: {s}")
                print(f"Error message: {str(e)}")

    def get_morgan_fingerprint(self, smiles, radius=2, nBits=1024):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            return list(fp)
        else:
            raise ValueError(f"Failed to create molecule from SMILES: {smiles}")

    def get_embeddings(self):
        squeezed_array = np.squeeze(self.all_embeddings)
        emb = pd.DataFrame(squeezed_array)
        filtered_df = self.df.drop(index=self.errored_indices).reset_index(drop=True)
        return emb, filtered_df
