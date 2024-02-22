import pandas as pd
from MolNet import load_dataset
import Embedding 
embedding_class = Embedding.Embedding()
df=load_dataset('bace_classification')
df['SMILES'] = df['SMILES'].astype(str)
texts = df['SMILES'].iloc[:3].tolist() 
print(texts)
api_key = 'OpenAi Key'  # Replace with actual API key
embeddings = embedding_class.get_embeddings("simcse", texts, api_key)
embeddings_df = pd.DataFrame(embeddings)

print(embeddings_df)
