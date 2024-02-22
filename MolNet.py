import deepchem as dc  # works with pip install --pre deepchem
import pandas as pd
from rdkit import Chem

def load_dataset(dataset_name):
    loader_function = getattr(dc.molnet, f'load_{dataset_name}', None)
    if loader_function is None:
        raise ValueError(f"Dataset {dataset_name} not found in DeepChem.")

    data = loader_function(featurizer=dc.feat.DummyFeaturizer(), splitter=None)
    tasks, datasets, transformers = data

    dataset = datasets[0]
    labels = [i[0] for i in dataset.y]
    df = pd.DataFrame()
    df['SMILES'] = dataset.X
    df['LABELS'] = labels
    return df

df = load_dataset('bbbp')
type_of_smiles = type(df['SMILES'].iloc[0])
type_of_smiles