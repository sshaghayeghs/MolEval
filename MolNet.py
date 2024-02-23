import deepchem as dc
import pandas as pd
from rdkit import Chem

def load_dataset(dataset_name):
    loader_function = getattr(dc.molnet, f'load_{dataset_name}', None)
    if loader_function is None:
        raise ValueError(f"Dataset {dataset_name} not found in DeepChem.")

    data = loader_function(featurizer=dc.feat.DummyFeaturizer(), splitter=None)
    tasks, datasets, transformers = data

    dataset = datasets[0]
    
    df = pd.DataFrame(dataset.X, columns=['SMILES'])
    for i, task in enumerate(tasks):
        df[task] = dataset.y[:, i]
    
    return df
