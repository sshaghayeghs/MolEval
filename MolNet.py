import deepchem as dc
import pandas as pd
from rdkit import Chem

def load_dataset(dataset_name, split=False):
    loader_function = getattr(dc.molnet, f'load_{dataset_name}', None)
    if loader_function is None:
        raise ValueError(f"Dataset {dataset_name} not found in DeepChem.")

    featurizer = dc.feat.DummyFeaturizer()

    if split:
        if dataset_name in ['bbbp', 'bace_classification', 'hiv']:
            splitter = dc.splits.ScaffoldSplitter()
        else:
            splitter = dc.splits.RandomSplitter()
        data = loader_function(featurizer=featurizer, splitter=splitter)
        tasks, (train_dataset, valid_dataset, test_dataset), transformers = data
        train_df = convert_to_dataframe(train_dataset, tasks)
        valid_df = convert_to_dataframe(valid_dataset, tasks)
        test_df = convert_to_dataframe(test_dataset, tasks)
        return train_df, valid_df, test_df
    else:
        data = loader_function(featurizer=featurizer, splitter=None)
        tasks, (dataset,), transformers = data
        df = convert_to_dataframe(dataset, tasks)
        return df

def convert_to_dataframe(dataset, tasks):
    df = pd.DataFrame(dataset.X, columns=['SMILES'])
    for i, task in enumerate(tasks):
        df[task] = dataset.y[:, i]
    return df
