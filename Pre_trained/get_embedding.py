import datamol as dm
import platformdirs
import datamol as dm
import os
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from molfeat.trans.pretrained.hf_transformers import HFModel
import pandas as pd

def get_embedding(t,data):
  df = pd.DataFrame()
  try:
      chemgpt_local_dir = dm.fs.join(platformdirs.user_cache_dir("molfeat"), t)
      mapper = dm.fs.get_mapper(chemgpt_local_dir)
      mapper.fs.delete(chemgpt_local_dir, recursive=True)
  except FileNotFoundError:
      pass

  # make sure we clear the cache of the HFModel function
  HFModel._load_or_raise.cache_clear()
  os.environ["TOKENIZERS_PARALLELISM"] = "false" # annoying huggingface warning
  #data = dm.freesolv().iloc[:100]
  transformer = PretrainedHFTransformer(kind=t, notation='smiles', dtype=float)
  features = transformer(data)
  df=pd.DataFrame(features)
  return df
  #df.to_csv(f"{t}_embedding.csv",index=False)
