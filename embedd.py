'This code is used to embedd the SMILES strings of the polymers into a 300-dimensional'
'space using the mol2vec model. The embeddings are then saved as a parquet file.'

from rdkit import Chem
import numpy as np 
import pandas as pd
import sys

from gensim.models import word2vec
from tqdm import tqdm
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec 


# In[5]:
def get_data(data_path: str):
    return pd.read_csv(data_path)['SMILES']

def get_model(model_path: str):
    return word2vec.Word2Vec.load(model_path)

# In[3]:

def embedd_data(data: pd.Series, model: word2vec.Word2Vec):
    sentences = list()
    smiles_as_words = [f'{i}' for i in data]

    for word in tqdm(smiles_as_words): 
        mol = Chem.MolFromSmiles(word)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        sentences.append(sentence)
    
    polymer_embeddings = [DfVec(x) for x in sentences2vec(sentences, model, unseen='UNK')]

    return polymer_embeddings

# In[6]:
def transform_and_save(data: pd.Series, model: word2vec.Word2Vec, save_path: str):
    embedded_polymers = embedd_data(data, model)
    df_X = pd.DataFrame([x.vec for x in embedded_polymers])
    df_X.to_parquet(save_path, compression='snappy')


# In[6]:
if __name__ == '__main__':
    if len(sys.argv) == 4:
        data_path = sys.argv[1]
        model_path = sys.argv[2]
        save_path = sys.argv[3]
        data = get_data(data_path)
        model = get_model(model_path)
        transform_and_save(data, model, save_path)
    else:
        print('Usage: embedd.py data_path model_path save_path')

