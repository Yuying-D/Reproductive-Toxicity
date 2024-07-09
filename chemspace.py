import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


df1 = pd.read_csv('0.csv')
df2 = pd.read_csv('1.csv')


def calculate_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None

df1['fingerprints'] = df1['SMILES'].apply(calculate_fingerprints)
df2['fingerprints'] = df2['SMILES'].apply(calculate_fingerprints)
df1['source'] = '0'
df2['source'] = '1'
df = pd.concat([df1, df2]).drop_duplicates('SMILES').reset_index(drop=True)
df = df[df['fingerprints'].notnull()]
fingerprints = np.array([list(fp) for fp in df['fingerprints']])

pca = PCA(n_components=2)
pca_results = pca.fit_transform(fingerprints)
df['x_pca'], df['y_pca'] = pca_results[:, 0], pca_results[:, 1]


def assign_color(row):
    if row['source'] == '0':
        return '#808080'  
    else:
        return '#90EE90'  

df['color'] = df.apply(assign_color, axis=1)


plt.figure(figsize=(10, 8))

fullset = df[df['source'] == '0']
plt.scatter(fullset['x_pca'], fullset['y_pca'], s=6, alpha=0.6, c=fullset['color'])


subset = df[df['source'] == '1']
plt.scatter(subset['x_pca'], subset['y_pca'], s=6, alpha=0.6, c=subset['color'])


plt.title('Chemical Space Distribution by PCA')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()
