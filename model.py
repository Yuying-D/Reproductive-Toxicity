from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import dgl
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dgl.nn.pytorch import GATConv
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset, TensorDataset



class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x



class TransformerModel(nn.Module):
    def __init__(self, model_dir):
        super(TransformerModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_dir)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出



class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, allow_zero_in_degree=True)

    def forward(self, g, inputs):
        h = self.layer1(g, inputs)
        h = h.flatten(1)
        h = self.layer2(g, h)
        hg = dgl.mean_nodes(g, 'h')  
        return hg



class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z



class MolecularDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, model_dir, max_length=128):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.model_dir = model_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        encodings = self.tokenizer(smiles, padding='max_length', truncation=True, return_tensors="pt",
                                   max_length=self.max_length)
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        g = smiles_to_graph(smiles)
        node_features = torch.randn(g.number_of_nodes(), 10)
        g.ndata['h'] = node_features

        return input_ids, attention_mask, g, label



def collate_fn(batch):
    input_ids, attention_masks, graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, batched_graph, labels



def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid smiles string: {smiles}')
    g = dgl.graph(([], []))
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j and mol.GetBondBetweenAtoms(i, j) is not None:
                g.add_edges(i, j)
    g = dgl.add_self_loop(g)  # 添加自环
    return g



def preprocess_data(csv_file, model_dir):
    df = pd.read_csv(csv_file)
    smiles_list = df['SMILES'].tolist()
    labels = df['Label'].values
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return smiles_list, labels, tokenizer



def main():
    model_dir = '/Users/chz/Desktop/GPT/heart/workplace/ML/bert-base-uncased'
   
    smiles_list, labels, tokenizer = preprocess_data('/Users/chz/Desktop/GPT/hqz/ml/458.csv', model_dir)

   
    smiles_train, smiles_temp, labels_train, labels_temp = train_test_split(smiles_list, labels, test_size=0.4,
                                                                            random_state=8)
    smiles_val, smiles_test, labels_val, labels_test = train_test_split(smiles_temp, labels_temp, test_size=0.5,
                                                                        random_state=21)

    # 构建DataLoader
    train_dataset = MolecularDataset(smiles_train, labels_train, tokenizer, model_dir)
    val_dataset = MolecularDataset(smiles_val, labels_val, tokenizer, model_dir)
    test_dataset = MolecularDataset(smiles_test, labels_test, tokenizer, model_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    cnn_feature_dim = 30  
    transformer_feature_dim = 30  
    gat_feature_dim = 30  
    combined_feature_dim = cnn_feature_dim + transformer_feature_dim + gat_feature_dim

    
    cnn_model = CNN(input_dim=128, output_dim=cnn_feature_dim)
    transformer_model = TransformerModel(model_dir=model_dir)
    gat_model = GAT(in_dim=10, hidden_dim=20, out_dim=gat_feature_dim, num_heads=2)

    
    vae_model = VAE(input_dim=1738, latent_dim=1000)

    cnn_model.train()
    transformer_model.train()
    gat_model.train()

    all_train_features = []
    all_train_labels = []

    for input_ids, attention_mask, g, labels in train_loader:
        cnn_features = cnn_model(input_ids.float())
        transformer_features = transformer_model(input_ids, attention_mask)
        gat_features = gat_model(g, g.ndata['h']).detach()

        combined_features = torch.cat([cnn_features, transformer_features, gat_features], dim=1)
        combined_features_2d = combined_features.view(combined_features.size(0), -1).float()

        reconstructed_features, _ = vae_model(combined_features_2d)
        all_train_features.append(reconstructed_features.detach().numpy())
        all_train_labels.append(labels.numpy())

    all_train_features = np.concatenate(all_train_features, axis=0)
    all_train_labels = np.concatenate(all_train_labels, axis=0)

    
    xgboost_model = xgb.XGBClassifier()
    xgboost_model.fit(all_train_features, all_train_labels)

   
    def extract_features(loader, cnn_model, transformer_model, gat_model, vae_model):
        all_features = []
        all_labels = []
        for input_ids, attention_mask, g, labels in loader:
            cnn_features = cnn_model(input_ids.float())
            transformer_features = transformer_model(input_ids, attention_mask)
            gat_features = gat_model(g, g.ndata['h']).detach()

            combined_features = torch.cat([cnn_features, transformer_features, gat_features], dim=1)
            combined_features_2d = combined_features.view(combined_features.size(0), -1).float()

            reconstructed_features, _ = vae_model(combined_features_2d)
            all_features.append(reconstructed_features.detach().numpy())
            all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_features, all_labels

    val_features, val_labels = extract_features(val_loader, cnn_model, transformer_model, gat_model, vae_model)
    val_predictions = xgboost_model.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Validation Accuracy: {val_accuracy}')

    test_features, test_labels = extract_features(test_loader, cnn_model, transformer_model, gat_model, vae_model)
    test_predictions = xgboost_model.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {test_accuracy}')


    tp = np.sum((test_predictions == 1) & (test_labels == 1))
    tn = np.sum((test_predictions == 0) & (test_labels == 0))
    fp = np.sum((test_predictions == 1) & (test_labels == 0))
    fn = np.sum((test_predictions == 0) & (test_labels == 1))

    se = recall_score(test_labels, test_predictions)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(test_labels, test_predictions)
    q = (tp + tn) / (tp + tn + fp + fn)
    P = precision_score(test_labels, test_predictions)
    F1 = f1_score(test_labels, test_predictions)
    BA = (se + sp) / 2
    AUC = roc_auc_score(test_labels, test_predictions)

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    print(f'Sensitivity (SE): {se}')
    print(f'Specificity (SP): {sp}')
    print(f'MCC: {mcc}')
    print(f'Q: {q}')
    print(f'Precision (P): {P}')
    print(f'F1 Score: {F1}')
    print(f'Balanced Accuracy (BA): {BA}')
    print(f'AUC: {AUC}')

    
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

  
    explainer = shap.Explainer(xgboost_model)
    shap_values = explainer.shap_values(test_features)


    shap.summary_plot(shap_values, test_features, feature_names=[f'Feature_{i}' for i in range(test_features.shape[1])])


if __name__ == "__main__":
    main()
