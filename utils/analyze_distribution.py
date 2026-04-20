import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, DataStructs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_maccs_fingerprints(df):
    """
    Compute MACCS keys for SMILES in a DataFrame.
    Returns list of MACCS fingerprints, valid SMILES, and corresponding labels.
    """
    fingerprints = []
    valid_smiles = []
    valid_labels = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing MACCS keys"):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES at index {idx}: {smiles}")
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fingerprints.append(fp)
        valid_smiles.append(smiles)
        valid_labels.append(row['y'])
    return fingerprints, valid_smiles, valid_labels

def compute_pairwise_tanimoto(fps):
    """
    Compute average pairwise Tanimoto similarity for a list of fingerprints.
    Returns mean and standard deviation.
    """
    similarities = []
    for i in tqdm(range(len(fps)), desc="Computing pairwise Tanimoto"):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0, np.std(similarities) if similarities else 0

def compute_cross_set_tanimoto(fps1, fps2):
    """
    Compute average Tanimoto similarity between two sets of fingerprints.
    """
    similarities = []
    for fp1 in tqdm(fps1, desc="Computing cross-set Tanimoto"):
        for fp2 in fps2:
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0, np.std(similarities) if similarities else 0

def plot_tsne(train_fps, val_fps, train_labels, val_labels, output_path):
    """
    Apply t-SNE to MACCS fingerprints and plot train/val distributions.
    Saves 2D scatter plot to output_path.
    """
    # Convert fingerprints to numpy array
    all_fps = train_fps + val_fps
    fp_array = np.array([np.array(list(fp.ToBitString()), dtype=int) for fp in all_fps])
    labels = np.array(train_labels + val_labels)
    set_indicator = np.array(['Train'] * len(train_fps) + ['Ext_Val'] * len(val_fps))                     # Edit labels based on your test or analysis

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_fps)-1))
    tsne_results = tsne.fit_transform(fp_array)

    # Plot
    plt.figure(figsize=(10, 6))
    for set_name in ['Train', 'Ext_Val']:                     # Edit labels based on your test or analysis
        mask = set_indicator == set_name
        plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                    label=set_name, alpha=0.6, s=50)
    plt.title("t-SNE of MACCS Fingerprints (Train vs Ext_Val)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"t-SNE plot saved to: {output_path}")

def analyze_dataset_diversity(train_df_path, val_df_path, tsne_plot_path):
    """
    Analyze chemical diversity of train and val DataFrames using MACCS keys and t-SNE.
    """
    # Load DataFrames an check class distribution
    print("Loading datasets...")
    train_df = pd.read_csv(train_df_path)  
    val_df = pd.read_csv(val_df_path)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    train_counts = train_df['y'].value_counts()
    val_counts = val_df['y'].value_counts()
    print(f"Train class distribution: {train_counts.to_dict()}")
    print(f"Val class distribution: {val_counts.to_dict()}")

    # Compute MACCS fingerprints
    train_fps, train_smiles, train_labels = compute_maccs_fingerprints(train_df)
    val_fps, val_smiles, val_labels = compute_maccs_fingerprints(val_df)
    print(f"Valid MACCS keys: {len(train_fps)} train, {len(val_fps)} val")

    # Compute pairwise Tanimoto within sets
    train_mean, train_std = compute_pairwise_tanimoto(train_fps)
    val_mean, val_std = compute_pairwise_tanimoto(val_fps)
    print(f"Train set - Avg Tanimoto: {train_mean:.3f} ± {train_std:.3f}")
    print(f"Val set - Avg Tanimoto: {val_mean:.3f} ± {val_std:.3f}")

    # Compute cross-set Tanimoto
    cross_mean, cross_std = compute_cross_set_tanimoto(train_fps, val_fps)
    print(f"Train-Val - Avg Tanimoto: {cross_mean:.3f} ± {cross_std:.3f}")

    # Check for duplicates and generate the t-SNE plot
    duplicates = set(train_smiles) & set(val_smiles)
    print(f"Duplicate SMILES between train and val: {len(duplicates)}")

    plot_tsne(train_fps, val_fps, train_labels, val_labels, tsne_plot_path)