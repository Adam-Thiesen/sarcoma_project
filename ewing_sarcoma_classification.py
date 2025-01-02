import seaborn as sns
import numpy as np
import pandas as pd
import h5py
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc
import scipy as sp
import os


# 1) process_h5ad

def process_h5ad(file_path):
    with h5py.File(file_path, "r") as f:
        # Access metadata
        metadata = {}
        for key in f['obs'].keys():
            
            if isinstance(f[f'obs/{key}'], h5py.Dataset):
                metadata[key] = f[f'obs/{key}'][:]
        
       
        metadata_df = pd.DataFrame(metadata)

        # Check if 'Patient ID' exists with 'categories' and 'codes'
        if 'Patient ID' in f['obs']:
            try:
                categories = f['obs/Patient ID/categories'][:]
                codes = f['obs/Patient ID/codes'][:]
                
                categories = [
                    x.decode('utf-8') if isinstance(x, bytes) else x 
                    for x in categories
                ]
                metadata_df['Patient ID'] = [categories[code] for code in codes]
            except Exception as e:
                print(f"Could not process 'Patient ID': {e}")

        feature_data = f['X'][:]  # Extract feature matrix
        var_names = f['var/_index'][:].astype(str)  # Feature names (column names)

    
    feature_df = pd.DataFrame(feature_data, columns=var_names)

    if 'Tissue ID' in metadata_df.columns:
        # Remove b'...' from Tissue IDs
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID']
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
        )
        # Use Tissue ID as index
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index = metadata_df.index

    return metadata_df, feature_df



# 2) extract_histological_type
def extract_histological_type(file_path):
    """Extracts 'Histological Subtype' from a .h5ad file."""
    try:
        with h5py.File(file_path, "r") as f:
            categories = f['obs/Histological Subtype/categories'][:]
            codes = f['obs/Histological Subtype/codes'][:]
            categories = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in categories
            ]
            histological_type = [categories[code] for code in codes]
        return histological_type
    except Exception as e:
        print(f"Error extracting histological type from {file_path}: {e}")
        return []



# 3) process_combination for same samples, different features

def process_combination(file1, file2):
   
    metadata1, feature_df1 = process_h5ad(file1)
    metadata2, feature_df2 = process_h5ad(file2)

    common_index = metadata1.index.intersection(metadata2.index)
    if len(common_index) == 0:
        print("No matching samples between these two files. Returning empty.")
        return pd.DataFrame()

    
    metadata1 = metadata1.loc[common_index].copy()
    feature_df1 = feature_df1.loc[common_index].copy()
    metadata2 = metadata2.loc[common_index].copy()
    feature_df2 = feature_df2.loc[common_index].copy()

    combined_metadata = metadata1

    combined_features = pd.concat([feature_df1, feature_df2], axis=1)

    merged_df = pd.concat([combined_metadata, combined_features], axis=1)

    histological_type1 = extract_histological_type(file1)
    histological_type2 = extract_histological_type(file2)
    
    
    if len(histological_type1) != len(metadata1) or len(histological_type2) != len(metadata2):
        print("Warning: Histological subtype array length mismatch. Returning empty.")
        return pd.DataFrame()

    

    merged_df['Histological Subtype'] = histological_type1

    # Filter for specific 'Histological Type' values
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].apply(lambda x: 1 if x == 'Ewing Sarcoma' else 0)
    filtered_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    if filtered_df.empty:
        raise ValueError(f"No rows with 'Tissue ID' ending in '.oid0' in file: {file_path}")

    # Ensure 'Tissue ID' is a string
    filtered_df['Tissue ID'] = filtered_df['Tissue ID'].astype(str)

    #filtered_df = merged_df
    
    print(filtered_df.head())

    return filtered_df

# 5) get_global_min_max

def get_global_min_max(auc_matrices):
    global_min = np.inf
    global_max = -np.inf
    for auc_matrix in auc_matrices:
        # Exclude diagonal entries
        valid_values = auc_matrix[~np.eye(auc_matrix.shape[0], dtype=bool)]
        if valid_values.size > 0:
            global_min = min(global_min, np.nanmin(valid_values))
            global_max = max(global_max, np.nanmax(valid_values))
    return global_min, global_max


# 6) run_logistic_regression

def run_logistic_regression(filtered_df, num_iterations=10):
    """Performs logistic regression with GroupKFold to stratify by Patient ID."""
    try:
        # The columns in your metadata = first few columns
        data_array = filtered_df.iloc[:, 3:-1].to_numpy()  # Features
        print(data_array[:5])
        labels = filtered_df.iloc[:, -1].to_numpy() # Labels
        print(labels[:5])

        groups = filtered_df['Patient ID'].to_numpy()      # Grouping

        gkf = GroupKFold(n_splits=5)
        aucs = []

        for _ in range(num_iterations):
            for train_idx, test_idx in gkf.split(data_array, labels, groups=groups):
                train_data, test_data = data_array[train_idx], data_array[test_idx]
                train_labels, test_labels = labels[train_idx], labels[test_idx]

                # Feature selection (t-test)
                _, p_values = sp.stats.ttest_ind(
                    train_data[train_labels == 0],
                    train_data[train_labels == 1],
                    axis=0,
                    equal_var=False
                )
                selected_features = np.where(p_values < 0.05)[0]

                # Skip if no features are selected
                if selected_features.size == 0:
                    continue

                # Train logistic regression
                model = LR(
                    penalty='l1', #l2 penalty
                    C=100, 
                    class_weight='balanced', 
                    solver='liblinear', 
                    max_iter=2000
                ).fit(train_data[:, selected_features], train_labels)

                
                probs = model.predict_proba(test_data[:, selected_features])[:, 1]
                
                fpr, tpr, _ = roc_curve(test_labels, probs)
                fold_auc = auc(fpr, tpr)
                aucs.append(fold_auc)

        
        return np.mean(aucs) if aucs else None
    except Exception as e:
        print(f"Error during logistic regression: {e}")
        return None

# 7) test_combinations

def test_combinations(backbone, file_ids):

    base_path = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/"
    files = [f"{base_path}ad_wsi.{backbone}-{i}.h5ad" for i in file_ids]
    num_files = len(files)

    # Initialize an empty square matrix
    auc_matrix = np.full((num_files, num_files), -1, dtype=float)  # Diagonal = -1

    # Iterate over all unique combinations of file pairs
    for i, j in combinations(range(num_files), 2):
        print(f"Processing combination: {files[i]} + {files[j]}")  # Debug
        filtered_df = process_combination(files[i], files[j])
        if not filtered_df.empty:
            auc_val = run_logistic_regression(filtered_df, num_iterations=100)
            print("THE AUC VALUE IS:")
            print(auc_val)
            if auc_val is not None:
                auc_matrix[i, j] = auc_val
                auc_matrix[j, i] = auc_val

    return auc_matrix


# 8) plot_backbone_heatmap

def plot_backbone_heatmap(auc_matrix, file_ids, backbone, vmin, vmax, output_dir):
    labels = [f"{backbone}-{file_id}".upper() for file_id in file_ids]
    mask = np.eye(auc_matrix.shape[0], dtype=bool)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        auc_matrix,
        annot=True,
        fmt=".3f",
        cmap="Reds",
        mask=mask,
        xticklabels=labels,  # Apply uppercase labels
        yticklabels=labels,  # Apply uppercase labels
        cbar_kws={'label': 'AUC'},
        annot_kws={"size": 10, "weight": "bold"},
        vmin=vmin,
        vmax=vmax
    )
    plt.title(f"AUROC Heatmap for {backbone.upper()}", fontsize=16, weight="bold")
    plt.xlabel("Scale 1", fontsize=12, weight="bold")
    plt.ylabel("Scale 2", fontsize=12, weight="bold")
    plt.xticks(fontsize=10, rotation=45, weight="bold")
    plt.yticks(fontsize=10, weight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"heatmap_es_oid_pid_{backbone}.png"), dpi=300)
    plt.close()

# 9) compute AUC matrices and plot

if __name__ == "__main__":
    OUTPUT_DIR = "./output_heatmaps"
    BACKBONES = ['uni', 'conch', 'ctranspath', 'inception']
    FILE_IDS = [1, 2, 3, 4]

    auc_matrices = [test_combinations(backbone, FILE_IDS) for backbone in BACKBONES]
    global_min, global_max = get_global_min_max(auc_matrices)

    for backbone, auc_matrix in zip(BACKBONES, auc_matrices):
        plot_backbone_heatmap(auc_matrix, FILE_IDS, backbone, global_min, global_max, OUTPUT_DIR)
