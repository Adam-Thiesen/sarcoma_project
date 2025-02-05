import numpy as np
import pandas as pd
import h5py
import os
import scipy as sp
import pickle

from sklearn.linear_model import LogisticRegression as LR


# process_h5ad

def process_h5ad(file_path, prefix=""):
    """
    Loads .h5ad data, returns (metadata_df, feature_df).
    feature_df columns are renamed with `prefix` + original_name
    to avoid collisions.
    """
    with h5py.File(file_path, "r") as f:
        # Access metadata
        metadata = {}
        for key in f['obs'].keys():
            if isinstance(f[f'obs/{key}'], h5py.Dataset):
                metadata[key] = f[f'obs/{key}'][:]
        
        # Convert metadata to DataFrame
        metadata_df = pd.DataFrame(metadata)

        # Check if 'Patient ID' exists with 'categories' and 'codes'
        if 'Patient ID' in f['obs']:
            try:
                categories = f['obs/Patient ID/categories'][:]
                codes = f['obs/Patient ID/codes'][:]
                # Decode categories if necessary
                categories = [x.decode('utf-8') if isinstance(x, bytes) else x for x in categories]
                metadata_df['Patient ID'] = [categories[code] for code in codes]
            except Exception as e:
                print(f"Could not process 'Patient ID': {e}")

        # Feature matrix and var names
        feature_data = f['X'][:]
        var_names = f['var/_index'][:].astype(str)  # column names

    # Rename feature columns with the prefix
    renamed_columns = [f"{prefix}_{col}" for col in var_names]
    feature_df = pd.DataFrame(feature_data, columns=renamed_columns)

    # Clean up Tissue ID if present
    if 'Tissue ID' in metadata_df.columns:
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID']
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
        )
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index = metadata_df.index

    return metadata_df, feature_df


# extract_histological_subtype

def extract_histological_type(file_path):
    """
    Extracts the "Histological Subtype" from file_path
    as a list of strings. 
    """
    try:
        with h5py.File(file_path, "r") as f:
            categories = f['obs/Histological Subtype/categories'][:]
            codes = f['obs/Histological Subtype/codes'][:]
            categories = [x.decode('utf-8') if isinstance(x, bytes) else x for x in categories]
            histological_type = [categories[code] for code in codes]
        return histological_type
    except Exception as e:
        print(f"Error extracting histological type from {file_path}: {e}")
        return []


# process_combination

def process_combination(file1, file2):
    """
    Merges two .h5ad files with the same samples (rows),
    different features (columns), and binarizes Histological Subtype
    (Alveolar=0, Embryonal=1). Returns merged DataFrame.
    
    Optionally enforce tissue = .oid0
    """
    # Use fixed prefix so we have exactly "conch-1" and "conch-2" keys
    prefix1 = "conch-1"
    prefix2 = "conch-2"

    # Load data with unique prefixes
    meta1, feat1 = process_h5ad(file1, prefix=prefix1)
    meta2, feat2 = process_h5ad(file2, prefix=prefix2)

    # Intersect on common Tissue IDs
    common_idx = meta1.index.intersection(meta2.index)
    if len(common_idx) == 0:
        print("No matching samples between the two .h5ad files.")
        return pd.DataFrame()

    meta1 = meta1.loc[common_idx].copy()
    feat1 = feat1.loc[common_idx].copy()
    meta2 = meta2.loc[common_idx].copy()
    feat2 = feat2.loc[common_idx].copy()

    combined_meta = meta1  # they should match, but we'll just take meta1
    combined_feat = pd.concat([feat1, feat2], axis=1)
    merged_df = pd.concat([combined_meta, combined_feat], axis=1)

    # Extract histological subtypes
    hist1 = extract_histological_type(file1)
    hist2 = extract_histological_type(file2)
    # hist1 and hist2 should each match the length of meta for each file
    if (len(hist1) != len(meta1)) or (len(hist2) != len(meta2)):
        print("Histological subtype array mismatch. Check data.")
        return pd.DataFrame()
    
    # only need one set of subtypes because the rows match
    merged_df['Histological Subtype'] = hist1

    # filter only Alveolar / Embryonal
    merged_df = merged_df[merged_df['Histological Subtype'].isin(['Alveolar RMS', 'Embryonal RMS'])].copy()

    # binarize (0=Alveolar, 1=Embryonal)
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].map({
        'Alveolar RMS': 0,
        'Embryonal RMS': 1
    })

    # Tissue ID must end with '.oid0' (uncomment if needed)
    # merged_df['Tissue ID'] = merged_df['Tissue ID'].astype(str)
    # merged_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    # if merged_df.empty:
    #    raise ValueError("No rows with 'Tissue ID' ending in '.oid0'.")

    return merged_df


# train_final_model

def train_final_model(filtered_df, output_clf_path="alv_emb_clf3.pkl"):
    """
    Train on entire dataset (no cross-validation), using T-test
    to select features with p < 0.05. Save final model as `output_clf_path`.
    
    Adds an attribute `model.features` that shows:
      {
        "conch-1": [feature names kept],
        "conch-2": [feature names kept]
      }
    """
    #   merged_df.iloc[:, 3:-1] = features
    #   merged_df.iloc[:, -1]   = label
    
    X = filtered_df.iloc[:, 3:-1].to_numpy()
    y = filtered_df.iloc[:, -1].to_numpy()

    # Column names for the features
    feature_cols = filtered_df.columns[3:-1]

    # T-test feature selection on the entire dataset
    group0 = X[y == 0]  # Alveolar
    group1 = X[y == 1]  # Embryonal
    _, pvals = sp.stats.ttest_ind(group0, group1, axis=0, equal_var=False)
    sel_mask = (pvals < 0.05)
    if not any(sel_mask):
        raise ValueError("No features passed the p<0.05 threshold. Model cannot be trained.")

    # Filter to selected features
    X_sel = X[:, sel_mask]
    selected_feature_names = feature_cols[sel_mask]

    # Fit logistic regression on entire dataset
    model = LR(
        penalty='l1', 
        C=100, 
        class_weight='balanced',
        solver='liblinear', 
        max_iter=2000
    )
    model.fit(X_sel, y)

    # Build a dictionary that records which features came from which conch
    feature_dict = {
        "conch-1": [],
        "conch-2": []
    }
    for feat in selected_feature_names:
        # We assume each column name is "conch-1_originalFeature" or "conch-2_originalFeature".
        if feat.startswith("conch-1_"):
            # Remove the "conch-1_" portion:
            original_name = feat.replace("conch-1_", "")
            feature_dict["conch-1"].append(original_name)
        elif feat.startswith("conch-2_"):
            # Remove the "conch-2_" portion:
            original_name = feat.replace("conch-2_", "")
            feature_dict["conch-2"].append(original_name)
        else:
            # If for some reason there's neither conch-1 nor conch-2, you can add logic here:
            pass

    # Add the dictionary to the model (a custom attribute)
    model.features = feature_dict

    # Pickle (save) the trained model
    with open(output_clf_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nModel trained and saved to {output_clf_path}.")
    print("Selected features by conch prefix:")
    for pfx, feats in feature_dict.items():
        print(f"  {pfx}: {len(feats)} features")
        # Uncomment to print the actual names:
        # for name in feats:
        #     print(f"    {name}")



if __name__ == "__main__":
    # Example usage (adjust paths as needed):
    file1 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-1.h5ad"
    file2 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-2.h5ad"

    filtered_df = process_combination(file1, file2)
    if filtered_df.empty:
        print("No data after merging or mismatch encountered. Exiting.")
    else:
        train_final_model(filtered_df, output_clf_path="alv_emb_clf3.pkl")
