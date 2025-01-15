import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
import scipy as sp

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.metrics import ConfusionMatrixDisplay



# process_h5ad file containing metadata and features

def process_h5ad(file_path):
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

        feature_data = f['X'][:]
        var_names = f['var/_index'][:].astype(str)  # column names

    feature_df = pd.DataFrame(feature_data, columns=var_names)

    if 'Tissue ID' in metadata_df.columns:
        metadata_df['Tissue ID'] = (
            metadata_df['Tissue ID']
            .astype(str)
            .str.replace(r"^b'|'$", "", regex=True)
        )
        metadata_df.index = metadata_df['Tissue ID']
        feature_df.index = metadata_df.index

    return metadata_df, feature_df



# extract_histological_type

def extract_histological_type(file_path):
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
    different features (columns).
    """
    meta1, feat1 = process_h5ad(file1)
    meta2, feat2 = process_h5ad(file2)

    common_idx = meta1.index.intersection(meta2.index)
    if len(common_idx) == 0:
        print("No matching samples.")
        return pd.DataFrame()

    meta1 = meta1.loc[common_idx].copy()
    feat1 = feat1.loc[common_idx].copy()
    meta2 = meta2.loc[common_idx].copy()
    feat2 = feat2.loc[common_idx].copy()

    combined_meta = meta1
    combined_feat = pd.concat([feat1, feat2], axis=1)
    merged_df = pd.concat([combined_meta, combined_feat], axis=1)

    # histological subtype
    hist1 = extract_histological_type(file1)
    hist2 = extract_histological_type(file2)
    if len(hist1) != len(meta1) or len(hist2) != len(meta2):
        print("Subtype array mismatch.")
        return pd.DataFrame()

    # For demonstration, let's do a simple binarization:
    # If your real classes are "Alveolar RMS" vs "Embryonal RMS", map them to 0,1
    # or "Ewing Sarcoma" vs. something else, etc. 
    # Just adapt as needed:
    merged_df['Histological Subtype'] = hist1
    # Example: 
    merged_df = merged_df[merged_df['Histological Subtype'].isin(['Alveolar RMS', 'Embryonal RMS'])].copy()

    # Convert string categories to numeric
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].map({
        'Alveolar RMS': 0,
        'Embryonal RMS': 1
    })
    # Count the occurrences of 0s and 1s in the 'Histological Type' column
    type_counts = merged_df['Histological Subtype'].value_counts()

    # Print the counts
    print(f"Number of 0s (Alveolar): {type_counts.get(0, 0)}")
    print(f"Number of 1s (Embryonal): {type_counts.get(1, 0)}")


    # Ensure 'Tissue ID' is a string
    merged_df['Tissue ID'] = merged_df['Tissue ID'].astype(str)

    # Filter for 'Tissue ID' ending with '.oid0'
    final_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    if final_df.empty:
        raise ValueError(f"No rows with 'Tissue ID' ending in '.oid0' in file: {file_path}")
    # Filter Tissue ID if needed
    if 'Tissue ID' in merged_df.columns:
        merged_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]

    return merged_df


def run_crossval_and_plot_roc(filtered_df, 
                              output_dir="./output_roc", 
                              num_iterations=100, 
                              splits=5):
    os.makedirs(output_dir, exist_ok=True)
    
    data = filtered_df.iloc[:, 3:-1].to_numpy()
    labels = filtered_df.iloc[:, -1].to_numpy()
    groups = filtered_df['Patient ID'].to_numpy()

    # 1) ROC data
    all_fprs = []
    all_tprs = []
    all_aucs = []

    # 2) Evaluate model, store probabilities for each fold
    all_probs = []
    all_true = []

    for iteration in range(num_iterations):
        # Move StratifiedGroupKFold creation *inside* the loop
        sgkf = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=iteration  # use the iteration as the seed
        )

        for train_idx, test_idx in sgkf.split(data, labels, groups=groups):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # T-test feature selection
            _, pvals = sp.stats.ttest_ind(
                X_train[y_train == 0], 
                X_train[y_train == 1], 
                axis=0,
                equal_var=False
            )
            sel_features = np.where(pvals < 0.05)[0]
            if len(sel_features) == 0:
                continue

            model = LR(
                penalty='l1',
                C=100,
                class_weight='balanced',
                solver='liblinear',
                max_iter=2000
            )
            model.fit(X_train[:, sel_features], y_train)

            probs = model.predict_proba(X_test[:, sel_features])[:, 1]
            # For ROC, we compute at *all* thresholds
            fpr, tpr, _ = roc_curve(y_test, probs)
            fold_auc = auc(fpr, tpr)
            all_aucs.append(fold_auc)
            all_fprs.append(fpr)
            all_tprs.append(tpr)

            # For confusion matrix, we'll store true labels & probabilities
            all_probs.extend(probs)
            all_true.extend(y_test)

    
    #       PLOT THE AVERAGE ROC (existing logic)
    
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    for fpr, tpr in zip(all_fprs, all_tprs):
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        tpr_list.append(interp_tpr)

    tpr_list = np.array(tpr_list)  
    mean_tpr = tpr_list.mean(axis=0)
    std_tpr = tpr_list.std(axis=0)

    mean_auc = auc(mean_fpr, mean_tpr)
    avg_of_fold_aucs = np.mean(all_aucs)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_fpr, mean_tpr, label=f"Mean ROC (AUC={mean_auc:.3f})", color='b')
    plt.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color='b', alpha=0.2,
        label=r"$\pm$ 1 std. dev."
    )
    plt.plot([0,1], [0,1], 'r--', label="Chance")
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Average ROC ({num_iterations}x{splits}-fold CV)")
    plt.legend(loc="lower right")

    roc_path = os.path.join(output_dir, "new_conch1_conch2_best_combo_mean_roc.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"\n===== ROC SUMMARY =====")
    print(f"Mean ROC curve saved: {roc_path}")
    print(f"Mean AUC of mean ROC: {mean_auc:.3f}")
    print(f"Average of per-fold AUCs: {avg_of_fold_aucs:.3f}")

    
    #       CONFUSION MATRIX FOR MULTIPLE THRESHOLDS
    
    # Cnfusion matrices at threshold 0.3, 0.5, 0.7

    thresholds = [0.3, 0.5, 0.7]
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    for thres in thresholds:
        # Accumulate confusion matrix
        sum_conf_mat = np.zeros((2, 2), dtype=int)

        # Use the entire set of predictions & labels:
        preds_custom = (all_probs >= thres).astype(int)
        cm = confusion_matrix(all_true, preds_custom)
        sum_conf_mat += cm

        # Convert to row-wise percentages
        conf_mat_float = sum_conf_mat.astype(float)
        row_sums = conf_mat_float.sum(axis=1, keepdims=True)
        percentage_cm = np.divide(
            conf_mat_float, 
            row_sums, 
            out=np.zeros_like(conf_mat_float),
            where=(row_sums != 0)
        ) * 100

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(6, 5))

        # Pass custom display labels
        disp = ConfusionMatrixDisplay(
            confusion_matrix=percentage_cm,
            display_labels=["Alveolar", "Embryonal"]
        )
        disp.plot(cmap="Blues", ax=ax, values_format=".2f")

        # Increase font size for text inside squares
        for text_obj in disp.text_.ravel():
            text_obj.set_fontsize(14)
            val_str = text_obj.get_text()  # e.g. "12.35"
            if "%" in val_str:
                val_str = val_str.replace("%", "")  # remove the '%' if present
            try:
                val_float = float(val_str)
                text_obj.set_text(f"{val_float:.2f}%")
            except ValueError:
                text_obj.set_text("0%")  # fallback

        # ---- Here is where we rotate the y-axis ticks and label ----
        # Ensure the ticks match your two classes
        ax.set_yticks([0, 1])  # positions for 2 classes
        ax.set_yticklabels(["Alveolar", "Embryonal"], rotation=90, va='center')

        ax.set_ylabel("True label", rotation=90, labelpad=15)

        # Optionally rotate the x-axis labels as well, if you prefer:
        # ax.set_xticklabels(["Alveolar", "Embryonal"], rotation=90, va='center')

        ax.set_xlabel("Predicted label")

        ax.set_title(f"Confusion Matrix at threshold={thres}")

        plt.tight_layout()  # Make sure nothing is cut off

        cm_path = os.path.join(output_dir, f"new_improved_confmat3_percent_threshold_{thres}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        print(f"Confusion matrix at threshold={thres} saved: {cm_path}")

        print("Here")

        precision_val = precision_score(all_true, preds_custom, zero_division=0)
        recall_val = recall_score(all_true, preds_custom, zero_division=0)
        accuracy_val = accuracy_score(all_true, preds_custom)
        f1_val = f1_score(all_true, preds_custom, zero_division=0)

        print("Here 2")

        print(f"== Metrics at threshold={thres} ==")
        print(f" Precision: {precision_val:.3f}")
        print(f" Recall:    {recall_val:.3f}")
        print(f" Accuracy:  {accuracy_val:.3f}")
        print(f" F1 Score:  {f1_val:.3f}")
        print("")




# MAIN

if __name__ == "__main__":
    # EXAMPLE: best combo is conch-2 + conch-3
    file1 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-1.h5ad"
    file2 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-2.h5ad"

    filtered_df = process_combination(file1, file2)
    if filtered_df.empty:
        print("No data for best combo.")
    else:
        run_crossval_and_plot_roc(
            filtered_df,
            output_dir="./output_best_combo_roc",
            num_iterations=100,  
            splits=5
        )
