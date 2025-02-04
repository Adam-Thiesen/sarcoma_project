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
    precision_score, recall_score, f1_score, accuracy_score,
    ConfusionMatrixDisplay, precision_recall_curve
)


##########################
# process_h5ad
##########################
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


##########################
# extract_histological_type
##########################
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


##########################
# process_combination
##########################
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
    merged_df['Histological Subtype'] = hist1
    merged_df = merged_df[merged_df['Histological Subtype'].isin(['Alveolar RMS', 'Embryonal RMS'])].copy()

    # Convert string categories to numeric
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].map({
        'Alveolar RMS': 0,
        'Embryonal RMS': 1
    })

    # Count the occurrences
    type_counts = merged_df['Histological Subtype'].value_counts()
    print(f"Number of 0s (Alveolar): {type_counts.get(0, 0)}")
    print(f"Number of 1s (Embryonal): {type_counts.get(1, 0)}")

    # Ensure 'Tissue ID' is a string
    merged_df['Tissue ID'] = merged_df['Tissue ID'].astype(str)

    # Filter for 'Tissue ID' ending with '.oid0'
    final_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    if final_df.empty:
        raise ValueError(f"No rows with 'Tissue ID' ending in '.oid0' in file: {file1} or {file2}")

    # If you need just the filtered subset, do:
    merged_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    
    return merged_df


##########################
# run_crossval_and_plot_roc
##########################
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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   CROSS-VALIDATION LOOP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for iteration in range(num_iterations):
        sgkf = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=iteration
        )

        for train_idx, test_idx in sgkf.split(data, labels, groups=groups):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # T-test feature selection
            _, pvals = sp.stats.ttest_ind(
                X_train[y_train == 0], 
                X_train[y_train == 1], 
                axis=0, equal_var=False
            )
            sel_features = np.where(pvals < 0.05)[0]
            if len(sel_features) == 0:
                # If no features survive, skip
                continue

            model = LR(
                penalty='l1', C=100, class_weight='balanced',
                solver='liblinear', max_iter=2000
            )
            model.fit(X_train[:, sel_features], y_train)

            probs = model.predict_proba(X_test[:, sel_features])[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs, drop_intermediate=False)

            # Force the curve to start at (0,0) and end at (1,1)
            fpr = np.insert(fpr,  0, 0.0)
            tpr = np.insert(tpr,  0, 0.0)
            fpr = np.append(fpr,  1.0)
            tpr = np.append(tpr,  1.0)

            fold_auc = auc(fpr, tpr)

            # Store for later averaging
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aucs.append(fold_auc)

            # Store for confusion matrix & PR curve
            all_probs.extend(probs)
            all_true.extend(y_test)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   AVERAGE ROC PLOT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list = []
    for fpr_arr, tpr_arr in zip(all_fprs, all_tprs):
        interp_tpr = np.interp(mean_fpr, fpr_arr, tpr_arr)
        tpr_list.append(interp_tpr)
    tpr_list = np.array(tpr_list)

    mean_tpr = tpr_list.mean(axis=0)
    std_tpr = tpr_list.std(axis=0)

    mean_auc_tpr = auc(mean_fpr, mean_tpr)
    mean_auc_all = np.mean(all_aucs)
    std_auc_all  = np.std(all_aucs)

    plt.figure(figsize=(6, 5))

    # Insert corners for mean ROC
    mean_fpr = np.insert(mean_fpr, 0, 0.0)
    mean_tpr = np.insert(mean_tpr, 0, 0.0)
    mean_fpr = np.append(mean_fpr, 1.0)
    mean_tpr = np.append(mean_tpr, 1.0)

    # Insert corners for std TPR
    std_tpr = np.insert(std_tpr, 0, std_tpr[0])
    std_tpr = np.append(std_tpr, std_tpr[-1])

    # Fill ±1 std. dev. band
    plt.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color='b', alpha=0.2,
        label=r"$\pm$ 1 std. dev. (TPR)"
    )

    # Mean ROC line
    plt.plot(
        mean_fpr, mean_tpr, 
        label=f"Mean ROC (AUC={mean_auc_all:.3f} ± {std_auc_all:.3f})",
        color='b'
    )
    # Diagonal
    plt.plot([0,1], [0,1], 'r--', label="Chance")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Average ROC ({num_iterations}x{splits}-fold CV)")
    plt.legend(loc="lower right")

    roc_path = os.path.join(output_dir, "mean_roc.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print("\n===== ROC SUMMARY =====")
    print(f"Plot saved to: {roc_path}")
    print(f"Mean AUC (fold-level)  : {mean_auc_all:.3f} ± {std_auc_all:.3f}")
    print(f"Mean AUC (interp. TPR) : {mean_auc_tpr:.3f}")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   CONFUSION MATRICES AT SELECTED THRESHOLDS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    thresholds = [0.3, 0.5, 0.7]
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)

    for thres in thresholds:
        sum_conf_mat = np.zeros((2, 2), dtype=int)
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

        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=percentage_cm,
            display_labels=["Alveolar", "Embryonal"]
        )
        disp.plot(cmap="Blues", ax=ax, values_format=".2f")

        # Increase font size for text inside squares
        for text_obj in disp.text_.ravel():
            text_obj.set_fontsize(14)
            val_str = text_obj.get_text()
            if "%" in val_str:
                val_str = val_str.replace("%", "")
            try:
                val_float = float(val_str)
                text_obj.set_text(f"{val_float:.2f}%")
            except ValueError:
                text_obj.set_text("0%")  # fallback

        ax.set_yticks([0, 1])  
        ax.set_yticklabels(["Alveolar", "Embryonal"], rotation=90, va='center')
        ax.set_ylabel("True label", rotation=90, labelpad=15)
        ax.set_xlabel("Predicted label")
        ax.set_title(f"Confusion Matrix at threshold={thres}")

        cm_path = os.path.join(output_dir, f"confmat_threshold_{thres}.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()

        precision_val = precision_score(all_true, preds_custom, zero_division=0)
        recall_val = recall_score(all_true, preds_custom, zero_division=0)
        accuracy_val = accuracy_score(all_true, preds_custom)
        f1_val = f1_score(all_true, preds_custom, zero_division=0)

        print(f"\nConfusion matrix at threshold={thres} saved: {cm_path}")
        print(f"== Metrics at threshold={thres} ==")
        print(f" Precision: {precision_val:.3f}")
        print(f" Recall:    {recall_val:.3f}")
        print(f" Accuracy:  {accuracy_val:.3f}")
        print(f" F1 Score:  {f1_val:.3f}")


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   PRECISION, RECALL, AND F1 CURVE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute precision and recall for all thresholds
    precisions, recalls, pr_thresholds = precision_recall_curve(all_true, all_probs, pos_label=1)
    # Compute F1 = 2*(P*R)/(P+R)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    # We will plot vs. `pr_thresholds`, which has one fewer element than precision/recall
    # so we'll plot precisions[:-1], recalls[:-1], f1_scores[:-1]
    plt.figure(figsize=(6, 5))
    plt.plot(pr_thresholds, precisions[:-1], label="Precision")
    plt.plot(pr_thresholds, recalls[:-1], label="Recall")
    plt.plot(pr_thresholds, f1_scores[:-1], label="F1 Score")

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs. Threshold")
    plt.legend(loc="best")

    prf1_path = os.path.join(output_dir, "precision_recall_f1_curve.png")
    plt.savefig(prf1_path, dpi=300)
    plt.close()

    print(f"Saved Precision/Recall/F1 curve to: {prf1_path}")


##########################
# MAIN
##########################
if __name__ == "__main__":
    # Example usage:
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


