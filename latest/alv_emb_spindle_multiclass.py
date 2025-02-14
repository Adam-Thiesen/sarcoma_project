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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import label_binarize

from sklearn.metrics import classification_report

# Adjust some matplotlib defaults
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})


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
                categories = [
                    x.decode('utf-8') if isinstance(x, bytes) else x 
                    for x in categories
                ]
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
            categories = [
                x.decode('utf-8') if isinstance(x, bytes) else x 
                for x in categories
            ]
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

    # Attach the Histological Subtype from the first file 
    # (assuming file1 and file2 are from the same set of patients).
    merged_df['Histological Subtype'] = hist1

    # ------------------------------------------------------------------------
    # Filter for the 3 classes of interest and map them to numeric [0,1,2].
    # Adjust these exact string matches based on your file's content.
    # For example, if your actual keys are "Alveolar RMS", "Embryonal RMS", "Spindle Cell RMS",
    # then set them here exactly.
    # ------------------------------------------------------------------------
    valid_classes = ["Alveolar RMS", "Embryonal RMS", "Spindle Cell RMS"]
    merged_df = merged_df[merged_df['Histological Subtype'].isin(valid_classes)].copy()

    class_map = {
        "Alveolar RMS": 0,
        "Embryonal RMS": 1,
        "Spindle Cell RMS": 2
    }
    merged_df['Histological Subtype'] = merged_df['Histological Subtype'].map(class_map)

    # Ensure 'Tissue ID' is a string
    merged_df['Tissue ID'] = merged_df['Tissue ID'].astype(str)

    # Filter for 'Tissue ID' ending with '.oid0'
    final_df = merged_df[merged_df['Tissue ID'].str.endswith('.oid0')]
    if final_df.empty:
        raise ValueError("No rows with 'Tissue ID' ending in '.oid0'.")
    
    # If you only want those Tissue IDs in your final set:
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
    
    # -------------------------------------------------------------
    # For convenience, let's define columns:
    #   - The first 3 columns: Tissue ID, Patient ID, <maybe others>
    #   - Then feature columns from col index 3 to -1
    #   - The last column (col index -1) is "Histological Subtype"
    # -------------------------------------------------------------
    data = filtered_df.iloc[:, 3:-1].to_numpy()
    labels = filtered_df.iloc[:, -1].to_numpy()
    groups = filtered_df['Patient ID'].to_numpy()

    # We'll do a 3-class OVR logistic regression. 
    # For feature selection, we can do a one-way ANOVA (f_classif).
    from sklearn.feature_selection import f_classif
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc

    # We will store per-class ROC data for each iteration/fold
    # so that we can compute a macro-average or micro-average ROC.
    all_fprs = {0:[], 1:[], 2:[]}
    all_tprs = {0:[], 1:[], 2:[]}
    all_aucs = {0:[], 1:[], 2:[]}

    # We'll also store the final predictions and ground truth for confusion matrix
    all_preds = []
    all_true = []
    

    classes = np.array([0, 1, 2])  # your 3 classes

    for iteration in range(num_iterations):
        # Move StratifiedGroupKFold inside the loop 
        sgkf = StratifiedGroupKFold(
            n_splits=splits,
            shuffle=True,
            random_state=iteration  # use the iteration as the seed
        )

        for train_idx, test_idx in sgkf.split(data, labels, groups=groups):
            X_train, X_test = data[train_idx], data[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # ===== 1) Feature Selection via ANOVA (for multi-class) =====
            F, pvals = f_classif(X_train, y_train)
            sel_features = np.where(pvals < 0.05)[0]
            if len(sel_features) == 0:
                # If no features pass, skip this fold.
                continue

            # ===== 2) One-vs-Rest Logistic Regression =====
            model = LR(
                penalty='l1',
                C=100,
                class_weight='balanced',
                solver='liblinear',
                max_iter=2000,
                multi_class='ovr'
            )
            model.fit(X_train[:, sel_features], y_train)

            # ===== 3) Predict Probability (for each of the 3 classes) =====
            probs = model.predict_proba(X_test[:, sel_features])  # shape (n_test, 3)

            # We'll gather ROC data by "binarizing" the y_test and 
            # computing ROC curve for each class vs rest.
            # For confusion matrix, we'll take argmax(probs).
            custom_pred = []
            spindle_threshold = 0.00023  # <--- tweak this as desired

            for row_probs in probs:
                p_alv = row_probs[0]  # Probability for Alveolar
                p_emb = row_probs[1]  # Probability for Embryonal
                p_spi = row_probs[2]  # Probability for Spindle

                if p_spi >= spindle_threshold:
                    # Force it to be Spindle Cell if above threshold
                    custom_pred.append(2)
                else:
                    # Otherwise pick the better of Alveolar vs Embryonal
                    custom_pred.append(np.argmax(row_probs[:2]))  # 0 if p_alv > p_emb, else 1

            y_pred = np.array(custom_pred)
            all_preds.append(y_pred)
            all_true.append(y_test)
            

            # --- One-vs-Rest ROC for each class ---
            # First, binarize y_test for each class
            # e.g. for class k: y_test_binary = (y_test == k).astype(int)
            for k in classes:
                y_test_bin = (y_test == k).astype(int)
                probs_k    = probs[:, k]  # Probability that the sample is class k
                fpr, tpr, _ = roc_curve(y_test_bin, probs_k)
                roc_auc = auc(fpr, tpr)
                all_fprs[k].append(fpr)
                all_tprs[k].append(tpr)
                all_aucs[k].append(roc_auc)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       ROC PLOTS (Macro-average across classes)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We'll compute a "macro-average" by first averaging within each class 
    # and then averaging across classes. One way is to interpolate each 
    # class's fpr/tpr, average them, then average across classes.
    # For demonstration, here's one approach.

    mean_fpr = np.linspace(0, 1, 100)

    # For storing "per-class average TPR"
    class_mean_tpr = {}
    class_mean_auc = {}

    for k in classes:
        tpr_list = []
        for fpr, tpr in zip(all_fprs[k], all_tprs[k]):
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            tpr_list.append(interp_tpr)
        tpr_list = np.array(tpr_list)
        mean_tpr = tpr_list.mean(axis=0)
        class_mean_tpr[k] = mean_tpr
        class_mean_auc[k] = np.mean(all_aucs[k])  # average AUC for that class

    # Macro-average: average the mean TPRs across the 3 classes
    macro_tpr = np.mean(np.array(list(class_mean_tpr.values())), axis=0)
    macro_auc = auc(mean_fpr, macro_tpr)

    plt.figure(figsize=(7, 6))

    # Plot each class's average ROC
    colors = ['blue', 'green', 'red']
    class_labels = ["Alveolar", "Embryonal", "Spindle Cell"]

    for idx, k in enumerate(classes):
        plt.plot(
            mean_fpr,
            class_mean_tpr[k],
            color=colors[idx],
            label=f"Class {class_labels[idx]} avg AUC={class_mean_auc[k]:.3f}",
            alpha=0.8
        )

    # Then plot the macro-average
    plt.plot(
        mean_fpr,
        macro_tpr,
        color='black',
        label=f"Macro-average ROC (AUC={macro_auc:.3f})",
        linestyle='--',
        linewidth=2
    )

    plt.plot([0,1], [0,1], 'r--', label="Chance", alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"3-Class Average ROC Curves")
    plt.legend(loc="lower right")

    roc_path = os.path.join(output_dir, "extreme5_threshold_improved34_multiclass_mean_roc.pdf")
    plt.savefig(roc_path, dpi=300, transparent=True)
    plt.close()

    print(f"\n===== ROC SUMMARY =====")
    print(f"Macro-average ROC curve saved: {roc_path}")
    print(f"Macro-average AUC: {macro_auc:.3f}")
    for idx, k in enumerate(classes):
        print(f"  - {class_labels[idx]} one-vs-rest AUC (avg): {class_mean_auc[k]:.3f}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       CONFUSION MATRIX (3-Class)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We collect all predictions across folds/iterations,
    # then build a single confusion matrix. 
    # (Or you could average confusion matrices per fold, etc.)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    classes = [0, 1, 2]
    class_labels = ["Alveolar", "Embryonal", "Spindle"]
    
    report = classification_report(
    all_true, 
    all_preds, 
    labels=classes, 
    target_names=class_labels, 
    digits=3
    )
    print(report)


    # Make the confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=classes)

    # Make the confusion matrix
    cm = confusion_matrix(all_true, all_preds, labels=classes)

    # Convert to row-wise percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc = (cm / row_sums) * 100  # handle zero division as needed

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_perc, 
        display_labels=class_labels
    )
    disp.plot(cmap="Blues", ax=ax, values_format=".2f")

    # Overwrite text with a '%' sign
    for text_obj in disp.text_.ravel():
        val_str = text_obj.get_text()  # e.g. '10.45'
        try:
            val_float = float(val_str)
            text_obj.set_text(f"{val_float:.2f}%")
        except ValueError:
            # fallback if needed
            text_obj.set_text("0%")

    #ax.set_title("Confusion Matrix (Row-wise Percent)")
    plt.tight_layout()
    plt.savefig("extreme5_threshold_improved34_multiclass_percent_confmat.pdf", dpi=300, transparent=True)
    plt.close()



##########################
# MAIN
##########################
if __name__ == "__main__":
    # EXAMPLE: best combo is conch-1 + conch-2
    file1 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-1.h5ad"
    file2 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.conch-2.h5ad"

    filtered_df = process_combination(file1, file2)
    if filtered_df.empty:
        print("No data for best combo.")
    else:
        run_crossval_and_plot_roc(
            filtered_df,
            output_dir="./output_best_combo_roc",
            num_iterations=10,  # reduce for quick tests, increase for better estimates
            splits=5
        )
