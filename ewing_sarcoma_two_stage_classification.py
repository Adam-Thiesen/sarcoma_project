import pandas as pd
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import scipy as sp
import os


# 1. Process H5AD File (single file)
def process_h5ad_with_labels(file_path):
    """
    Reads a single H5AD file, extracts metadata, features,
    and adds RMS vs. NRSTS (RMS_Label) and Ewing vs. Other (Ewing_Label) columns.
    """
    with h5py.File(file_path, "r") as f:
        metadata = {}
        for key in f['obs']:
            if isinstance(f[f'obs/{key}'], h5py.Dataset):
                metadata[key] = f[f'obs/{key}'][:]

        metadata_df = pd.DataFrame(metadata)

        # Process 'Patient ID'
        if 'Patient ID' in f['obs']:
            categories = f['obs/Patient ID/categories'][:]
            codes = f['obs/Patient ID/codes'][:]
            categories = [x.decode('utf-8') if isinstance(x, bytes) else x for x in categories]
            metadata_df['Patient ID'] = [categories[code] for code in codes]

        # Process 'Histological Subtype' and 'Histological Type'
        if 'Histological Subtype' in f['obs']:
            categories = f['obs/Histological Subtype/categories'][:]
            codes = f['obs/Histological Subtype/codes'][:]
            categories = [x.decode('utf-8') if isinstance(x, bytes) else x for x in categories]
            metadata_df['Histological Subtype'] = [categories[code] for code in codes]

        if 'Histological Type' in f['obs']:
            categories = f['obs/Histological Type/categories'][:]
            codes = f['obs/Histological Type/codes'][:]
            categories = [x.decode('utf-8') if isinstance(x, bytes) else x for x in categories]
            metadata_df['Histological Type'] = [categories[code] for code in codes]

        # Add RMS vs. NRSTS labels (1=RMS, 0=NRSTS)
        metadata_df['RMS_Label'] = metadata_df['Histological Type'].apply(
            lambda x: 1 if x == 'RMS' else (0 if x == 'NRSTS' else np.nan)
        )

        # Add Ewing's vs. Other labels (among NRSTS, 1=Ewing, 0=Other)
        metadata_df['Ewing_Label'] = metadata_df.apply(
            lambda row: 1 if (row['Histological Subtype'] == 'Ewing Sarcoma'
                              and row['RMS_Label'] == 0) else 0,
            axis=1
        )

        # Extract feature data
        feature_data = f['X'][:]
        var_names = f['var/_index'][:].astype(str)
        feature_df = pd.DataFrame(feature_data, columns=var_names)

        # Align metadata and features by dropping rows with NaN in RMS_Label
        valid_indices = metadata_df.dropna(subset=['RMS_Label']).index
        metadata_df = metadata_df.loc[valid_indices].copy()
        metadata_df['RMS_Label'] = metadata_df['RMS_Label'].astype(int)
        feature_df = feature_df.loc[valid_indices].copy()

        # Set index to something simple (optional)
        metadata_df.index = range(len(metadata_df))
        feature_df.index = range(len(feature_df))

    return metadata_df, feature_df


# 1a. Process Two H5AD Files and Combine
def process_h5ad_with_labels_two_files(file_path1, file_path2):
    """
    Reads two H5AD files, processes them each, merges metadata, and
    concatenates features side-by-side. Returns combined metadata & features.
    """
    print(f"Reading and processing File 1: {file_path1}")
    metadata1, features1 = process_h5ad_with_labels(file_path1)

    print(f"Reading and processing File 2: {file_path2}")
    metadata2, features2 = process_h5ad_with_labels(file_path2)

    # Intersection of indexes to keep only common rows
    common_idx = metadata1.index.intersection(metadata2.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping samples between the two files.")

    metadata1 = metadata1.loc[common_idx].copy()
    metadata2 = metadata2.loc[common_idx].copy()
    features1 = features1.loc[common_idx].copy()
    features2 = features2.loc[common_idx].copy()

    # Combine metadata
    combined_metadata = metadata1.copy()
    # Concatenate features side-by-side
    combined_features = pd.concat([features1, features2], axis=1)

    return combined_metadata, combined_features


# 2. Cross-validated two-stage workflow
def cross_validated_two_stage_workflow(metadata, features, splits=5):
    """
    Perform a cross-validation to get out-of-fold predictions and probabilities
    for each stage. Then use them to build a two-stage confusion matrix & ROC curves.
    """
    data = features.to_numpy()
    stage1_labels = metadata['RMS_Label'].to_numpy()     # 1=RMS, 0=NRSTS
    stage2_labels = metadata['Ewing_Label'].to_numpy()   # 1=Ewing, 0=Other
    groups = metadata['Patient ID'].to_numpy() if 'Patient ID' in metadata.columns else None

    n_samples = len(metadata)
    out_of_fold_stage1_preds = np.full(n_samples, np.nan)
    out_of_fold_stage1_probs = np.full(n_samples, np.nan)
    out_of_fold_stage2_preds = np.full(n_samples, np.nan)
    out_of_fold_stage2_probs = np.full(n_samples, np.nan)

    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import StratifiedKFold
        sgkf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

    stage1_aucs = []
    stage2_aucs = []

    fold_num = 0
    for train_idx, test_idx in sgkf.split(data, stage1_labels, groups=groups):
        fold_num += 1
        print("\n============================================")
        print(f"Processing Fold {fold_num} / {splits}")
        print("============================================")

        X_train, X_test = data[train_idx], data[test_idx]
        y_train_stage1, y_test_stage1 = stage1_labels[train_idx], stage1_labels[test_idx]

        print(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")

        # ============= Stage 1: RMS vs. NRSTS =============
        _, p_values = sp.stats.ttest_ind(
            X_train[y_train_stage1 == 0],
            X_train[y_train_stage1 == 1],
            axis=0,
            equal_var=False
        )
        selected_features_stage1 = np.where(p_values < 0.05)[0]
        print(f"Stage 1: # features passing p<0.05 = {len(selected_features_stage1)}")

        if len(selected_features_stage1) == 0:
            print("No features passed p<0.05. Using all features as fallback.")
            selected_features_stage1 = np.arange(X_train.shape[1])

        model_stage1 = LR(
            penalty='l1',
            C=100,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000
        )
        model_stage1.fit(X_train[:, selected_features_stage1], y_train_stage1)

        stage1_probs_test = model_stage1.predict_proba(X_test[:, selected_features_stage1])[:, 1]
        stage1_preds_test = (stage1_probs_test >= 0.5).astype(int)

        try:
            fold_auc_stage1 = roc_auc_score(y_test_stage1, stage1_probs_test)
            stage1_aucs.append(fold_auc_stage1)
            print(f"Stage 1 AUC (Fold {fold_num}): {fold_auc_stage1:.3f}")
        except Exception as e:
            print(f"Could not compute Stage 1 AUC (Fold {fold_num}): {e}")

        out_of_fold_stage1_preds[test_idx] = stage1_preds_test
        out_of_fold_stage1_probs[test_idx] = stage1_probs_test

        # ============= Stage 2: Ewing vs. Other (only among NRSTS) =============
        train_nrsts_mask = (y_train_stage1 == 0)
        train_nrsts_idx = train_idx[train_nrsts_mask]
        X_train_nrsts = data[train_nrsts_idx]
        y_train_stage2 = stage2_labels[train_nrsts_idx]  # 1=Ewing, 0=Other

        _, p_values2 = sp.stats.ttest_ind(
            X_train_nrsts[y_train_stage2 == 0],
            X_train_nrsts[y_train_stage2 == 1],
            axis=0,
            equal_var=False
        )
        selected_features_stage2 = np.where(p_values2 < 0.05)[0]
        print(f"Stage 2: # features passing p<0.05 = {len(selected_features_stage2)}")

        if len(selected_features_stage2) == 0:
            print("No features passed p<0.05 for Stage 2. Using all features as fallback.")
            selected_features_stage2 = np.arange(X_train_nrsts.shape[1])

        model_stage2 = LR(
            penalty='l1',
            C=100,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000
        )
        model_stage2.fit(X_train_nrsts[:, selected_features_stage2], y_train_stage2)

        # Predict on test samples that were predicted NRSTS by Stage 1
        test_pred_nrsts_idx = test_idx[stage1_preds_test == 0]
        print(f"Stage 2 Test subset size (predicted NRSTS) for Fold {fold_num}: {len(test_pred_nrsts_idx)}")

        if len(test_pred_nrsts_idx) > 0:
            X_test_nrsts = data[test_pred_nrsts_idx][:, selected_features_stage2]
            stage2_probs_test = model_stage2.predict_proba(X_test_nrsts)[:, 1]
            stage2_preds_test = (stage2_probs_test >= 0.5).astype(int)

            out_of_fold_stage2_preds[test_pred_nrsts_idx] = stage2_preds_test
            out_of_fold_stage2_probs[test_pred_nrsts_idx] = stage2_probs_test

            # Evaluate Stage 2 AUC only among truly NRSTS in the test fold
            mask_true_nrsts = (y_test_stage1 == 0)
            test_true_nrsts_idx = test_idx[mask_true_nrsts]
            final_eval_idx = np.intersect1d(test_true_nrsts_idx, test_pred_nrsts_idx)

            print(f"Stage 2 evaluation subset (true NRSTS + predicted NRSTS) for Fold {fold_num}: {len(final_eval_idx)}")
            if len(final_eval_idx) > 1:
                y_true_ewings = stage2_labels[final_eval_idx]
                try:
                    probs_ewings = model_stage2.predict_proba(
                        data[final_eval_idx][:, selected_features_stage2]
                    )[:, 1]
                    fold_auc_stage2 = roc_auc_score(y_true_ewings, probs_ewings)
                    stage2_aucs.append(fold_auc_stage2)
                    print(f"Stage 2 AUC (Fold {fold_num}): {fold_auc_stage2:.3f}")
                except Exception as e:
                    print(f"Could not compute Stage 2 AUC for Fold {fold_num}: {e}")
            else:
                print(f"Not enough samples to compute Stage 2 AUC for Fold {fold_num}.")
        else:
            print(f"No test samples predicted as NRSTS for Fold {fold_num}, skipping Stage 2 AUC.")

    # End CV loop
    print("\nFinished Cross-Validation.")
    print(f"Number of folds with valid Stage 1 AUC: {len(stage1_aucs)}")
    print(f"Number of folds with valid Stage 2 AUC: {len(stage2_aucs)}")

    if len(stage1_aucs) > 0:
        print(f"Stage 1 Avg AUC (CV): {np.mean(stage1_aucs):.3f}")
    if len(stage2_aucs) > 0:
        print(f"Stage 2 Avg AUC (CV): {np.mean(stage2_aucs):.3f}")

    return (
        out_of_fold_stage1_preds,
        out_of_fold_stage1_probs,
        out_of_fold_stage2_preds,
        out_of_fold_stage2_probs
    )


# 3. Build two-stage confusion matrix from OOF predictions
def build_two_stage_confusion_matrix(
    metadata, stage1_preds, stage2_preds, 
    output_path="two_stage_confusion_matrix.png"
):
    """
    Build a combined confusion matrix and save to 'output_path':
      - If stage1_preds == 1 => predicted RMS
      - If stage1_preds == 0 => predicted NRSTS, then:
         stage2_preds == 1 => Ewing
         stage2_preds == 0 => Other NRSTS
    """
    combined_pred_labels = np.full(len(metadata), np.nan)

    # Stage 1: predicted RMS => label=1
    mask_pred_rms = (stage1_preds == 1)
    combined_pred_labels[mask_pred_rms] = 1

    # Stage 1: predicted NRSTS => check Stage 2
    mask_pred_nrsts = (stage1_preds == 0)
    combined_pred_labels[mask_pred_nrsts & (stage2_preds == 1)] = 2  # Ewing
    combined_pred_labels[mask_pred_nrsts & (stage2_preds == 0)] = 3  # Other NRSTS

    # Fallback for leftover NaN
    combined_pred_labels = np.where(np.isnan(combined_pred_labels), 3, combined_pred_labels)

    # True Label: 1 = RMS, 2 = Ewing, 3 = Other NRSTS
    true_labels = metadata.apply(
        lambda row: 1 if row['RMS_Label'] == 1 else (2 if row['Ewing_Label'] == 1 else 3),
        axis=1
    ).values

    cm = confusion_matrix(true_labels, combined_pred_labels, labels=[1, 2, 3])
    disp = ConfusionMatrixDisplay(cm, display_labels=['RMS', 'Ewing', 'Other NRSTS'])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (Two-Stage, CV OOF)")

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

    # Print classification report in console
    print("Classification Report (CV OOF):")
    print(classification_report(
        true_labels,
        combined_pred_labels,
        labels=[1, 2, 3],
        target_names=['RMS', 'Ewing', 'Other NRSTS']
    ))
    return cm


# 4. Plot Stage-1 and Stage-2 ROC curves and save as separate images
def plot_roc_curves(
    metadata, stage1_probs, stage2_probs,
    stage1_output_path="stage1_roc.png",
    stage2_output_path="stage2_roc.png"
):
    """
    Plots aggregated out-of-fold ROC curves separately for:
      - Stage 1: RMS vs. NRSTS
      - Stage 2: Ewing vs. Other (in truly NRSTS subset)
    Saves each plot to the specified paths.
    """
    # True labels
    true_stage1 = metadata['RMS_Label'].values  # 1=RMS, 0=NRSTS
    true_stage2 = metadata['Ewing_Label'].values  # 1=Ewing, 0=Other

    # ---- Stage 1 ROC ----
    valid_idx_stage1 = ~np.isnan(stage1_probs)
    y_true_s1 = true_stage1[valid_idx_stage1]
    y_probs_s1 = stage1_probs[valid_idx_stage1]

    if len(np.unique(y_true_s1)) < 2:
        print("Cannot compute Stage 1 ROC: only one class present.")
    else:
        fpr_s1, tpr_s1, _ = roc_curve(y_true_s1, y_probs_s1)
        auc_s1 = roc_auc_score(y_true_s1, y_probs_s1)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr_s1, tpr_s1, label=f"Stage 1 ROC (AUC={auc_s1:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Stage 1 ROC Curve: RMS vs. NRSTS (CV OOF)")
        plt.legend()

        # Save
        plt.savefig(stage1_output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Stage 1 ROC curve saved to: {stage1_output_path}")

    # ---- Stage 2 ROC ----
    valid_idx_stage2 = ~np.isnan(stage2_probs)
    truly_nrsts = (true_stage1 == 0)
    final_idx_stage2 = valid_idx_stage2 & truly_nrsts

    if np.sum(final_idx_stage2) < 2:
        print("Not enough samples to plot Stage 2 ROC (less than 2).")
        return

    y_true_s2 = true_stage2[final_idx_stage2]
    y_probs_s2 = stage2_probs[final_idx_stage2]

    if len(np.unique(y_true_s2)) < 2:
        print("Cannot compute Stage 2 ROC: only one class present.")
        return

    fpr_s2, tpr_s2, _ = roc_curve(y_true_s2, y_probs_s2)
    auc_s2 = roc_auc_score(y_true_s2, y_probs_s2)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_s2, tpr_s2, label=f"Stage 2 ROC (AUC={auc_s2:.3f})", color='orange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage 2 ROC Curve: Ewing vs. Other NRSTS (CV OOF)")
    plt.legend()

    # Save
    plt.savefig(stage2_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stage 2 ROC curve saved to: {stage2_output_path}")


# 5. Main script
if __name__ == "__main__":
    # Example: combine conch-1 and conch-2
    file_path1 = "/ad_wsi.uni-1.h5ad"
    file_path2 = "/ad_wsi.uni-4.h5ad"

    # Optional: create an output directory
    output_dir = "./output_figures"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1) Read, process, and combine both files
    metadata, features = process_h5ad_with_labels_two_files(file_path1, file_path2)

    # Step 2) Perform cross-validation to get out-of-fold predictions & probabilities
    (
        oof_stage1_preds,
        oof_stage1_probs,
        oof_stage2_preds,
        oof_stage2_probs
    ) = cross_validated_two_stage_workflow(metadata, features, splits=5)

    # Step 3) Build and save the combined confusion matrix
    cm_save_path = os.path.join(output_dir, "two_stage_confusion_matrix.png")
    combined_cm = build_two_stage_confusion_matrix(
        metadata, oof_stage1_preds, oof_stage2_preds, 
        output_path=cm_save_path
    )

    # Step 4) Plot and save the aggregated ROC curves (Stage 1 & Stage 2)
    roc_stage1_path = os.path.join(output_dir, "stage1_roc.png")
    roc_stage2_path = os.path.join(output_dir, "stage2_roc.png")
    plot_roc_curves(
        metadata, 
        oof_stage1_probs, 
        oof_stage2_probs,
        stage1_output_path=roc_stage1_path,
        stage2_output_path=roc_stage2_path
    )
