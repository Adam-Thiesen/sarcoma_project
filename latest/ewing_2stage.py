import pandas as pd
import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import scipy as sp
import os
from sklearn.metrics import classification_report


# Adjust some matplotlib defaults
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})


def plot_precision_recall_f1_curves(
    y_true,
    y_prob,
    title="Precision/Recall/F1",
    output_path="precision_recall_f1.pdf",
    show_plot=False
):
    """
    Plots precision, recall, and F1 scores as functions of the decision threshold.
    """
    # Ensure arrays are numpy
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Exclude NaNs if necessary
    valid_mask = ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    if len(np.unique(y_true)) < 2:
        print(f"[{title}] Not enough positive/negative samples to compute precision/recall/F1.")
        return

    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    plt.figure(figsize=(7, 6))
    plt.plot(thresholds, precisions, label='Precision', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='red')
    plt.plot(thresholds, f1s, label='F1-score', color='green')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"[{title}] Plot saved to: {output_path}")
    if show_plot:
        plt.show()
    plt.close()


def process_h5ad_with_labels(file_path):
    """
    Reads a single H5AD file and extracts metadata/features,
    adding RMS vs. NRSTS (RMS_Label) and Ewing vs. Other (Ewing_Label) columns.
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
            categories = [x.decode('utf-8') for x in categories]
            metadata_df['Patient ID'] = [categories[code] for code in codes]

        # Process 'Histological Subtype' and 'Histological Type'
        if 'Histological Subtype' in f['obs']:
            categories = f['obs/Histological Subtype/categories'][:]
            codes = f['obs/Histological Subtype/codes'][:]
            categories = [x.decode('utf-8') for x in categories]
            metadata_df['Histological Subtype'] = [categories[code] for code in codes]

        if 'Histological Type' in f['obs']:
            categories = f['obs/Histological Type/categories'][:]
            codes = f['obs/Histological Type/codes'][:]
            categories = [x.decode('utf-8') for x in categories]
            metadata_df['Histological Type'] = [categories[code] for code in codes]

        # Add RMS vs. NRSTS labels (1=RMS, 0=NRSTS)
        metadata_df['RMS_Label'] = metadata_df['Histological Type'].apply(
            lambda x: 1 if x == 'RMS' else (0 if x == 'NRSTS' else np.nan)
        )

        # Add Ewing vs. Other (among NRSTS, 1=Ewing, 0=Other)
        metadata_df['Ewing_Label'] = metadata_df.apply(
            lambda row: 1 if (row['Histological Subtype'] == 'Ewing Sarcoma'
                              and row['RMS_Label'] == 0) else 0,
            axis=1
        )

        # Feature data
        feature_data = f['X'][:]
        var_names = f['var/_index'][:].astype(str)
        feature_df = pd.DataFrame(feature_data, columns=var_names)

        valid_indices = metadata_df.dropna(subset=['RMS_Label']).index
        metadata_df = metadata_df.loc[valid_indices].copy()
        metadata_df['RMS_Label'] = metadata_df['RMS_Label'].astype(int)
        feature_df = feature_df.loc[valid_indices].copy()

        metadata_df.index = range(len(metadata_df))
        feature_df.index = range(len(feature_df))

    return metadata_df, feature_df


def process_h5ad_with_labels_two_files(file_path1, file_path2):
    print(f"Reading and processing File 1: {file_path1}")
    metadata1, features1 = process_h5ad_with_labels(file_path1)

    print(f"Reading and processing File 2: {file_path2}")
    metadata2, features2 = process_h5ad_with_labels(file_path2)

    common_idx = metadata1.index.intersection(metadata2.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping samples.")

    metadata1 = metadata1.loc[common_idx].copy()
    metadata2 = metadata2.loc[common_idx].copy()
    features1 = features1.loc[common_idx].copy()
    features2 = features2.loc[common_idx].copy()

    combined_metadata = metadata1.copy()
    combined_features = pd.concat([features1, features2], axis=1)

    return combined_metadata, combined_features


def cross_validated_two_stage_workflow(metadata, features, splits=5):
    """
    Stage 1 => RMS (1) vs. NRSTS (0).
    Stage 2 => Ewing (1) vs. Other (0), among predicted NRSTS.
    
    Returns:
    --------
    oof_stage1_preds, oof_stage1_probs, oof_stage2_preds, oof_stage2_probs,
    stage1_aucs, stage2_aucs
    """
    data = features.to_numpy()
    stage1_labels = metadata['RMS_Label'].to_numpy()      # RMS=1, NRSTS=0
    stage2_labels = metadata['Ewing_Label'].to_numpy()    # Ewing=1, Other=0

    groups = metadata['Patient ID'].to_numpy() if 'Patient ID' in metadata.columns else None

    n_samples = len(metadata)
    oof_stage1_preds = np.full(n_samples, np.nan)
    oof_stage1_probs = np.full(n_samples, np.nan)
    oof_stage2_preds = np.full(n_samples, np.nan)
    oof_stage2_probs = np.full(n_samples, np.nan)

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
        y_train_s1, y_test_s1 = stage1_labels[train_idx], stage1_labels[test_idx]

        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        # === Stage 1: RMS=1 vs NRSTS=0 ===
        _, pvals = sp.stats.ttest_ind(
            X_train[y_train_s1 == 0],  # NRSTS
            X_train[y_train_s1 == 1],  # RMS
            axis=0, equal_var=False
        )
        s1_feats = np.where(pvals < 0.05)[0]
        print(f"Stage 1: # features p<0.05 = {len(s1_feats)}")

        if len(s1_feats) == 0:
            print("No features pass p<0.05; using all features.")
            s1_feats = np.arange(X_train.shape[1])

        model_s1 = LR(
            penalty='l1',
            C=100,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000
        )
        model_s1.fit(X_train[:, s1_feats], y_train_s1)

        s1_probs_test = model_s1.predict_proba(X_test[:, s1_feats])[:, 1]
        s1_preds_test = (s1_probs_test >= 0.75).astype(int)

        try:
            auc_s1 = roc_auc_score(y_test_s1, s1_probs_test)
            stage1_aucs.append(auc_s1)
            print(f"Stage 1 AUC (Fold {fold_num}): {auc_s1:.3f}")
        except:
            print(f"Could not compute Stage 1 AUC (Fold {fold_num}).")

        oof_stage1_preds[test_idx] = s1_preds_test
        oof_stage1_probs[test_idx] = s1_probs_test

        # === Stage 2: Ewing=1 vs Other=0 (only among NRSTS) ===
        mask_train_nrsts = (y_train_s1 == 0)
        train_nrsts_idx = train_idx[mask_train_nrsts]
        X_train_nrsts = data[train_nrsts_idx]
        y_train_s2 = stage2_labels[train_nrsts_idx]

        _, pvals2 = sp.stats.ttest_ind(
            X_train_nrsts[y_train_s2 == 0],
            X_train_nrsts[y_train_s2 == 1],
            axis=0, equal_var=False
        )
        s2_feats = np.where(pvals2 < 0.05)[0]
        print(f"Stage 2: # features p<0.05 = {len(s2_feats)}")

        if len(s2_feats) == 0:
            s2_feats = np.arange(X_train_nrsts.shape[1])

        model_s2 = LR(
            penalty='l1',
            C=100,
            class_weight='balanced',
            solver='liblinear',
            max_iter=2000
        )
        model_s2.fit(X_train_nrsts[:, s2_feats], y_train_s2)

        # Only test on samples predicted as NRSTS => s1_preds_test==0
        test_pred_nrsts = test_idx[s1_preds_test == 0]
        print(f"Stage 2 test subset (pred NRSTS) = {len(test_pred_nrsts)}")

        if len(test_pred_nrsts) > 0:
            X_test_nrsts = data[test_pred_nrsts][:, s2_feats]
            s2_probs_test = model_s2.predict_proba(X_test_nrsts)[:, 1]
            s2_preds_test = (s2_probs_test >= 0.5).astype(int)

            oof_stage2_preds[test_pred_nrsts] = s2_preds_test
            oof_stage2_probs[test_pred_nrsts] = s2_probs_test

            # Evaluate Stage 2 AUC among truly NRSTS in the test fold
            mask_test_nrsts = (y_test_s1 == 0)
            test_true_nrsts_idx = test_idx[mask_test_nrsts]
            final_eval_idx = np.intersect1d(test_true_nrsts_idx, test_pred_nrsts)

            print(f"Stage 2 evaluation subset (true+pred NRSTS) = {len(final_eval_idx)}")
            if len(final_eval_idx) > 1:
                y_true_ewings = stage2_labels[final_eval_idx]
                try:
                    final_probs = model_s2.predict_proba(data[final_eval_idx][:, s2_feats])[:, 1]
                    auc_s2 = roc_auc_score(y_true_ewings, final_probs)
                    stage2_aucs.append(auc_s2)
                    print(f"Stage 2 AUC (Fold {fold_num}): {auc_s2:.3f}")
                except:
                    print(f"Could not compute Stage 2 AUC (Fold {fold_num}).")
            else:
                print(f"Not enough Stage 2 samples to compute AUC (Fold {fold_num}).")
        else:
            print(f"No NRSTS predicted in test fold {fold_num}, skipping Stage 2 AUC.")

    print("\nFinished Cross-Validation.")
    print(f"Stage 1 AUC folds: {len(stage1_aucs)} | Stage 2 AUC folds: {len(stage2_aucs)}")
    if stage1_aucs:
        print(f"Stage 1 Average AUC: {np.mean(stage1_aucs):.3f}")
    if stage2_aucs:
        print(f"Stage 2 Average AUC: {np.mean(stage2_aucs):.3f}")

    # Return the extra info
    return (
        oof_stage1_preds,
        oof_stage1_probs,
        oof_stage2_preds,
        oof_stage2_probs,
        stage1_aucs,
        stage2_aucs
    )



def build_two_stage_confusion_matrix(
    metadata, stage1_preds, stage2_preds, 
    output_path="two_stage_confusion_matrix.pdf"
):
    """
    Stage 1 => RMS=1, NRSTS=0
      - If stage1_preds==1 => predicted RMS => label=1
      - Else => predicted NRSTS => check Stage 2:
         stage2_preds==1 => Ewing => label=2
         stage2_preds==0 => Other => label=3

    True labels:
      RMS_Label=1 => RMS => 1
      Ewing_Label=1 => Ewing => 2
      else => Other => 3

    Final matrix order: [1, 2, 3] => [RMS, Ewing, NRSTS].
    """
    combined_pred_labels = np.full(len(metadata), np.nan)

    # Stage 1: predicted RMS => label=1
    mask_rms = (stage1_preds == 1)
    combined_pred_labels[mask_rms] = 1

    # Stage 1: predicted NRSTS => Ewing=2 or Other=3
    mask_nrsts = (stage1_preds == 0)
    combined_pred_labels[mask_nrsts & (stage2_preds == 1)] = 2
    combined_pred_labels[mask_nrsts & (stage2_preds == 0)] = 3

    # Fill leftover NaN as Other=3 (shouldn't happen often)
    combined_pred_labels = np.where(np.isnan(combined_pred_labels), 3, combined_pred_labels)

    # True: if RMS_Label=1 => 1, elif Ewing_Label=1 => 2, else 3
    def get_true_label(row):
        if row['RMS_Label'] == 1:
            return 1
        elif row['Ewing_Label'] == 1:
            return 2
        else:
            return 3

    true_labels = metadata.apply(get_true_label, axis=1).values

    cm = confusion_matrix(true_labels, combined_pred_labels, labels=[1, 2, 3])

    
    disp = ConfusionMatrixDisplay(cm, display_labels=['RMS', 'Ewing', 'NRSTS'])
    disp.plot(cmap="Blues")

    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

    print("Classification Report (CV OOF):")
    print(classification_report(
        true_labels,
        combined_pred_labels,
        labels=[1, 2, 3],
        target_names=['RMS', 'Ewing', 'NRSTS']
    ))
    return cm


def plot_roc_curves(
    metadata,
    stage1_probs, stage2_probs,
    stage1_auc_values=None, stage2_auc_values=None,  # new
    stage1_output_path="stage1_roc.pdf",
    stage2_output_path="stage2_roc.pdf"
):
    """
    Plots aggregated out-of-fold ROC curves:
      - Stage 1: RMS(1) vs. NRSTS(0)
      - Stage 2: Ewing(1) vs. Other(0), among truly NRSTS
    Also shows mean ± std from cross-validation in the legend if provided.
    """
    # True Stage 1
    y_true_s1 = metadata['RMS_Label'].values  # RMS=1, NRSTS=0
    # True Stage 2
    y_true_s2 = metadata['Ewing_Label'].values

    # ---- Stage 1 ROC ----
    valid_idx_s1 = ~np.isnan(stage1_probs)
    y_s1 = y_true_s1[valid_idx_s1]
    p_s1 = stage1_probs[valid_idx_s1]

    if len(np.unique(y_s1)) < 2:
        print("Cannot compute Stage 1 ROC: only one class present.")
    else:
        fpr_s1, tpr_s1, _ = roc_curve(y_s1, p_s1)
        auc_s1_agg = roc_auc_score(y_s1, p_s1)  # aggregated OOF AUC

        # If fold-by-fold values were given, compute mean ± std:
        if stage1_auc_values is not None and len(stage1_auc_values) > 0:
            mean_s1 = np.mean(stage1_auc_values)
            std_s1  = np.std(stage1_auc_values)
            label_s1 = f"AUC={auc_s1_agg:.3f}\n±{std_s1:.3f}"
        else:
            label_s1 = f"Stage 1 ROC (AUC={auc_s1_agg:.3f})"

        plt.figure(figsize=(6, 6))
        plt.plot(fpr_s1, tpr_s1, label=label_s1, color='blue')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Stage 1 ROC: RMS vs. NRSTS")
        plt.legend()

        plt.savefig(stage1_output_path, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        print(f"Stage 1 ROC curve saved to: {stage1_output_path}")

    # ---- Stage 2 ROC ----
    valid_idx_s2 = ~np.isnan(stage2_probs)
    truly_nrsts = (y_true_s1 == 0)
    final_idx_s2 = valid_idx_s2 & truly_nrsts

    if np.sum(final_idx_s2) < 2:
        print("Not enough samples to plot Stage 2 ROC.")
        return

    y_s2 = y_true_s2[final_idx_s2]
    p_s2 = stage2_probs[final_idx_s2]

    if len(np.unique(y_s2)) < 2:
        print("Cannot compute Stage 2 ROC: only one class present.")
        return

    fpr_s2, tpr_s2, _ = roc_curve(y_s2, p_s2)
    auc_s2_agg = roc_auc_score(y_s2, p_s2)

    if stage2_auc_values is not None and len(stage2_auc_values) > 0:
        mean_s2 = np.mean(stage2_auc_values)
        std_s2  = np.std(stage2_auc_values)
        label_s2 = f"AUC={auc_s2_agg:.3f}\n±{std_s2:.3f}"
    else:
        label_s2 = f"Stage 2 ROC (AUC={auc_s2_agg:.3f})"

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_s2, tpr_s2, label=label_s2, color='orange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stage 2 ROC: Ewing's vs. Other (NRSTS)")
    plt.legend()

    plt.savefig(stage2_output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Stage 2 ROC curve saved to: {stage2_output_path}")



if __name__ == "__main__":
    # Example paths
    file_path1 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.uni-1.h5ad"
    file_path2 = "/projects/rubinstein-lab/USERS/domans/sarcoma-features/ad_wsi.uni-4.h5ad"

    output_dir = "./output_figures"
    os.makedirs(output_dir, exist_ok=True)

       # 1) Read & combine data
    metadata, features = process_h5ad_with_labels_two_files(file_path1, file_path2)

    # 2) Cross-validation (notice we unpack 6 outputs now)
    (
        oof_s1_preds,
        oof_s1_probs,
        oof_s2_preds,
        oof_s2_probs,
        stage1_auc_values,
        stage2_auc_values
    ) = cross_validated_two_stage_workflow(metadata, features, splits=5)

    # 3) Confusion matrix
    cm_path = os.path.join(output_dir, "two_stage_confusion_matrix.pdf")
    build_two_stage_confusion_matrix(metadata, oof_s1_preds, oof_s2_preds, cm_path)

    # 4) Plot Stage 1 & 2 ROC with fold-by-fold stats
    roc_s1_path = os.path.join(output_dir, "stage1_roc.pdf")
    roc_s2_path = os.path.join(output_dir, "stage2_roc.pdf")
    plot_roc_curves(
        metadata,
        stage1_probs=oof_s1_probs,
        stage2_probs=oof_s2_probs,
        stage1_auc_values=stage1_auc_values,   # <--- pass the list of fold AUCs
        stage2_auc_values=stage2_auc_values,   # <--- pass the list of fold AUCs
        stage1_output_path=roc_s1_path,
        stage2_output_path=roc_s2_path
    )

    # 5) Precision/Recall/F1 curves
    # Stage 1
    prf1_s1_path = os.path.join(output_dir, "stage1_prf1.pdf")
    plot_precision_recall_f1_curves(
        y_true=metadata["RMS_Label"].values,
        y_prob=oof_s1_probs,
        title="Stage 1 (RMS vs. NRSTS): Precision/Recall/F1",
        output_path=prf1_s1_path
    )

    # Stage 2
    prf1_s2_path = os.path.join(output_dir, "stage2_prf1.pdf")
    plot_precision_recall_f1_curves(
        y_true=metadata["Ewing_Label"].values,
        y_prob=oof_s2_probs,
        title="Stage 2 (Ewing vs. Other): Precision/Recall/F1",
        output_path=prf1_s2_path
    )
