import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             roc_curve, confusion_matrix, precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})

################################################################################
# Dataset Definition
################################################################################
class RhabdomyosarcomaDataset(Dataset):
    def __init__(self, data_dir, labels_df, start_col=8, num_features=768):
        self.data = []
        self.labels = []
        self.image_ids = []
        self.tile_indices = []
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.start_col = start_col
        self.num_features = num_features

        self.labels_df['slide_id'] = self.labels_df['slide_id'].astype(str).str.strip()

        for file_name in os.listdir(data_dir):
            if file_name.endswith('.gz'):
                base_name = file_name.replace('.gz', '').strip()
                label_row = self.labels_df[self.labels_df['slide_id'] == base_name]
                if label_row.empty:
                    print(f"Warning: No label found for file {file_name}, skipping...")
                    continue

                file_path = os.path.join(data_dir, file_name)
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
                    features = df.iloc[:, self.start_col:self.start_col + self.num_features] \
                                   .apply(pd.to_numeric, errors='coerce') \
                                   .fillna(0).values
                    self.data.append(features)
                    self.tile_indices.append(df.index.to_list())
                    label = int(label_row['labels'].values[0])
                    self.labels.append(label)
                    self.image_ids.append(base_name)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.image_ids[idx],
            self.tile_indices[idx]
        )


################################################################################
# Transformer Model Components
################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)

class SimpleTransformerWithClassToken(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super(SimpleTransformerWithClassToken, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_encoding = None

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tiles, _ = x.size()

        x = self.embedding(x)
        # Create positional encoding (or re-create if sequence is longer than a previous fold)
        if (self.positional_encoding is None) or (num_tiles + 1 > self.positional_encoding.pe.size(0)):
            self.positional_encoding = PositionalEncoding(
                embed_dim=self.embedding.out_features,
                max_len=num_tiles + 1
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)

        cls_output = x[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits.squeeze(1), None

################################################################################
# Training and Evaluation
################################################################################
def train(model, dataloader, criterion, optimizer, class_counts, device, epochs=100):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}')

    return epoch_losses

def evaluate(model, dataloader, output_file=None):
    """
    Returns standard metrics, plus FPR/TPR for ROC, plus raw predictions and labels
    so we can build a pooled confusion matrix across folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs, _ = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Accuracy (using threshold=0.5)
    final_preds = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, final_preds)

    # ROC / AUC
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    auroc = roc_auc_score(all_targets, all_probs)

    # Classification Report (Precision, Recall, F1) using threshold=0.5
    report = classification_report(
        all_targets, 
        final_preds, 
        target_names=['EMBRYONAL', 'ALVEOLAR'], 
        output_dict=True
    )
    precision = report['weighted avg']['precision']
    recall    = report['weighted avg']['recall']
    f1        = report['weighted avg']['f1-score']

    # Optionally save to file
    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"AUC: {auroc:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"F1: {f1:.3f}\n")

    return accuracy, auroc, precision, recall, f1, fpr, tpr, final_preds, all_targets, all_probs


################################################################################
# Main Script / Cross-Validation
################################################################################
def main(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels
    labels_df = pd.read_csv('/flashscratch/thiesa/Pytorch3/labels.csv')
    label_mapping = {'EMBRYONAL': 0, 'ALVEOLAR': 1}
    labels_df['labels'] = labels_df['labels'].map(label_mapping)
    print(labels_df.head())

    # Dataset
    data_dir = '/flashscratch/thiesa/ctransapth_20x_features'
    dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

    # Class counts and pos_weight
    class_counts = np.bincount(dataset.labels)
    majority_class_count = max(class_counts)
    minority_class_count = min(class_counts)
    pos_weight_value = majority_class_count / minority_class_count
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(f"Positional Weight Value: {pos_weight_value:.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Setup for StratifiedGroupKFold
    image_ids = np.array(dataset.image_ids)
    unique_image_ids = np.unique(image_ids)
    unique_labels = [dataset.labels[np.where(image_ids == img_id)[0][0]] for img_id in unique_image_ids]
    patient_ids = labels_df.set_index('slide_id').loc[unique_image_ids, 'patient_id'].values

    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, random_state=random_seed, shuffle=True)

    fold_results = []
    all_fold_losses = []

    # For ROC averaging
    fold_fprs = []
    fold_tprs = []
    fold_aucs = []

    # We'll store all predictions & labels from all folds for 
    # threshold-based metrics and confusion matrix:
    all_folds_probs = []
    all_folds_labels = []

    for fold, (train_index, test_index) in enumerate(stratified_group_kfold.split(unique_image_ids, unique_labels, groups=patient_ids)):
        print(f"Training fold {fold + 1}")

        train_image_ids = unique_image_ids[train_index]
        test_image_ids  = unique_image_ids[test_index]

        train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
        test_indices  = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

        train_dataset = Subset(dataset, train_indices)
        test_dataset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = SimpleTransformerWithClassToken(input_dim=768)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # Train
        fold_losses = train(model, train_loader, criterion, optimizer, class_counts, device, epochs=75)
        all_fold_losses.append(fold_losses)

        # Evaluate
        accuracy, auroc, precision, recall, f1, fpr, tpr, preds_fold, labels_fold, probs_fold = evaluate(
            model, test_loader, output_file=None
        )
        print(f"Fold {fold + 1} | Accuracy: {accuracy:.3f}, AUC: {auroc:.3f}, "
              f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        fold_results.append((accuracy, auroc, precision, recall, f1))
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)
        fold_aucs.append(auroc)

        # Store predictions & labels globally
        all_folds_probs.extend(probs_fold)
        all_folds_labels.extend(labels_fold)

    ############################################################################
    # 1) Plot training loss per fold
    ############################################################################
    plt.figure(figsize=(10, 6))
    for fold_idx, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f"Fold {fold_idx + 1}")
    plt.title("Training Loss per Epoch (Each Fold)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_loss.pdf", dpi=300, transparent=True)
    plt.show()

    ############################################################################
    # 2) Compute & plot average ROC curve
    ############################################################################
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interpolated = []
    for i in range(len(fold_fprs)):
        tpr_interp = np.interp(mean_fpr, fold_fprs[i], fold_tprs[i])
        tprs_interpolated.append(tpr_interp)

    tprs_interpolated = np.array(tprs_interpolated)
    mean_tpr = tprs_interpolated.mean(axis=0)
    std_tpr = tprs_interpolated.std(axis=0)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2,
                     label="±1 std dev")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guess")
    plt.title("Average ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("average_roc_curve.pdf", dpi=300, transparent=True)
    plt.show()

    ############################################################################
    # 3) Precision, Recall, F1 vs. Threshold (pooled across all folds)
    ############################################################################
    all_folds_probs = np.array(all_folds_probs)
    all_folds_labels = np.array(all_folds_labels)

    thresholds = np.linspace(0, 1, 101)
    precision_list = []
    recall_list = []
    f1_list = []

    for thr in thresholds:
        preds_thr = (all_folds_probs >= thr).astype(int)
        # We can compute P/R/F1 via precision_recall_fscore_support:
        prec, rec, fscore, _ = precision_recall_fscore_support(
            all_folds_labels, preds_thr, average='weighted', zero_division=0
        )
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(fscore)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision_list, label='Precision')
    plt.plot(thresholds, recall_list,    label='Recall')
    plt.plot(thresholds, f1_list,        label='F1 Score')
    plt.title("Precision, Recall, and F1 vs. Threshold (All Folds Pooled)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)  # P/R/F are between 0 and 1
    plt.legend()
    plt.savefig("precision_recall_f1_by_threshold.pdf", dpi=300, transparent=True)
    plt.show()

    ############################################################################
    # 4) Confusion Matrix (percentages) using pooled predictions (threshold=0.5)
    ############################################################################
    preds_05 = (all_folds_probs >= 0.5).astype(int)
    conf_mat = confusion_matrix(all_folds_labels, preds_05)

    # Convert to percentage per row
    conf_mat_percent = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_mat_percent, 
        annot=True, 
        fmt=".2%", 
        cmap='Blues', 
        xticklabels=['EMBRYONAL', 'ALVEOLAR'], 
        yticklabels=['EMBRYONAL', 'ALVEOLAR']
    )
    plt.title("Confusion Matrix (Percent) @ Threshold=0.5")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix_percent.pdf", dpi=300, transparent=True)
    plt.show()

    ############################################################################
    # Print & save final average metrics across folds (using the 0.5 threshold)
    ############################################################################
    avg_accuracy  = np.mean([res[0] for res in fold_results])
    avg_auroc     = np.mean([res[1] for res in fold_results])
    avg_precision = np.mean([res[2] for res in fold_results])
    avg_recall    = np.mean([res[3] for res in fold_results])
    avg_f1        = np.mean([res[4] for res in fold_results])

    print(f"Average Accuracy: {avg_accuracy:.3f}")
    print(f"Average AUC: {avg_auroc:.3f}")
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")

    with open("cross_validation_metrics2_pe.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_accuracy:.3f}\n")
        f.write(f"Average AUC: {avg_auroc:.3f}\n")
        f.write(f"Average Precision: {avg_precision:.3f}\n")
        f.write(f"Average Recall: {avg_recall:.3f}\n")
        f.write(f"Average F1: {avg_f1:.3f}\n")
        f.write("Learning rate = 0.00001\n")
        f.write("Features used = combined_features\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model with 5-fold cross-validation.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for cross-validation")
    args = parser.parse_args()

    main(args.seed)
