import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    classification_report,
    roc_curve,
    auc,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns  # For a nice heatmap
import argparse

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

class SimpleTransformerWithFeatureAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super(SimpleTransformerWithFeatureAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x):
        batch_size, num_tiles, _ = x.size()
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Switch batch & sequence for the transformer
        x = self.transformer(x)
        self.attention_weights = x  # (not raw attention, but post-transformer)
        x = x.mean(dim=0)  # aggregate tile embeddings
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1), self.attention_weights.transpose(0, 1)

def train(model, dataloader, criterion, optimizer, class_counts, device, epochs=100):
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)  # shape [batch_size]

            optimizer.zero_grad()
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

    return epoch_losses

def evaluate(model, dataloader, output_file="metrics_output_kfold2.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_preds = []  # Will hold raw probabilities
    image_labels = []
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)
            outputs, attn_weights = model(inputs)
            probs = torch.sigmoid(outputs)

            # Collect raw probabilities and labels
            image_preds.append(probs.cpu().numpy())
            image_labels.append(labels.cpu().numpy())

            # Binarize predictions at 0.5
            pred_label = (probs > 0.5).float().item()
            true_label = labels.item()
            if pred_label == true_label:
                correct_images.append(image_ids[0])
            else:
                incorrect_images.append(image_ids[0])

    # Convert lists of arrays to single numpy arrays
    image_preds = np.concatenate(image_preds)  # shape: (N,)
    image_labels = np.concatenate(image_labels)  # shape: (N,)

    # Threshold at 0.5 for standard metrics
    final_preds = (image_preds > 0.5).astype(int)

    # Accuracy
    accuracy = accuracy_score(image_labels, final_preds)

    # AUROC (using raw probabilities!)
    auroc = roc_auc_score(image_labels, image_preds)

    # Classification report
    report = classification_report(
        image_labels, 
        final_preds,
        target_names=['ALVEOLAR', 'EMBRYONAL'],
        output_dict=True
    )
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    # Write metrics to file
    with open(output_file, "w") as f:
        f.write(f"Accuracy: {accuracy * 100}%\n")
        f.write(f"AUC: {auroc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Correct images: {correct_images}\n")
        f.write(f"Incorrect images: {incorrect_images}\n")
        f.write("Learning rate: 0.00001\n")
        f.write("Random state: 42\n")

    # IMPORTANT: Now we return the *raw probabilities* (image_preds) 
    # as well as the *binarized predictions* (final_preds).
    return accuracy, auroc, precision, recall, image_preds, final_preds, image_labels


def main(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_df = pd.read_csv('/flashscratch/thiesa/Pytorch3/labels.csv')
    label_mapping = {'ALVEOLAR': 0, 'EMBRYONAL': 1}
    labels_df['labels'] = labels_df['labels'].map(label_mapping)
    print(labels_df.head())

    # Initialize dataset
    data_dir = '/flashscratch/thiesa/20x_features'
    dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

    # Class counts / pos_weight
    class_counts = np.bincount(dataset.labels)
    majority_class_count = max(class_counts)
    minority_class_count = min(class_counts)
    pos_weight_value = majority_class_count / minority_class_count
    print(f"Positional Weight Value: {pos_weight_value}")
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Unique image IDs, labels, patient IDs
    image_ids = np.array(dataset.image_ids)
    unique_image_ids = np.unique(image_ids)
    unique_labels = [dataset.labels[np.where(image_ids == img_id)[0][0]]
                     for img_id in unique_image_ids]
    patient_ids = labels_df.set_index('slide_id').loc[unique_image_ids, 'patient_id'].values

    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, random_state=random_seed, shuffle=True)

    fold_results = []
    all_fold_losses = []

    # For plotting the individual ROC curves:
    fold_fprs = []
    fold_tprs = []

    # Initialize a matrix to accumulate confusion matrices across folds
    accum_cm = np.zeros((2, 2), dtype=np.float32)

    for fold, (train_index, test_index) in enumerate(stratified_group_kfold.split(
        unique_image_ids, unique_labels, groups=patient_ids)):
        print(f"Training fold {fold + 1}")

        train_image_ids = unique_image_ids[train_index]
        test_image_ids = unique_image_ids[test_index]

        train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
        test_indices  = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

        train_dataset = Subset(dataset, train_indices)
        test_dataset  = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = SimpleTransformerWithFeatureAttention(input_dim=768)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # Train the model on this fold
        fold_losses = train(model, train_loader, criterion, optimizer, class_counts, device, epochs=75)
        all_fold_losses.append(fold_losses)

        # Evaluate on this fold
        accuracy, auroc, precision, recall, fold_probs, fold_binary_preds, fold_labels = evaluate(model, test_loader)

        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, AUC: {auroc:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")

        fold_results.append((accuracy, auroc, precision, recall))

        # Use the raw probabilities to compute the ROC curve
        fpr, tpr, _ = roc_curve(fold_labels, fold_probs)
        fold_fprs.append(fpr)
        fold_tprs.append(tpr)

        # Use the thresholded predictions to build the confusion matrix
        cm = confusion_matrix(fold_labels, fold_binary_preds)
        accum_cm += cm

   
    # Plot training loss for each fold

    plt.figure(figsize=(10, 6))
    for fold_idx, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f"Fold {fold_idx + 1}")
    plt.title("Training Loss per Epoch for Each Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # We'll just name it after the last fold for simplicity
    plt.savefig(f"training_loss_fold_{fold_idx + 1}.png")
    plt.show()

 
    # Compute & Print Average Performance
    
    avg_accuracy  = np.mean([res[0] for res in fold_results])
    avg_auroc     = np.mean([res[1] for res in fold_results])
    avg_precision = np.mean([res[2] for res in fold_results])
    avg_recall    = np.mean([res[3] for res in fold_results])

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average AUC: {avg_auroc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    with open("cross_validation_metrics2.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Average AUC: {avg_auroc:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write("Learning rate = 0.00001\n")
        f.write("Features used = combined_features\n")

    
    # Plot the 5 individual ROC curves on the same graph
    # and report mean ± std of fold AUC
    
    plt.figure(figsize=(7, 6))
    for i in range(len(fold_fprs)):
        fold_auc = fold_results[i][1]  # The second element is the AUC
        plt.plot(fold_fprs[i], fold_tprs[i],
                 lw=1.5,
                 label=f"Fold {i+1} (AUC = {fold_auc:.2f})")

    # Diagonal reference line
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Fold')
    plt.legend(loc='lower right')
    plt.savefig('fold_rocs.png')
    plt.show()

    # Print mean ± std of fold AUC
    fold_aucs = [res[1] for res in fold_results]
    mean_auroc = np.mean(fold_aucs)
    std_auroc = np.std(fold_aucs)
    print(f"Mean AUC across folds = {mean_auroc:.4f} ± {std_auroc:.4f}")

    
    # (Optional) Plot the "Average" ROC (mean TPR ± std TPR)
    # as you had originally
   
    base_fpr = np.linspace(0, 1, 101)
    tprs = []

    for i in range(len(fold_fprs)):
        tpr_interp = np.interp(base_fpr, fold_fprs[i], fold_tprs[i])
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(base_fpr, mean_tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(
        base_fpr, mean_tpr,
        color='b',
        label=r'Mean ROC (AUC = %0.2f)' % mean_auc,
        lw=2, alpha=.8
    )
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(base_fpr, tpr_lower, tpr_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Average ROC across folds')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('average_roc.png')
    plt.show()

    
    # Plot Average Confusion Matrix
    
    avg_cm = accum_cm / 5.0  # since n_splits=5
    plt.figure(figsize=(5, 4))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["ALVEOLAR (Pred)", "EMBRYONAL (Pred)"],
                yticklabels=["ALVEOLAR (True)", "EMBRYONAL (True)"])
    plt.title("Average Confusion Matrix (5-fold CV)")
    plt.savefig("average_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate model with different random seeds using 5-fold cross-validation."
    )
    parser.add_argument('--seed', type=int, required=True, help="Random seed for cross-validation")
    args = parser.parse_args()
    main(args.seed)
