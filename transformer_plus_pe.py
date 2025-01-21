import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
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
                    features = df.iloc[:, self.start_col:self.start_col + self.num_features].apply(pd.to_numeric, errors='coerce').fillna(0).values
                    self.data.append(features)
                    self.tile_indices.append(df.index.to_list())
                    label = int(label_row['labels'].values[0])
                    self.labels.append(label)
                    self.image_ids.append(base_name)

        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long), self.image_ids[idx], self.tile_indices[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # Create positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0).to(x.device)

class SimpleTransformerWithFeatureAttention(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=4, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super(SimpleTransformerWithFeatureAttention, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = None  # Initialize as None; will be set dynamically
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x):
        batch_size, num_tiles, _ = x.size()
        x = self.embedding(x)

        # Dynamically create PositionalEncoding if needed
        if self.positional_encoding is None or x.size(1) > (self.positional_encoding.pe.size(0) if self.positional_encoding else 0):
            self.positional_encoding = PositionalEncoding(self.embedding.out_features, max_len=x.size(1))

        x = self.positional_encoding(x)  # Add positional encoding
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Switch batch and sequence dimensions for transformer
        x = self.transformer(x)  # Self-attention within the tile

        self.attention_weights = x  # Store attention weights before aggregation

        x = x.mean(dim=0)  # Aggregate tile embeddings
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1), self.attention_weights.transpose(0, 1)  # Return attention weights per tile



def train(model, dataloader, criterion, optimizer, class_counts, device, epochs=100):
    model.train()
    epoch_losses = []  # List to store loss at each epoch

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)  # Ensure labels have shape [batch_size]

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

def evaluate(model, dataloader, output_file="metrics_output_kfold2_pe.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_preds = []
    image_labels = []
    correct_images = []
    incorrect_images = []

    with torch.no_grad():
        for inputs, labels, image_ids, tile_indices in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            labels = labels.view(-1)  # Ensure labels have shape [batch_size]
            outputs, attn_weights = model(inputs)
            probs = torch.sigmoid(outputs)  # Apply sigmoid for probabilities

            image_preds.append(probs.cpu().numpy())
            image_labels.append(labels.cpu().numpy())

            # Determine if the image was correctly classified
            pred_label = (probs > 0.5).float().item()
            true_label = labels.item()
            if pred_label == true_label:
                correct_images.append(image_ids[0])
            else:
                incorrect_images.append(image_ids[0])

    # Convert lists to numpy arrays
    image_preds = np.concatenate(image_preds)
    image_labels = np.concatenate(image_labels)

    # Calculate accuracy based on predictions
    final_preds = (image_preds > 0.5).astype(int)
    accuracy = accuracy_score(image_labels, final_preds)

    # Calculate AUROC
    auroc = roc_auc_score(image_labels, image_preds)

    # Calculate precision and recall
    report = classification_report(image_labels, final_preds, target_names=['ALVEOLAR', 'EMBRYONAL'], output_dict=True)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']

    # Save metrics to file
    with open(output_file, "w") as f:
        f.write(f"Accuracy: {accuracy * 100}%\n")
        f.write(f"AUC: {auroc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Correct images: {correct_images}")
        f.write(f"Incorrect images: {incorrect_images}")
        f.write("Learning rate: 0.0001")
        f.write("Random state: 42")

    return accuracy, auroc, precision, recall

def main(random_seed):
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Define the device to use for PyTorch (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the labels
    labels_df = pd.read_csv('/flashscratch/thiesa/Pytorch3/labels.csv')

    # Map string labels to integers
    label_mapping = {'ALVEOLAR': 0, 'EMBRYONAL': 1}
    labels_df['labels'] = labels_df['labels'].map(label_mapping)
    print(labels_df.head())

    # Initialize the dataset
    data_dir = '/flashscratch/thiesa/20x_features'
    dataset = RhabdomyosarcomaDataset(data_dir, labels_df)

    # Calculate class counts and pos_weight
    class_counts = np.bincount(dataset.labels)
    majority_class_count = max(class_counts)
    minority_class_count = min(class_counts)

    # Calculate pos_weight as the ratio of majority to minority
    pos_weight_value = majority_class_count / minority_class_count
    print(f"Positional Weight Value: {pos_weight_value}")
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # Define the loss function with pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Create a unique list of image IDs and their corresponding labels
    image_ids = np.array(dataset.image_ids)
    unique_image_ids = np.unique(image_ids)
    unique_labels = [dataset.labels[np.where(image_ids == img_id)[0][0]] for img_id in unique_image_ids]

    # Extract patient_ids corresponding to unique_image_ids
    patient_ids = labels_df.set_index('slide_id').loc[unique_image_ids, 'patient_id'].values

    # StratifiedGroupKFold setup
    stratified_group_kfold = StratifiedGroupKFold(n_splits=5, random_state=random_seed, shuffle=True)

    fold_results = []
    all_fold_losses = []

    # Perform stratified group splitting
    for fold, (train_index, test_index) in enumerate(stratified_group_kfold.split(unique_image_ids, unique_labels, groups=patient_ids)):
        print(f"Training fold {fold + 1}")

        # Create train and test datasets for the current fold
        train_image_ids = unique_image_ids[train_index]
        test_image_ids = unique_image_ids[test_index]

        train_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in train_image_ids]
        test_indices = [i for i, img_id in enumerate(dataset.image_ids) if img_id in test_image_ids]

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Initialize model, loss function, and optimizer
        model = SimpleTransformerWithFeatureAttention(input_dim=768)
        model.to(device)  # Move model to the appropriate device

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        # Train the model
        fold_losses = train(model, train_loader, criterion, optimizer, class_counts, device, epochs=75)
        all_fold_losses.append(fold_losses)

        # Evaluate the model
        accuracy, auroc, precision, recall = evaluate(model, test_loader)
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, AUC: {auroc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        fold_results.append((accuracy, auroc, precision, recall))

    # Plot training loss for each fold
    plt.figure(figsize=(10, 6))
    for fold, losses in enumerate(all_fold_losses):
        plt.plot(losses, label=f"Fold {fold + 1}")
    plt.title("Training Loss per Epoch for Each Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"training_loss_fold_{fold + 1}.png")
    plt.show()

    # Calculate and print average results across folds
    avg_accuracy = np.mean([result[0] for result in fold_results])
    avg_auroc = np.mean([result[1] for result in fold_results])
    avg_precision = np.mean([result[2] for result in fold_results])
    avg_recall = np.mean([result[3] for result in fold_results])

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average AUC: {avg_auroc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    # Save metrics to file
    with open("cross_validation_metrics2_pe.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"Average AUC: {avg_auroc:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write("Learning rate = 0.00001")
        f.write("Features used = combined_features")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate model with different random seeds using 5-fold cross-validation.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for cross-validation")
    args = parser.parse_args()

    main(args.seed)
