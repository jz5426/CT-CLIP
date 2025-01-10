"""
main evaluation training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import sys

from data import CTReportXRayClassificationDataset
from eval_utils import XrayClassificationModel

# Example usage
if __name__ == "__main__":
    from torchvision.models import resnet18

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train Vision Model Wrapper")
    parser.add_argument("--use_binary_classification", action="store_true", help="Toggle for binary classification")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Example dataset
    class ExampleDataset(Dataset):
        def __init__(self, num_samples, num_features):
            self.data = torch.randn(num_samples, 3, 224, 224)  # Example input size for vision models
            self.labels_binary = torch.randint(0, 2, (num_samples,))  # Binary labels
            self.labels_multilabel = torch.randint(0, 2, (num_samples, 5))  # Multi-label classification

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels_binary[idx], self.labels_multilabel[idx]

    # Load a pre-trained vision model
    pretrained_model = resnet18(pretrained=True)

    # Initialize the wrapper model
    num_classes = 1 if args.use_binary_classification else 5  # Change based on binary or multi-label classification
    model = XrayClassificationModel(vision_model=pretrained_model, isLinearProbe=True, in_features=512, num_classes=num_classes)

    # Set up the dataset and data loaders
    train_dataset = CTReportXRayClassificationDataset(
        data_folder='',
        cfg='',
        report_file='',
        labels=''
    )
    val_dataset = CTReportXRayClassificationDataset(
        data_folder='',
        cfg='',
        report_file='',
        labels=''
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Training loop configuration
    criterion = nn.BCEWithLogitsLoss() if not args.use_binary_classification else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Early stopping setup
    patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training and validation loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels_binary, labels_multilabel in train_loader:
            inputs = inputs.to(device)
            labels = labels_binary.to(device) if args.use_binary_classification else labels_multilabel.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels.float() if not args.use_binary_classification else labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {total_loss/len(train_loader):.4f}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels_binary, labels_multilabel in val_loader:
                inputs = inputs.to(device)
                labels = labels_binary.to(device) if args.use_binary_classification else labels_multilabel.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels.float() if not args.use_binary_classification else labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")
