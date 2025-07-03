import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    if epoch is not None:
        wandb.log({"train_loss": epoch_loss, "epoch": epoch})
    return epoch_loss

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)

def kfold(X, y, model_class, model_kwargs, epochs=10, batch_size=128, n_splits=7, device='cpu'):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    bal_acc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold+1}/{n_splits}")
        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        train_ds = TensorDataset(torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx]))
        val_ds = TensorDataset(torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2)
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)
        preds, labels = evaluate(model, val_loader, device)
        f1 = f1_score(labels, preds, average='weighted')
        bal_acc = balanced_accuracy_score(labels, preds)
        print(f"F1-score: {f1:.4f} | Balanced accuracy: {bal_acc:.4f}")
        wandb.log({f"fold{fold+1}_f1": f1, f"fold{fold+1}_bal_acc": bal_acc})
        f1_scores.append(f1)
        bal_acc_scores.append(bal_acc)
    print("\nResumen KFold:")
    print(f"F1-score promedio: {np.mean(f1_scores):.4f}")
    print(f"Balanced accuracy promedio: {np.mean(bal_acc_scores):.4f}")
    wandb.log({"kfold_f1_mean": np.mean(f1_scores), "kfold_bal_acc_mean": np.mean(bal_acc_scores)})
    return f1_scores, bal_acc_scores
