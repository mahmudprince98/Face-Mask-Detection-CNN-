import os, yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from dataset import get_loaders
from model import create_model
from utils import set_seed, ensure_dir, compute_report, plot_confusion

def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, y_true, y_pred

def train():
    with open("src/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, classes = get_loaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"], cfg["data"]["test_dir"],
        cfg["data"]["img_size"], cfg["train"]["batch_size"], cfg["train"]["num_workers"]
    )

    model = create_model(num_classes=len(classes)).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    ensure_dir(cfg["train"]["save_dir"])
    best_path = os.path.join(cfg["train"]["save_dir"], cfg["train"]["best_model_name"])

    best_val_acc = 0.0
    patience, wait = cfg["train"]["early_stopping_patience"], 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['train']['epochs']}")
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        # validation
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch} -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, best_path)
            print(f"✔ Saved best model to {best_path}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    print("\nTraining complete. Best Val Acc:", best_val_acc)

    # Final evaluation on test (if available)
    if test_loader is not None:
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        compute_report(y_true, y_pred, classes)
        plot_confusion(y_true, y_pred, classes, save_path=os.path.join(cfg["train"]["save_dir"], "confusion_matrix.png"))

if __name__ == "__main__":
    train()
