"""
Brain CT Scan - Hemorrhage vs Normal Classifier
================================================
Model     : EfficientNet-B3 with fine-tuning (best feature learner for medical imaging)
GradCAM   : Full GradCAM + GradCAM++ implementation
Dataset   : drive/MyDrive/processed_dataset/{train,val,test}/{hemorrhage,normal}
Environment: Google Colab (GPU)

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')
    !python brain_hemorrhage_model.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────
class Config:
    # Paths
    DATA_ROOT = "/content/processed_dataset"
    TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
    VAL_DIR     = os.path.join(DATA_ROOT, "val")
    TEST_DIR    = os.path.join(DATA_ROOT, "test")
    SAVE_DIR    = "/content/drive/MyDrive/hemorrhage_model_outputs"

    # Model
    MODEL_NAME  = "efficientnet_b3"   # Strong feature extractor, compact
    NUM_CLASSES = 2                    # hemorrhage, normal
    IMG_SIZE    = 224                  # EfficientNet default

    # Training
    BATCH_SIZE  = 32
    EPOCHS      = 40
    LR          = 1e-4                 # Low LR for fine-tuning
    WEIGHT_DECAY= 1e-4
    UNFREEZE_EPOCH = 10               # Unfreeze backbone at this epoch (two-phase training)

    # GradCAM
    GRADCAM_SAMPLES = 8               # How many test images to visualize

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = Config()
os.makedirs(cfg.SAVE_DIR, exist_ok=True)
print(f"✅ Device: {cfg.DEVICE}")
print(f"✅ Save dir: {cfg.SAVE_DIR}")


# ── Data Transforms ────────────────────────────────────────────────────────────
# Medical imaging augmentations — careful, no color jitter (CT is grayscale converted to RGB)
train_transforms = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # subtle — CT contrast variance
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],            # ImageNet stats (pretrained)
                         [0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Datasets & Loaders ─────────────────────────────────────────────────────────
def build_loaders():
    train_ds = datasets.ImageFolder(cfg.TRAIN_DIR, transform=train_transforms)
    val_ds   = datasets.ImageFolder(cfg.VAL_DIR,   transform=val_test_transforms)
    test_ds  = datasets.ImageFolder(cfg.TEST_DIR,  transform=val_test_transforms)

    # Class weights to handle imbalance (hemorrhage cases often fewer)
    class_counts = np.array([len(os.listdir(os.path.join(cfg.TRAIN_DIR, c)))
                              for c in train_ds.classes])
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[label] for _, label in train_ds.samples])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n📁 Classes : {train_ds.classes}  →  {train_ds.class_to_idx}")
    print(f"   Train   : {len(train_ds)} images")
    print(f"   Val     : {len(val_ds)} images")
    print(f"   Test    : {len(test_ds)} images")
    print(f"   Class counts (train): {dict(zip(train_ds.classes, class_counts))}")

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# ── Model ──────────────────────────────────────────────────────────────────────
class HemorrhageDetector(nn.Module):
    """
    EfficientNet-B3 backbone with custom classification head.
    Two-phase training: head first → then fine-tune backbone.
    """
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        # Load pretrained backbone
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

        # Backbone: all layers except classifier
        self.features = base.features        # Conv feature extractor
        self.avgpool  = base.avgpool         # Adaptive avg pool

        in_features = base.classifier[1].in_features  # 1536 for B3

        # Custom head — deeper for better feature separation
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        # Start with frozen backbone
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen — training head only")

    def unfreeze_backbone(self, unfreeze_last_n_blocks=3):
        """Unfreeze only the last N blocks for fine-tuning (avoid catastrophic forgetting)"""
        blocks = list(self.features.children())
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"🔓 Unfroze last {unfreeze_last_n_blocks} backbone blocks")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ── GradCAM ────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    GradCAM: Gradient-weighted Class Activation Mapping
    Hooks into the last conv block of EfficientNet to visualize
    which regions of the CT scan activated the prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0).to(cfg.DEVICE)
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot)

        # Pool gradients across channels
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx, torch.softmax(output, dim=1)[0].detach().cpu().numpy()


def overlay_gradcam(image_tensor, cam, alpha=0.5):
    """Overlay CAM heatmap on original image"""
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).numpy()
    img = (img * std + mean).clip(0, 1)

    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray(cam).resize(
        (img.shape[1], img.shape[0]), Image.BILINEAR))

    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = alpha * heatmap + (1 - alpha) * img
    return img, heatmap, overlay.clip(0, 1)


# ── Training ───────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}", end="\r")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"],   label="Val Loss",   linewidth=2)
    ax1.axvline(cfg.UNFREEZE_EPOCH, color="red", linestyle="--", label="Backbone unfrozen")
    ax1.set_title("Loss Curve", fontsize=14)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"],   label="Val Acc",   linewidth=2)
    ax2.axvline(cfg.UNFREEZE_EPOCH, color="red", linestyle="--", label="Backbone unfrozen")
    ax2.set_title("Accuracy Curve", fontsize=14)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📊 Training curves saved → {save_path}")


# ── GradCAM Visualization ──────────────────────────────────────────────────────
def visualize_gradcam(model, test_loader, class_to_idx, save_dir, n_samples=8):
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Hook into the LAST conv block of EfficientNet-B3 (features[-1])
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    # Collect samples (mix of both classes)
    samples, sample_labels = [], []
    for imgs, labels in test_loader:
        for img, label in zip(imgs, labels):
            samples.append(img)
            sample_labels.append(label.item())
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, n_samples * 3.5))
    fig.suptitle("GradCAM — Brain CT Hemorrhage Detection\n"
                 "(Left: Original | Center: Heatmap | Right: Overlay)",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, (img_tensor, true_label) in enumerate(zip(samples, sample_labels)):
        cam, pred_idx, probs = gradcam.generate(img_tensor)
        orig, heatmap, overlay = overlay_gradcam(img_tensor, cam)

        true_name = idx_to_class[true_label]
        pred_name = idx_to_class[pred_idx]
        conf      = probs[pred_idx] * 100
        correct   = "✅" if true_label == pred_idx else "❌"

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"True: {true_name}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("GradCAM Heatmap", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{correct} Pred: {pred_name} ({conf:.1f}%)", fontsize=9)
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "gradcam_visualization.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"🧠 GradCAM visualization saved → {save_path}")


def visualize_gradcam_grid(model, test_loader, class_to_idx, save_dir, n_per_class=4):
    """
    Separate GradCAM grids per class — shows what features the model
    learned for hemorrhage vs normal independently.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    class_samples = {0: [], 1: []}
    for imgs, labels in test_loader:
        for img, label in zip(imgs, labels):
            lbl = label.item()
            if len(class_samples[lbl]) < n_per_class:
                class_samples[lbl].append(img)
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    for class_idx, class_name in idx_to_class.items():
        imgs_list = class_samples[class_idx]
        fig, axes = plt.subplots(n_per_class, 3, figsize=(10, n_per_class * 3))
        fig.suptitle(f"GradCAM — Class: {class_name.upper()}", fontsize=14, fontweight="bold")

        for i, img_tensor in enumerate(imgs_list):
            cam, pred_idx, probs = gradcam.generate(img_tensor, class_idx=class_idx)
            orig, heatmap, overlay = overlay_gradcam(img_tensor, cam)

            axes[i, 0].imshow(orig);    axes[i, 0].axis("off")
            axes[i, 1].imshow(heatmap); axes[i, 1].axis("off")
            axes[i, 2].imshow(overlay); axes[i, 2].axis("off")

            if i == 0:
                axes[i, 0].set_title("Original",   fontsize=10)
                axes[i, 1].set_title("GradCAM",    fontsize=10)
                axes[i, 2].set_title("Overlay",    fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"gradcam_{class_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"🧠 GradCAM [{class_name}] saved → {save_path}")


# ── Confusion Matrix & Metrics ─────────────────────────────────────────────────
def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=1, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n📋 Classification Report:\n{report}")
    print(f"📊 Confusion matrix saved → {save_path}")
    return report


# ── ROC / AUC ──────────────────────────────────────────────────────────────────
def plot_roc_curve(all_labels, all_probs, save_path):
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Hemorrhage Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"📈 ROC AUC: {roc_auc:.4f}  |  Saved → {save_path}")
    return roc_auc


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  Brain CT Hemorrhage Detection — EfficientNet-B3 + GradCAM")
    print("="*60)

    # ── Data
    train_loader, val_loader, test_loader, class_to_idx = build_loaders()
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # ── Model
    model = HemorrhageDetector(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: {cfg.MODEL_NAME}")
    print(f"   Total params     : {total_params:,}")
    print(f"   Trainable params : {trainable_params:,}  (head only at start)")

    # ── Loss: Label smoothing helps generalization, avoids overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    # ── Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_path = os.path.join(cfg.SAVE_DIR, "best_model.pth")

    print(f"\n🚀 Starting training for {cfg.EPOCHS} epochs...\n")

    for epoch in range(1, cfg.EPOCHS + 1):

        # ── Phase 2: unfreeze backbone for fine-tuning
        if epoch == cfg.UNFREEZE_EPOCH:
            model.unfreeze_backbone(unfreeze_last_n_blocks=3)
            # Re-init optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": cfg.LR * 0.1},
                {"params": model.classifier.parameters(), "lr": cfg.LR},
            ], weight_decay=cfg.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - epoch, eta_min=1e-6)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        phase = "fine-tune" if epoch >= cfg.UNFREEZE_EPOCH else "head-only"
        print(f"\nEpoch {epoch:02d}/{cfg.EPOCHS} [{phase}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # Save best model (by val loss — better proxy for feature learning than acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("__")
                           and not callable(v) and not isinstance(v, torch.device)},
            }, best_model_path)
            print(f"  💾 Best model saved (val_loss={val_loss:.4f})")

    print("\n✅ Training complete!")

    # ── Plot training curves
    plot_training_curves(history, os.path.join(cfg.SAVE_DIR, "training_curves.png"))

    # ── Save training history
    with open(os.path.join(cfg.SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Load best model for evaluation
    print("\n📥 Loading best model for evaluation...")
    ckpt = torch.load(best_model_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"   Best epoch: {ckpt['epoch']}  |  Val Loss: {ckpt['val_loss']:.4f}")

    # ── Test set evaluation
    print("\n🔍 Evaluating on test set...")
    test_loss, test_acc, all_preds, all_labels, all_probs = evaluate(model, test_loader, criterion)
    print(f"   Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f}")

    # ── Confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          os.path.join(cfg.SAVE_DIR, "confusion_matrix.png"))

    # ── ROC-AUC
    roc_auc = plot_roc_curve(all_labels, all_probs,
                             os.path.join(cfg.SAVE_DIR, "roc_curve.png"))

    # ── GradCAM visualizations
    print("\n🧠 Generating GradCAM visualizations...")
    visualize_gradcam(model, test_loader, class_to_idx, cfg.SAVE_DIR,
                      n_samples=cfg.GRADCAM_SAMPLES)
    visualize_gradcam_grid(model, test_loader, class_to_idx, cfg.SAVE_DIR,
                           n_per_class=4)

    # ── Final summary
    print("\n" + "="*60)
    print("  📦 ALL OUTPUTS SAVED TO:", cfg.SAVE_DIR)
    print("="*60)
    print(f"  best_model.pth         → weights (reload anytime)")
    print(f"  training_curves.png    → loss & accuracy plots")
    print(f"  confusion_matrix.png   → per-class performance")
    print(f"  roc_curve.png          → ROC AUC: {roc_auc:.4f}")
    print(f"  gradcam_visualization.png  → mixed GradCAM overlays")
    print(f"  gradcam_hemorrhage.png     → GradCAM on hemorrhage class")
    print(f"  gradcam_normal.png         → GradCAM on normal class")
    print(f"  history.json           → raw training metrics")
    print("="*60)


if __name__ == "__main__":
    main()
