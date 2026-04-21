import os, json, random, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR   = '/kaggle/input/chest-xray-pneumonia/chest_xray'
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 10
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

print(f"Using: {DEVICE}")


# ── Transforms ───────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   val_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Sanity check — class order must be NORMAL=0, PNEUMONIA=1
print("Class mapping:", train_ds.class_to_idx)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")


# ── Model — UNFREEZE ALL LAYERS this time ────────────────────────────
model = models.resnet18(weights='IMAGENET1K_V1')

# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 2)
)
model = model.to(DEVICE)
print("Model ready. Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# ── Class weights (PNEUMONIA ~3x more samples than NORMAL) ───────────
counts  = np.array([len(os.listdir(os.path.join(DATA_DIR, 'train', c)))
                    for c in train_ds.classes])
weights = torch.tensor(1.0 / counts, dtype=torch.float).to(DEVICE)
weights = weights / weights.sum()
print(f"Class weights: { {c: round(w.item(),4) for c,w in zip(CLASS_NAMES, weights)} }")

criterion = nn.CrossEntropyLoss(weight=weights)

# Two param groups: lower LR for pretrained backbone, higher for new head
optimizer = optim.Adam([
    {'params': list(model.parameters())[:-4], 'lr': 1e-5},   # backbone
    {'params': list(model.parameters())[-4:], 'lr': 1e-3},   # classifier
])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training loop ────────────────────────────────────────────────────
best_val_acc = 0.0
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(1, EPOCHS + 1):

    # — Train —
    model.train()
    tr_loss, tr_correct, tr_total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        tr_loss    += loss.item() * imgs.size(0)
        tr_correct += (out.argmax(1) == labels).sum().item()
        tr_total   += imgs.size(0)

    # — Validate —
    model.eval()
    vl_loss, vl_correct, vl_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            vl_loss    += loss.item() * imgs.size(0)
            vl_correct += (out.argmax(1) == labels).sum().item()
            vl_total   += imgs.size(0)

    scheduler.step()

    tr_acc = tr_correct / tr_total
    vl_acc = vl_correct / vl_total
    history['train_loss'].append(tr_loss / tr_total)
    history['val_loss'].append(vl_loss / vl_total)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

    saved = ''
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), '/kaggle/working/pneumonia_best.pth')
        saved = '  ✓ saved'

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss {tr_loss/tr_total:.4f}  Acc {tr_acc:.4f} | "
          f"Val Loss {vl_loss/vl_total:.4f}  Acc {vl_acc:.4f}{saved}")

print(f"\nBest Val Accuracy: {best_val_acc:.4f}")


# ── Save ─────────────────────────────────────────────────────────────
with open('/kaggle/working/class_names.json', 'w') as f:
    json.dump(CLASS_NAMES, f)

for fname in ['pneumonia_best.pth', 'class_names.json']:
    path = f'/kaggle/working/{fname}'
    print(f"  {fname}  →  {os.path.getsize(path)/1e6:.2f} MB")


    # ── Test Evaluation ───────────────────────────────────────────────────
import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR   = '/kaggle/input/chest-xray-pneumonia/chest_xray'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


# ── Rebuild & load model ──────────────────────────────────────────────
def build_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, 2)
    )
    return m

model = build_model()
model.load_state_dict(torch.load('/kaggle/working/pneumonia_best.pth', map_location=DEVICE))
model.eval().to(DEVICE)
print("✓ Model loaded")



# ── Test dataloader ───────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_ds     = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), test_tf)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
print(f"Test samples: {len(test_ds)} | Class mapping: {test_ds.class_to_idx}")


# ── Run inference on full test set ────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        logits = model(imgs.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[:, 1]  # prob of PNEUMONIA
        preds  = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── Print metrics ─────────────────────────────────────────────────────
print("=" * 55)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
print(f"ROC-AUC Score : {roc_auc_score(all_labels, all_probs):.4f}")
acc = (all_preds == all_labels).mean()
print(f"Overall Accuracy: {acc*100:.2f}%")
print("=" * 55)



# ── Confusion matrix + ROC curve ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cmap='Blues', linewidths=.5, annot_kws={'size': 14})
axes[0].set_title('Confusion Matrix', fontsize=13)
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Per-class accuracy from CM
for i, cls in enumerate(CLASS_NAMES):
    cls_acc = cm[i, i] / cm[i].sum() * 100
    print(f"{cls} accuracy: {cls_acc:.1f}%")

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_val = roc_auc_score(all_labels, all_probs)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc_val:.3f}')
axes[1].plot([0,1],[0,1], 'k--', lw=1, label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=13)
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/test_results.png', dpi=150)
plt.show()



# ── Visual spot check: 12 random test images ─────────────────────────
import random
from PIL import Image

infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Collect all test image paths
test_paths, test_true = [], []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(DATA_DIR, 'test', cls_name)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_paths.append(os.path.join(cls_dir, fname))
            test_true.append(cls_idx)

indices = random.sample(range(len(test_paths)), 12)

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for ax, idx in zip(axes, indices):
    img   = Image.open(test_paths[idx]).convert('RGB')
    true  = CLASS_NAMES[test_true[idx]]

    tensor = infer_tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = CLASS_NAMES[probs.argmax().item()]
        conf   = probs.max().item() * 100

    correct = pred == true
    ax.imshow(img, cmap='gray')
    ax.set_title(f'True : {true}\nPred : {pred} ({conf:.1f}%)',
                 color='green' if correct else 'red', fontsize=9)
    ax.axis('off')

plt.suptitle('Spot Check  —  Green = Correct   Red = Wrong', fontsize=13)
plt.tight_layout()
plt.savefig('/kaggle/working/spot_check.png', dpi=150, bbox_inches='tight')
plt.show()

