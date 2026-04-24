import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Prefer pykan implementation (requested).
# Install with: pip install pykan
from kan import KAN


# =============================
# Easy-to-edit configuration
# =============================
@dataclass
class Config:
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 40
    min_epochs: int = 30
    early_stopping_patience: int = 5

    image_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    num_classes: int = 10

    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    mlp_ratio: int = 4

    # pykan settings (k=3 means cubic splines in pykan)
    kan_grid: int = 3
    kan_k: int = 3

    val_split: float = 0.1
    num_workers: int = 2


cfg = Config()


class KANLinearWrapper(nn.Module):
    """
    Wrapper around pykan.KAN to behave like a token-wise linear layer.

    Input shape : [batch, tokens, dim]
    Internal    : flatten to [batch * tokens, dim]
    Output shape: [batch, tokens, out_dim]

    This lets us drop KAN into a Transformer FFN where nn.Linear is usually used.
    """

    def __init__(self, in_dim: int, out_dim: int, grid: int = 3, k: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Simple 2-layer KAN mapping: in_dim -> out_dim
        self.kan = KAN(width=[in_dim, out_dim], grid=grid, k=k)

        # pykan recommendation for speed when symbolic branch is not used.
        if hasattr(self.kan, "speed"):
            self.kan.speed()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {d}")

        x_flat = x.reshape(b * t, d)
        y_flat = self.kan(x_flat)
        y = y_flat.reshape(b, t, self.out_dim)
        return y


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, cfg: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = embed_dim * mlp_ratio

        # KAN replaces ONLY the first FFN linear layer.
        self.mlp_fc1 = KANLinearWrapper(
            in_dim=embed_dim,
            out_dim=hidden_dim,
            grid=cfg.kan_grid,
            k=cfg.kan_k,
        )

        self.act = nn.GELU()
        # Keep second FFN linear layer standard.
        self.mlp_fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard self-attention block with residual.
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # Standard FFN block with residual.
        y = self.norm2(x)
        y = self.mlp_fc1(y)  # <- KAN replacement point.
        y = self.act(y)
        y = self.mlp_fc2(y)
        x = x + y
        return x


class SimpleViT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.image_size % cfg.patch_size == 0, "Patch size must divide image size."

        self.cfg = cfg
        self.num_patches_per_side = cfg.image_size // cfg.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_dim = cfg.in_channels * cfg.patch_size * cfg.patch_size

        # Patch embedding stays standard.
        self.patch_proj = nn.Linear(self.patch_dim, cfg.embed_dim)

        # Positional and class tokens stay standard.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, cfg.embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    cfg=cfg,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, H, W] -> [B, num_patches, patch_dim]."""
        b, c, _, _ = x.shape
        p = self.cfg.patch_size

        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(b, c, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, self.num_patches, -1)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        patches = self._patchify(x)
        tokens = self.patch_proj(patches)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_repr = x[:, 0]
        logits = self.head(cls_repr)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    return total_loss / total


@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def gather_test_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()
    return y_true, y_pred, y_prob


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Train/validation split.
    val_size = int(len(train_full) * cfg.val_split)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleViT(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    no_improve = 0

    print("\nStarting training...")
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{cfg.max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
            torch.save(best_state_dict, "best_kan_transformer_mnist.pt")
        else:
            no_improve += 1

        # Must train at least min_epochs.
        if epoch >= cfg.min_epochs and no_improve >= cfg.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no val loss improvement for {cfg.early_stopping_patience} epochs)."
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nBest model saved as best_kan_transformer_mnist.pt (epoch {best_epoch})")

    # Final test metrics.
    y_true, y_pred, y_prob = gather_test_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # One-vs-rest multi-class ROC-AUC from softmax probabilities.
    y_true_bin = label_binarize(y_true, classes=list(range(cfg.num_classes)))
    roc_auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro")

    print("\n===== Final Test Metrics =====")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f} (weighted)")
    print(f"Recall       : {recall:.4f} (weighted)")
    print(f"F1-score     : {f1:.4f} (weighted)")
    print(f"ROC-AUC (OvR): {roc_auc:.4f} (macro)")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
