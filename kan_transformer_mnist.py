import math
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

    # KAN layer settings
    spline_degree: int = 3  # cubic B-splines
    spline_num_bases: int = 8
    spline_x_min: float = -3.0
    spline_x_max: float = 3.0

    val_split: float = 0.1
    num_workers: int = 2


cfg = Config()


class KANLinear(nn.Module):
    """
    A simple KAN-style linear layer.

    output = base_linear(x) + spline_output(x)

    - base_linear(x): normal affine transform for stability.
    - spline_output(x): additional transform built from trainable
      B-spline basis coefficients.

    For each input dimension, we compute B-spline basis values, then combine
    them with trainable coefficients to produce each output dimension.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 3,
        num_bases: int = 8,
        x_min: float = -3.0,
        x_max: float = 3.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        self.num_bases = num_bases
        self.x_min = x_min
        self.x_max = x_max

        # Base linear component (standard nn.Linear behavior).
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Spline coefficients:
        # one set of basis coefficients per (output_dim, input_dim).
        self.spline_coeff = nn.Parameter(
            torch.zeros(out_features, in_features, num_bases)
        )

        # Build a clamped uniform knot vector as a non-trainable buffer.
        knots = self._build_uniform_clamped_knots(
            num_bases=num_bases,
            degree=degree,
            x_min=x_min,
            x_max=x_max,
        )
        self.register_buffer("knots", knots)

        self.reset_parameters()

    @staticmethod
    def _build_uniform_clamped_knots(
        num_bases: int,
        degree: int,
        x_min: float,
        x_max: float,
    ) -> torch.Tensor:
        """
        Build open-uniform (clamped) knots.

        Number of knots = num_bases + degree + 1
        Interior knot count = num_bases - degree - 1
        """
        n_knots = num_bases + degree + 1
        n_interior = num_bases - degree - 1

        left = torch.full((degree + 1,), float(x_min))
        right = torch.full((degree + 1,), float(x_max))

        if n_interior > 0:
            # Uniform interior knots excluding endpoints.
            interior = torch.linspace(x_min, x_max, n_interior + 2)[1:-1]
            knots = torch.cat([left, interior, right], dim=0)
        else:
            knots = torch.cat([left, right], dim=0)

        if knots.numel() != n_knots:
            raise ValueError("Incorrect knot vector length.")

        return knots

    def reset_parameters(self) -> None:
        # Standard init for base linear part.
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        if self.base_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.base_bias, -bound, bound)

        # Small init so spline part starts gentle.
        nn.init.normal_(self.spline_coeff, mean=0.0, std=0.01)

    def _bspline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis values for each scalar in x.

        Args:
            x: Tensor with shape [N, in_features]

        Returns:
            basis: Tensor [N, in_features, num_bases]
        """
        x = x.clamp(self.x_min, self.x_max)
        knots = self.knots
        degree = self.degree
        n_knots = knots.numel()

        # Degree-0 basis: indicator of knot intervals.
        # Count = n_knots - 1.
        intervals = n_knots - 1
        B = []
        for i in range(intervals):
            left = knots[i]
            right = knots[i + 1]
            # [left, right) for most intervals.
            # For the final interval include right endpoint.
            if i == intervals - 1:
                bi = ((x >= left) & (x <= right)).to(x.dtype)
            else:
                bi = ((x >= left) & (x < right)).to(x.dtype)
            B.append(bi)

        B = torch.stack(B, dim=-1)  # [N, in_features, intervals]

        # Cox-de Boor recursion up to desired degree.
        for d in range(1, degree + 1):
            new_count = n_knots - d - 1
            B_next = []
            for i in range(new_count):
                left_den = knots[i + d] - knots[i]
                right_den = knots[i + d + 1] - knots[i + 1]

                if float(left_den) == 0.0:
                    left_term = torch.zeros_like(B[..., i])
                else:
                    left_term = ((x - knots[i]) / left_den) * B[..., i]

                if float(right_den) == 0.0:
                    right_term = torch.zeros_like(B[..., i + 1])
                else:
                    right_term = ((knots[i + d + 1] - x) / right_den) * B[..., i + 1]

                B_next.append(left_term + right_term)

            B = torch.stack(B_next, dim=-1)

        # Guarantee exact endpoint behavior at x_max.
        x_max_mask = (x == self.x_max).unsqueeze(-1)
        if x_max_mask.any():
            B = torch.where(x_max_mask, torch.zeros_like(B), B)
            B[..., -1] = torch.where(
                (x == self.x_max),
                torch.ones_like(B[..., -1]),
                B[..., -1],
            )

        return B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [..., in_features]
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)

        # Base linear output.
        base_out = F.linear(x2d, self.base_weight, self.base_bias)

        # Spline basis values for each input dimension.
        # basis shape: [N, in_features, num_bases]
        basis = self._bspline_basis(x2d)

        # Combine basis with trainable coefficients.
        # spline_coeff shape: [out_features, in_features, num_bases]
        # Output shape after einsum: [N, out_features]
        spline_out = torch.einsum("nib,oib->no", basis, self.spline_coeff)

        out = base_out + spline_out
        out = out.reshape(*original_shape[:-1], self.out_features)
        return out


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
        self.mlp_fc1 = KANLinear(
            in_features=embed_dim,
            out_features=hidden_dim,
            degree=cfg.spline_degree,
            num_bases=cfg.spline_num_bases,
            x_min=cfg.spline_x_min,
            x_max=cfg.spline_x_max,
        )
        self.act = nn.GELU()
        self.mlp_fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard pre-norm attention with residual.
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        # Standard pre-norm FFN with residual.
        y = self.norm2(x)
        y = self.mlp_fc1(y)      # First linear replaced by KANLinear.
        y = self.act(y)
        y = self.mlp_fc2(y)      # Second linear stays nn.Linear.
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

        # Patch embedding.
        self.patch_proj = nn.Linear(self.patch_dim, cfg.embed_dim)

        # CLS token and position embeddings.
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
        """
        Convert image batch [B, C, H, W] into patch sequence [B, num_patches, patch_dim].
        """
        B, C, H, W = x.shape
        p = self.cfg.patch_size

        # Unfold into non-overlapping patches.
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # Shape: [B, C, n_h, n_w, p, p]
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, self.num_patches, -1)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        patches = self._patchify(x)
        tokens = self.patch_proj(patches)

        cls_tokens = self.cls_token.expand(B, -1, -1)
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
    running_loss = 0.0
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
        running_loss += loss.item() * batch_size
        total += batch_size

    return running_loss / total


@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


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

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load MNIST.
    full_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # Train/validation split.
    val_size = int(len(full_train) * cfg.val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(
        full_train,
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
    epochs_without_improvement = 0
    best_epoch = -1

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
            epochs_without_improvement = 0
            torch.save(best_state_dict, "best_kan_transformer_mnist.pt")
        else:
            epochs_without_improvement += 1

        # Enforce at least min_epochs of training.
        if epoch >= cfg.min_epochs and epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"(No validation loss improvement for {cfg.early_stopping_patience} epochs)"
            )
            break

    # Restore best checkpoint before final evaluation.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\nBest model saved to: best_kan_transformer_mnist.pt (epoch {best_epoch})")

    # Final test evaluation metrics.
    y_true, y_pred, y_prob = gather_test_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # Multi-class ROC-AUC using one-vs-rest.
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
