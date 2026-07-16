"""Training script for Translation Efficiency Network (TEN).

This script was extracted from the original kmTEN_5z.py, which mixed
training and inference code in one file.  Run this once to perform
5-fold cross-validation; the resulting ``.pth`` checkpoints (one per
fold) are then loaded by ``TENReward`` at inference time.

Usage
-----
    python train_ten2.py
"""

import csv
import math
import logging
import random
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from Bio import SeqIO
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader, Subset

from model import (
    TEN,
    AttentionResidualBlock,
    INPUT_SIZE,
    HIDDEN_SIZES,
    NUM_CLASSES,
    NHEAD,
    DROPOUT,
)

# ---------------------------------------------------------------------------
# Configuration (training-specific)
# ---------------------------------------------------------------------------

SAMPLING_RATE: float = 0.10
LEARNING_RATE: float = 0.0005
WEIGHT_DECAY: float = 0.001
BATCH_SIZE: int = 256
DEFAULT_SEED: int = 628
NUM_FOLDS: int = 5
NUM_EPOCHS: int = 100
MODEL_DIR: Path = Path(f"./model/")
LOSSDATA: Path = Path("LOSSDATA")

# Run only folds in [FOLD_START, FOLD_END] (1-based).  Keep 1/5 for the full
# cross-validation; use 2/2 to reproduce only fold 2.
FOLD_START: int = 1
FOLD_END: int = NUM_FOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: List[int]):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.one_hot_labels = self._build_one_hot(labels)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.one_hot_labels[idx]

    @staticmethod
    def _build_one_hot(labels, num_classes=NUM_CLASSES):
        labels_t = torch.LongTensor(labels).unsqueeze(1)
        one_hot = torch.zeros(len(labels), num_classes)
        return one_hot.scatter_(1, labels_t, 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    seq_file: Path,
    embeddings_file: Path,
    fasta_file: Path,
) -> Tuple[List[str], np.ndarray, List[float]]:
    codon_seq: List[str] = []
    forward_embed: List[List[float]] = []
    real_expression: List[float] = []

    with open(seq_file) as f:
        for record in SeqIO.parse(f, "fasta"):
            codon_seq.append(str(record.seq))

    with open(embeddings_file) as f:
        for line in f:
            cleaned = line.strip().strip("[]")
            tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
            forward_embed.append([float(t) for t in tokens])

    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                expression = line.strip().split()
                real_expression.append(float(expression[0]))

    return codon_seq, np.array(forward_embed), real_expression


def stratify_and_label(
    codon_sequences: List[str],
    expressions: List[float],
    seed: int,
) -> Tuple[List[int], np.ndarray]:
    n = len(expressions)
    assert len(codon_sequences) == n
    assert 0 < SAMPLING_RATE <= 1
    assert NUM_CLASSES > 0

    output_dir = MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "628kmxclassify_sampled_data.csv"

    indices = np.arange(n)
    strata = np.array_split(indices, NUM_CLASSES)
    labels = np.zeros(n, dtype=int)
    for layer_idx, layer_indices in enumerate(strata):
        labels[layer_indices] = layer_idx

    sampled_data: List[Tuple[str, float, int]] = []
    test_indices: List[int] = []
    for layer_idx, layer_indices in enumerate(strata):
        layer_size = len(layer_indices)
        if layer_size == 0:
            continue
        k = max(min(math.ceil(layer_size * SAMPLING_RATE), layer_size), 1)
        chosen = random.sample(list(layer_indices), k)
        test_indices.extend(chosen)
        for idx in chosen:
            sampled_data.append((codon_sequences[idx], expressions[idx], layer_idx))

    sampled_data.sort(key=lambda x: x[1])
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CDS", "TE", "label"])
        for seq, expr, lab in sampled_data:
            writer.writerow([seq, f"{expr:.4f}", lab])

    return labels.tolist(), np.array(sorted(test_indices))


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_embeddings, batch_labels, batch_one_hot_labels in train_loader:
        batch_embeddings = batch_embeddings.to(device)
        batch_one_hot_labels = batch_one_hot_labels.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_one_hot_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = (correct / max(total, 1)) * 100
    return avg_loss, accuracy


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch_embeddings, batch_labels, batch_one_hot_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            batch_one_hot_labels = batch_one_hot_labels.to(device)
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_one_hot_labels)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = (correct / max(total, 1)) * 100
    return avg_loss, np.array(all_probs), np.array(all_labels), accuracy


def evaluate_test(
    model_class,
    dataset,
    test_indices,
    checkpoint_paths,
    criterion,
    device,
):
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    fold_accuracies = []
    fold_predictions = []
    fold_roc_data = {i: [] for i in range(NUM_CLASSES)}
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = model_class(
            input_size=checkpoint["input_size"],
            hidden_sizes=checkpoint["hidden_sizes"],
            num_classes=checkpoint["num_classes"],
            nhead=checkpoint.get("nhead", NHEAD),
            dropout=checkpoint.get("dropout_rate", DROPOUT),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device)
        _, probs, labels, accuracy = evaluate_model(model, test_loader, criterion, device)
        fold_accuracies.append(accuracy)
        fold_predictions.append((probs, labels))
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
            auc_score = auc(fpr, tpr)
            fold_roc_data[i].append({"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_score})

    accuracy = float(np.mean(fold_accuracies))
    roc_data = []
    for i in range(NUM_CLASSES):
        auc_score = float(np.mean([d["auc"] for d in fold_roc_data[i]]))
        roc_data.append({"class": i, "fpr": fold_roc_data[i][-1]["fpr"], "tpr": fold_roc_data[i][-1]["tpr"], "auc": auc_score})
        logger.info("Test Class %d AUC: %.4f", i, auc_score)

    (MODEL_DIR / LOSSDATA).mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / LOSSDATA / "628test_preds_true.txt"
    with open(out, "w") as f:
        for fold_no, (probs, labels) in enumerate(fold_predictions, start=1):
            f.write(f"Fold {fold_no}:\n")
            f.write(f"Predictions: {probs.tolist()}\n")
            f.write(f"True Labels: {labels.tolist()}\n")
            f.write(f"Accuracy: {fold_accuracies[fold_no - 1]:.4f}%\n\n")
        f.write(f"Test accuracy: {accuracy:.4f}%\n")

    out = MODEL_DIR / LOSSDATA / "628test_roc_data.txt"
    with open(out, "w") as f:
        for d in roc_data:
            f.write(f"Class {d['class']}:\nAUC: {d['auc']:.4f}\nfpr: {d['fpr']}\ntpr: {d['tpr']}\n\n")

    return accuracy


def cross_validate(
    model_class,
    dataset,
    k,
    criterion,
    optimizer_fn,
    num_epochs,
    seed,
    device,
):
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_train_accs: List[List[float]] = []
    fold_train_losses: List[List[float]] = []
    fold_val_losses: List[List[float]] = []
    roc_data: List[Dict[str, Any]] = []
    final_predictions: List[Dict[str, Any]] = []
    intermediate_results: List[Dict[str, Any]] = []
    fold_checkpoints: List[Path] = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fold_no = fold + 1
        if fold_no < FOLD_START or fold_no > FOLD_END:
            continue
        # Reset the global RNG at the start of each fold so the fold is
        # reproducible in isolation and independent of the total epoch count.
        set_global_seed(seed + fold_no)
        logger.info("=== Fold %d / %d ===", fold_no, k)
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=True)
        model = model_class().to(device)
        optimizer = optimizer_fn(model.parameters())

        epoch_train_accs: List[float] = []
        epoch_train_losses: List[float] = []
        epoch_val_losses: List[float] = []

        val_probs: Optional[np.ndarray] = None
        val_labels: Optional[np.ndarray] = None

        for epoch in range(num_epochs):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, epoch_probs, epoch_labels, val_acc = evaluate_model(model, val_loader, criterion, device)
            epoch_train_accs.append(train_acc)
            epoch_train_losses.append(train_loss)
            epoch_val_losses.append(val_loss)
            val_probs = epoch_probs
            val_labels = epoch_labels

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %3d/%d  Train Loss: %.4f  Train Acc: %.2f%%  Val Loss: %.4f  Val Acc: %.2f%%",
                            epoch + 1, num_epochs, train_loss, train_acc, val_loss, val_acc)
            if (epoch + 1) % 100 == 0:
                intermediate_results.append({"fold": fold, "epochs_ran": epoch + 1, "train_acc": train_acc, "val_acc": val_acc})

        fold_train_accs.append(epoch_train_accs)
        fold_train_losses.append(epoch_train_losses)
        fold_val_losses.append(epoch_val_losses)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_size": INPUT_SIZE,
            "hidden_sizes": HIDDEN_SIZES,
            "num_classes": NUM_CLASSES,
            "use_transformer_residual": True,
            "nhead": NHEAD,
            "dropout_rate": DROPOUT,
        }
        checkpoint_path = MODEL_DIR / f"628kmTEN_fold{fold + 1}.pth"
        torch.save(ckpt, checkpoint_path)
        fold_checkpoints.append(checkpoint_path)

        assert val_labels is not None and val_probs is not None
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(val_labels == i, val_probs[:, i])
            auc_score = auc(fpr, tpr)
            roc_data.append({"fold": fold + 1, "class": i, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_score})
            logger.info("Fold %d, Class %d AUC: %.4f", fold + 1, i, auc_score)

        final_predictions.append({"fold": fold + 1, "predictions": val_probs, "true_labels": val_labels})

    _save_outputs(final_predictions, fold_train_accs, fold_train_losses, fold_val_losses, roc_data)
    return intermediate_results, fold_checkpoints


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _save_outputs(predictions, train_accs, train_losses, val_losses, roc_data):
    (MODEL_DIR / LOSSDATA).mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / LOSSDATA / "628finalepoch_preds_true.txt"
    with open(out, "w") as f:
        for e in predictions:
            f.write(f"Fold {e['fold']}:\nPredictions: {e['predictions'].tolist()}\nTrue Labels: {e['true_labels'].tolist()}\n\n")

    out = MODEL_DIR / LOSSDATA / "628train_val_losses.txt"
    with open(out, "w") as f:
        for fold, (acc, tl, vl) in enumerate(zip(train_accs, train_losses, val_losses)):
            f.write(f"Fold {fold + 1}:\nTrain accuracy: {acc}\nTrain Losses: {tl}\nVal Losses: {vl}\n\n")

    out = MODEL_DIR / LOSSDATA / "628roc_data.txt"
    with open(out, "w") as f:
        for d in roc_data:
            f.write(f"Fold {d['fold']}, Class {d['class']}:\nAUC: {d['auc']:.4f}\nfpr: {d['fpr']}\ntpr: {d['tpr']}\n\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    seed = DEFAULT_SEED
    set_global_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_dir = Path("./traindata")
    codon_seq, embeddings, real_expression = load_data(
        data_dir / "0614kmx_codon_seq.fasta",
        data_dir / "0614kmx_embeddings.txt",
        data_dir / "0614kmx_expression.fasta",
    )
    labels, test_indices = stratify_and_label(codon_seq, real_expression, seed)
    dataset = EmbeddingDataset(embeddings, labels)
    train_indices = np.setdiff1d(np.arange(len(labels)), test_indices)
    train_dataset = Subset(dataset, train_indices)

    criterion = nn.CrossEntropyLoss()
    optimizer_fn = lambda params: torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info("Starting %d-fold cross-validation for %d epochs ", NUM_FOLDS, NUM_EPOCHS)
    _, fold_checkpoints = cross_validate(TEN, train_dataset, NUM_FOLDS, criterion, optimizer_fn, NUM_EPOCHS, seed, device)
    test_accuracy = evaluate_test(TEN, dataset, test_indices, fold_checkpoints, criterion, device)
    logger.info("Test accuracy: %.4f%%", test_accuracy)
    logger.info("Done.")


if __name__ == "__main__":
    main()
