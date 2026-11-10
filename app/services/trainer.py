from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
import json
import pickle
import shutil
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: str | torch.device | None = None,
        lr: float = 0.003,
        weight_decay: float = 0.0,
        patience: int = 5,
        min_delta: float = 0.0,
        model_dir: str | Path | None = None,
        scaler_dir: str | Path | None = None,
        model_config: dict | None = None,
        scaler_state: dict | None = None,
        verbose: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module.")
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        
        root = Path(__file__).resolve().parents[2] / "storage"
        self.__model = model
        self.__device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.__model.to(self.__device)
        self.__criterion = criterion or nn.CrossEntropyLoss()
        self.__optimizer = optimizer or torch.optim.Adam(self.__model.parameters(), lr=lr, weight_decay=weight_decay)
        self.__patience = patience
        self.__min_delta = float(min_delta)
        
        # Every training run writes to a timestamped version directory.
        # The best run (by validation macro-F1) is also copied to release/ for inference.
        self.__version_root = Path(model_dir) if model_dir is not None else root / "models" / "versions"
        self.__release_dir = self.__version_root.parents[0] / "release"
        self.__scaler_dir = Path(scaler_dir) if scaler_dir is not None else self.__version_root
        self.__model_config = dict(model_config or {})
        self.__scaler_state = dict(scaler_state or {})
        self.__verbose = verbose
        self.__project_root = self.__resolve_project_root()
    
    def fit(self, train_loader: DataLoader, validation_loader: DataLoader, epochs: int, ) -> dict:
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        
        # history stores per-epoch train/validation metrics so curves can be
        # plotted after training and results are reproducible from the log alone.
        history = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_f1": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_macro_f1": [],
        }
        best_epoch = 0
        best_loss = float("inf")
        best_macro_f1 = 0.0
        best_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_confusion_matrix = []
        best_state = deepcopy(self.__model.state_dict())
        wait = 0
        stopped_early = False
        
        if self.__verbose:
            self.__print_start_banner(epochs, train_loader, validation_loader)
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.__train_epoch(train_loader)
            validation_metrics = self.__validate_epoch(validation_loader)
            
            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_macro_f1"].append(train_metrics["macro_f1"])
            history["validation_loss"].append(validation_metrics["loss"])
            history["validation_accuracy"].append(validation_metrics["accuracy"])
            history["validation_macro_f1"].append(validation_metrics["macro_f1"])
            
            # Early stopping is driven by validation loss with a min_delta
            # threshold, so tiny noise-driven improvements do not reset the wait
            # counter. The best weights (not the last epoch) are restored at the end.
            improved = validation_metrics["loss"] < best_loss - self.__min_delta
            if improved:
                best_loss = validation_metrics["loss"]
                best_macro_f1 = validation_metrics["macro_f1"]
                best_accuracy = validation_metrics["accuracy"]
                best_precision = validation_metrics["precision"]
                best_recall = validation_metrics["recall"]
                best_confusion_matrix = validation_metrics["confusion_matrix"]
                best_epoch = epoch
                best_state = deepcopy(self.__model.state_dict())
                wait = 0
            else:
                wait += 1
            
            if self.__verbose:
                self.__print_epoch(
                    epoch=epoch,
                    epochs=epochs,
                    best_epoch=best_epoch,
                    wait=wait,
                    patience=self.__patience,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                )
            
            if wait >= self.__patience:
                stopped_early = True
                break
        
        # Restore the weights from the best epoch, not the last epoch.
        self.__model.load_state_dict(best_state)
        result = {
            **history,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_accuracy": best_accuracy,
            "best_validation_precision": best_precision,
            "best_validation_recall": best_recall,
            "best_validation_confusion_matrix": best_confusion_matrix,
            "stopped_early": stopped_early,
        }
        result["artifacts"] = self.__save_artifacts(result)
        
        if self.__verbose:
            self.__print_end_banner(best_epoch, best_accuracy, best_loss, best_macro_f1)
        
        return result
    
    def __save_artifacts(self, result: dict) -> dict:
        # Each run creates an immutable version directory named
        # <modelclass>_<UTC timestamp>. The release/ folder is updated separately.
        self.__version_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{self.__model.__class__.__name__.lower()}_{stamp}"
        version_dir = self.__version_root / stem
        version_dir.mkdir(parents=True, exist_ok=False)
        
        model_path = version_dir / "model.pt"
        scaler_path = version_dir / "scaler.pkl"
        metadata_path = version_dir / "metadata.json"
        curves_dir = version_dir / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)
        
        saved_at = datetime.now(UTC).isoformat()
        
        # model.pt contains everything needed to reconstruct the model
        # (class name, model_config, state_dict) plus the full training result.
        torch.save({
            "model_class": self.__model.__class__.__name__,
            "model_state_dict": self.__model.state_dict(),
            "optimizer_state_dict": self.__optimizer.state_dict(),
            "criterion_class": self.__criterion.__class__.__name__,
            "device": str(self.__device),
            "trainer_config": {"patience": self.__patience, "min_delta": self.__min_delta},
            "model_config": self.__model_config,
            "result": result,
            "saved_at": saved_at,
        }, model_path)
        
        # The scaler/encoder state is saved separately so inference can apply
        # the exact same preprocessing that was used during training.
        with scaler_path.open("wb") as file:
            pickle.dump({
                "model_class": self.__model.__class__.__name__,
                "model_config": self.__model_config,
                "scaler_state": self.__scaler_state,
                "saved_at": saved_at,
            }, file)
        
        curve_files = self.__plot_curves(result, curves_dir)
        
        # metadata.json is the human-readable summary used by release comparison
        # and by inference (class_names, image_size, model_config).
        metadata = {
            "model_class": self.__model.__class__.__name__,
            "device": str(self.__device),
            "model_config": self.__model_config,
            "trainer_config": {"patience": self.__patience, "min_delta": self.__min_delta},
            "artifacts": {
                "version_dir": version_dir.name,
                "model_path": model_path.name,
                "scaler_path": scaler_path.name,
                "metadata_path": metadata_path.name,
                "curves_dir": curves_dir.name,
                "curves": curve_files,
            },
            "result_summary": {
                "best_epoch": result["best_epoch"],
                "best_validation_loss": round(result["best_validation_loss"], 4),
                "best_validation_macro_f1": round(result["best_validation_macro_f1"], 4),
                "best_validation_accuracy": round(result["best_validation_accuracy"], 4),
                "best_validation_precision": round(result["best_validation_precision"], 4),
                "best_validation_recall": round(result["best_validation_recall"], 4),
                "stopped_early": result["stopped_early"],
            },
            "saved_at": saved_at,
        }
        
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        release_updated = self.__sync_release(version_dir, metadata)
        
        return {
            "version_dir": self.__relative(version_dir),
            "model_path": self.__relative(model_path),
            "scaler_path": self.__relative(scaler_path),
            "metadata_path": self.__relative(metadata_path),
            "curves_dir": self.__relative(curves_dir),
            "curves": curve_files,
            "release_dir": self.__relative(self.__release_dir),
            "release_updated": release_updated,
        }
    
    def __plot_curves(self, result: dict, curves_dir: Path) -> dict:
        # Three curves are saved per run: loss, macro-F1 and accuracy.
        # A vertical dashed line marks the best epoch so overfitting
        # (train improves, validation degrades) is immediately visible.
        epochs = result["epoch"]
        if not epochs:
            return {}
            
        saved: dict[str, str] = {}
        
        loss_path = curves_dir / "loss_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, result["train_loss"], label="train", marker="o")
        ax.plot(epochs, result["validation_loss"], label="validation", marker="o")
        if result.get("best_epoch"):
            ax.axvline(result["best_epoch"], linestyle="--", color="gray", label="best epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training vs Validation Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(loss_path, dpi=150)
        plt.close(fig)
        saved["loss_curve"] = loss_path.name
        
        f1_path = curves_dir / "macro_f1_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, result["train_macro_f1"], label="train", marker="o")
        ax.plot(epochs, result["validation_macro_f1"], label="validation", marker="o")
        if result.get("best_epoch"):
            ax.axvline(result["best_epoch"], linestyle="--", color="gray", label="best epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro F1")
        ax.set_title("Training vs Validation Macro F1")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f1_path, dpi=150)
        plt.close(fig)
        saved["macro_f1_curve"] = f1_path.name
        
        acc_path = curves_dir / "accuracy_curve.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, result["train_accuracy"], label="train", marker="o")
        ax.plot(epochs, result["validation_accuracy"], label="validation", marker="o")
        if result.get("best_epoch"):
            ax.axvline(result["best_epoch"], linestyle="--", color="gray", label="best epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_title("Training vs Validation Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(acc_path, dpi=150)
        plt.close(fig)
        saved["accuracy_curve"] = acc_path.name
        
        return saved
    
    def __sync_release(self, version_dir: Path, metadata: dict) -> bool:
        # The release folder always contains the single best model so far,
        # ranked by (macro-F1, -loss, accuracy) in that order.
        # A new run only replaces it if it beats the existing key lexicographically.
        self.__release_dir.mkdir(parents=True, exist_ok=True)
        release_metadata = self.__release_dir / "metadata.json"
        candidate = metadata["result_summary"]
        
        if release_metadata.exists():
            existing = json.loads(release_metadata.read_text(encoding="utf-8-sig"))["result_summary"]
            candidate_key = (
                candidate["best_validation_macro_f1"],
                -candidate["best_validation_loss"],
                candidate["best_validation_accuracy"],
            )
            existing_key = (
                existing["best_validation_macro_f1"],
                -existing["best_validation_loss"],
                existing["best_validation_accuracy"],
            )
            if existing_key > candidate_key:
                return False
        
        for path in self.__release_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        
        # Only model.pt, scaler.pkl and metadata.json are copied to release/.
        # Curves and the full result stay in the version directory.
        for name in ("model.pt", "scaler.pkl", "metadata.json"):
            shutil.copy2(version_dir / name, self.__release_dir / name)
        
        return True
    
    def __train_epoch(self, loader: DataLoader) -> dict:
        return self.__run_epoch(loader, training=True)
    
    def __validate_epoch(self, loader: DataLoader) -> dict:
        return self.__run_epoch(loader, training=False)
    
    def __run_epoch(self, loader: DataLoader, training: bool) -> dict:
        # Shared implementation for train and validation:
        # - training mode enables gradient tracking and optimizer steps
        # - validation mode wraps everything in torch.no_grad()
        # Metrics are computed from a confusion matrix accumulated on CPU with bincount.
        self.__model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        confusion = None
        context = torch.enable_grad if training else torch.no_grad
        
        with context():
            for inputs, targets in loader:
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)
                
                if training:
                    self.__optimizer.zero_grad()
                
                logits = self.__model(inputs)
                loss = self.__criterion(logits, targets)
                
                if training:
                    loss.backward()
                    self.__optimizer.step()
                
                predictions = logits.argmax(dim=1)
                batch_size = targets.size(0)
                total_samples += batch_size
                total_loss += loss.item() * batch_size
                total_correct += (predictions == targets).sum().item()
                if confusion is None:
                    num_classes = logits.size(1)
                    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
                
                # Flatten (true, pred) pairs into a single index and accumulate
                # with bincount, then reshape back to the (C, C) confusion matrix.
                indices = (targets.detach().cpu() * confusion.size(0)) + predictions.detach().cpu()
                confusion += torch.bincount(indices, minlength=confusion.numel()).reshape(confusion.shape)
        
        if total_samples == 0:
            raise ValueError("DataLoader must contain at least one sample.")
        
        # Macro-averaged precision, recall and F1 computed directly from the
        # confusion matrix (no sklearn dependency during training).
        tp = confusion.diag().to(torch.float32)
        fp = confusion.sum(dim=0).to(torch.float32) - tp
        fn = confusion.sum(dim=1).to(torch.float32) - tp
        precision = (tp / (tp + fp).clamp_min(1.0)).mean().item()
        recall = (tp / (tp + fn).clamp_min(1.0)).mean().item()
        macro_f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1.0)).mean().item()
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "precision": precision,
            "recall": recall,
            "macro_f1": macro_f1,
            "confusion_matrix": confusion.tolist(),
        }
    
    def __resolve_project_root(self) -> Path:
        # Walk up from this file until a directory containing configs/settings.py
        # is found; all reported paths are made relative to this root.
        for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
            if (candidate / "configs" / "settings.py").exists():
                return candidate
        return Path(__file__).resolve().parents[2]
    
    def __relative(self, path: Path) -> str:
        # Artifact paths are reported relative to the project root so logs and
        # JSON output stay portable across machines.
        try:
            return str(path.resolve().relative_to(self.__project_root))
        except ValueError:
            return str(path)
    
    def __print_start_banner(self, epochs: int, train_loader: DataLoader, validation_loader: DataLoader) -> None:
        line = "-" * 40
        print(line)
        print("START TRAINING".center(40))
        print(line)
        rows = [
            ("Device", str(self.__device).upper()),
            ("Epochs", str(epochs)),
            ("Patience", str(self.__patience)),
            ("Learning Rate", f"{self.__optimizer.param_groups[0]['lr']}"),
            ("Dropout", f"{self.__model_config.get('dropout', 'n/a')}"),
            ("Train Batch Size", str(train_loader.batch_size)),
            ("Validation Batch Size", str(validation_loader.batch_size)),
        ]
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"- {label.ljust(width)} : {value}")
        print(line)
    
    def __print_epoch(
        self,
        *,
        epoch: int,
        epochs: int,
        best_epoch: int,
        wait: int,
        patience: int,
        train_metrics: dict,
        validation_metrics: dict,
    ) -> None:
        # One block per epoch: best-so-far and wait counter on top, train and
        # validation metrics below. The block is easy to scan in a terminal log.
        line = "-" * 40
        width = max(3, len(str(epochs)))
        print(f"[{epoch:0{width}d}/{epochs:0{width}d}]")
        print(f"- Best Epoch : {best_epoch:0{max(2, len(str(epochs)))}d}")
        print(f"- Waited     : {wait:02d} / {patience:02d} epochs")
        print()
        print("- Training:")
        print(f"    - Accuracy  : {train_metrics['accuracy'] * 100:.2f}%")
        print(f"    - Loss      : {train_metrics['loss']:.4f}")
        print(f"    - F1 Score  : {train_metrics['macro_f1']:.4f}")
        print("- Validation:")
        print(f"    - Accuracy  : {validation_metrics['accuracy'] * 100:.2f}%")
        print(f"    - Loss      : {validation_metrics['loss']:.4f}")
        print(f"    - F1 Score  : {validation_metrics['macro_f1']:.4f}")
        print(f"    - Precision : {validation_metrics['precision']:.4f}")
        print(f"    - Recall    : {validation_metrics['recall']:.4f}")
        print(line)
    
    def __print_end_banner(self, best_epoch: int, best_accuracy: float, best_loss: float, best_f1: float) -> None:
        line = "-" * 40
        print("END TRAINING".center(40))
        print(line)
        print(f"- Best Epoch : {best_epoch:02d}")
        print(f"    - Accuracy  : {best_accuracy * 100:.2f}%")
        print(f"    - Loss      : {best_loss:.4f}")
        print(f"    - F1 Score  : {best_f1:.4f}")
        print(line)
