import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn


class Evaluator:
    def __init__(
        self, 
        model_builder, 
        device: str | torch.device | None = None, 
        release_dir: str | Path | None = None, 
        results_root: str | Path | None = None
    ) -> None:
        root = Path(__file__).resolve().parents[2] / "storage"
        self.__model_builder = model_builder
        self.__device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.__release_dir = Path(release_dir) if release_dir is not None else root / "models" / "release"
        self.__results_root = Path(results_root) if results_root is not None else root / "test_results"
    
    def evaluate(self, test_loader, class_names: list[str] | None = None) -> dict:
        # The release directory is the single source of truth for evaluation:
        # metadata.json records the model class/config, model.pt holds the weights.
        metadata = self.__read_json(self.__release_dir / "metadata.json")
        checkpoint = torch.load(self.__release_dir / "model.pt", map_location=self.__device)
        model = self.__model_builder(dict(metadata.get("model_config") or {}))
        
        if not isinstance(model, nn.Module):
            raise TypeError("model_builder must return an nn.Module.")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.__device).eval()
        # .eval() disables dropout and switches BatchNorm to inference mode,
        # so the metrics reported here match what the deployed model would produce.
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = total_correct = total_samples = total_top_3 = 0
        total_inference_time = 0.0
        mistakes, all_true, all_pred = [], [], []
        confusion = None
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) != 3:
                    raise ValueError("test_loader must yield (inputs, targets, paths).")
                
                # Each batch must also carry the original paths, so misclassified
                # samples can later be copied and inspected by a human.
                inputs, targets, paths = batch
                inputs = inputs.to(self.__device)
                targets = targets.to(self.__device)
                
                # Inference time is measured around the forward pass only,
                # excluding data loading and metric computation.
                started = perf_counter()
                logits = model(inputs)
                
                total_inference_time += perf_counter() - started
                loss = loss_fn(logits, targets)
                preds = logits.argmax(dim=1)
                
                # Top-3 accuracy: whether the correct class appears in the three
                # highest-scoring logits. It shows the model is often "close"
                # even when top-1 is wrong, which matters for human-in-the-loop sorting.
                top_k = logits.topk(k=min(3, logits.size(1)), dim=1).indices
                probs = logits.softmax(dim=1)
                n = targets.size(0)
                total_samples += n
                total_loss += loss.item() * n
                total_correct += (preds == targets).sum().item()
                
                if confusion is None:
                    confusion = torch.zeros((logits.size(1), logits.size(1)), dtype=torch.int64)
                    if class_names is None:
                        class_names = [str(i) for i in range(logits.size(1))]
                
                true_batch = targets.cpu().tolist()
                pred_batch = preds.cpu().tolist()
                total_top_3 += top_k.eq(targets.unsqueeze(1)).any(dim=1).sum().item()
                all_true.extend(true_batch)
                all_pred.extend(pred_batch)
                
                # Build the confusion matrix incrementally on CPU with bincount:
                # index = true * num_classes + pred, then reshape back to (C, C).
                idx = (targets.cpu() * confusion.size(0)) + preds.cpu()
                confusion += torch.bincount(idx, minlength=confusion.numel()).reshape(confusion.shape)
                
                # Record every misclassified sample with its true/pred label,
                # confidence and top-3, so error analysis can be written later.
                for i, (p, t, path) in enumerate(zip(pred_batch, true_batch, paths)):
                    if p != t:
                        mistakes.append({
                            "source_path": str(path),
                            "true_index": t,
                            "pred_index": p,
                            "true_label": class_names[t],
                            "pred_label": class_names[p],
                            "confidence": round(float(probs[i, p].item()), 6),
                            "top_3_labels": [class_names[index] for index in top_k[i].cpu().tolist()],
                        })
        if total_samples == 0:
            raise ValueError("test_loader must contain at least one sample.")
        
        # Macro and weighted averages are both reported: macro treats each class
        # equally (important for a balanced sorting problem), weighted reflects
        # the actual class distribution in the test set.
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_true, all_pred, average="macro", zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_true, all_pred, average="weighted", zero_division=0)
        
        # A full per-class report and a confusion matrix are required by Task 5.
        report = classification_report(all_true, all_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))
        
        # Majority-class baseline: always predict the most frequent class.
        # The model must beat this baseline by a clear margin to show it has
        # learned class-discriminative features rather than exploiting imbalance.
        majority = max(set(all_true), key=all_true.count)
        baseline_pred = [majority] * len(all_true)
        b_macro_precision, b_macro_recall, b_macro_f1, _ = precision_recall_fscore_support(all_true, baseline_pred, average="macro", zero_division=0)
        b_weighted_precision, b_weighted_recall, b_weighted_f1, _ = precision_recall_fscore_support(all_true, baseline_pred, average="weighted", zero_division=0)
        baseline_accuracy = accuracy_score(all_true, baseline_pred)
        
        # Highest-confidence mistakes come first so the error analysis in the
        # report can focus on the most confident wrong predictions.
        mistakes = sorted(mistakes, key=lambda item: item["confidence"], reverse=True)
        result = {
            "tested_at": datetime.now(UTC).isoformat(),
            "model_class": metadata["model_class"],
            "release_dir": str(self.__release_dir),
            "sample_count": total_samples,
            "loss": total_loss / total_samples,
            "accuracy": accuracy_score(all_true, all_pred),
            "top_3_accuracy": total_top_3 / total_samples,
            "average_inference_time_ms": (total_inference_time / total_samples) * 1000,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "class_names": class_names,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "per_class": {label: report[label] for label in class_names},
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "error_analysis": mistakes[:5],
            "baseline": {
                "strategy": "majority_class",
                "majority_class_index": majority,
                "majority_class_label": class_names[majority],
                "accuracy": baseline_accuracy,
                "macro_precision": b_macro_precision,
                "macro_recall": b_macro_recall,
                "macro_f1": b_macro_f1,
                "weighted_precision": b_weighted_precision,
                "weighted_recall": b_weighted_recall,
                "weighted_f1": b_weighted_f1
            },
            "task1_comparison": {
                "objective": "6-class recyclable waste classification with balanced macro metrics over a majority baseline",
                "accuracy_delta_vs_baseline": accuracy_score(all_true, all_pred) - baseline_accuracy,
                "macro_f1_delta_vs_baseline": macro_f1 - b_macro_f1,
                "weighted_f1_delta_vs_baseline": weighted_f1 - b_weighted_f1,
                "beats_majority_baseline": accuracy_score(all_true, all_pred) > baseline_accuracy and macro_f1 > b_macro_f1
            },
            "misclassified_count": len(mistakes)
        }
        result["artifacts"] = self.__save_results(result, mistakes)
        (self.__results_root / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        
        return result
    
    def __save_results(self, result: dict, mistakes: list[dict]) -> dict:
        # Every run overwrites previous results so storage/test_results/ always
        # reflects the latest evaluation. Misclassified images are re-copied.
        self.__results_root.mkdir(parents=True, exist_ok=True)
        misclassified_dir = self.__results_root / "misclassified"
        if misclassified_dir.exists():
            shutil.rmtree(misclassified_dir)
        misclassified_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        
        # Misclassified images are renamed as 0001_true_as_pred.png so the file
        # name itself documents the mistake; this makes the report's error
        # analysis section easier to write and verify.
        for index, item in enumerate(mistakes, start=1):
            src = Path(item["source_path"])
            dst = misclassified_dir / f"{index:04d}_{item['true_label']}_as_{item['pred_label']}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            copied.append({**item, "copied_path": str(dst)})
        
        metrics_path = self.__results_root / "metrics.json"
        misclassified_path = self.__results_root / "misclassified.json"
        error_analysis_path = self.__results_root / "error_analysis.json"
        confusion_matrix_path = self.__results_root / "confusion_matrix.png"
        metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        misclassified_path.write_text(json.dumps(copied, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        error_analysis_path.write_text(json.dumps(result["error_analysis"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.__save_confusion_matrix(result["confusion_matrix"], result["class_names"], confusion_matrix_path)
        
        return {
            "results_root": str(self.__results_root),
            "metrics_path": str(metrics_path),
            "misclassified_path": str(misclassified_path),
            "error_analysis_path": str(error_analysis_path),
            "confusion_matrix_path": str(confusion_matrix_path),
            "misclassified_dir": str(misclassified_dir)
        }
    
    @staticmethod
    def __save_confusion_matrix(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
        # Rendered with matplotlib's non-interactive Agg backend (set in Docker),
        # so the figure can be produced without a display server.
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Confusion Matrix")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
    
    @staticmethod
    def __read_json(path: Path, default: dict | None = None) -> dict:
        if not path.exists() or path.stat().st_size == 0:
            if default is not None:
                return default
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))
