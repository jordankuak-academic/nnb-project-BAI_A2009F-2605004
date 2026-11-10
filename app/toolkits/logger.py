import csv
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class Logger:
    HEADERS = [
        "run_id", "operation_type", "model_name", "status", "started_at",
        "ended_at", "device", "seed", "dataset_manifest", "dataset_samples",
        "dataset_features", "dataset_classes", "train_split", "val_split", "test_split",
        "architecture", "hidden_layers", "total_parameters", "activation", "output_activation",
        "loss_function", "optimizer", "learning_rate", "batch_size", "epochs",
        "accuracy", "precision", "recall", "f1", "best_validation_loss",
        "best_epoch", "confusion_matrix", "notes", "config_json", "summary_json",
        "artifacts_json", "error_analysis_json", "baseline_comparison_json",
    ]

    def __init__(self, csv_path: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2] / "storage" / "logs"
        self.__csv_path = Path(csv_path) if csv_path is not None else root / "experiments.csv"
        self.__csv_path.parent.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        operation_type: str,
        model_name: str,
        *,
        device: str | None = None,
        seed: int | None = None,
        dataset_manifest: str | None = None,
        dataset_samples: int | None = None,
        dataset_features: int | None = None,
        dataset_classes: int | None = None,
        train_split: float | None = None,
        val_split: float | None = None,
        test_split: float | None = None,
        architecture: str | None = None,
        hidden_layers: int | None = None,
        total_parameters: int | None = None,
        activation: str | None = None,
        output_activation: str | None = None,
        loss_function: str | None = None,
        optimizer: str | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        config: dict | None = None,
        notes: str | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        started_at = self.__now()
        row = {
            "run_id": run_id,
            "operation_type": operation_type,
            "model_name": model_name,
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
            "device": device,
            "seed": seed,
            "dataset_manifest": dataset_manifest,
            "dataset_samples": dataset_samples,
            "dataset_features": dataset_features,
            "dataset_classes": dataset_classes,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "architecture": architecture,
            "hidden_layers": hidden_layers,
            "total_parameters": total_parameters,
            "activation": activation,
            "output_activation": output_activation,
            "loss_function": loss_function,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "best_validation_loss": None,
            "best_epoch": None,
            "confusion_matrix": None,
            "notes": notes,
            "config_json": self.__dump(config),
            "summary_json": None,
            "artifacts_json": None,
            "error_analysis_json": None,
            "baseline_comparison_json": None,
        }
        self.__append_row(row)
        return run_id

    def complete_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        accuracy: float | None = None,
        precision: float | None = None,
        recall: float | None = None,
        f1: float | None = None,
        best_validation_loss: float | None = None,
        best_epoch: int | None = None,
        confusion_matrix: list | None = None,
        summary: dict | None = None,
        artifacts: dict | None = None,
        error_analysis: dict | None = None,
        baseline_comparison: dict | None = None,
    ) -> None:
        ended_at = self.__now()
        updates = {
            "status": status,
            "ended_at": ended_at,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "confusion_matrix": self.__dump(confusion_matrix),
            "summary_json": self.__dump(summary),
            "artifacts_json": self.__dump(artifacts),
            "error_analysis_json": self.__dump(error_analysis),
            "baseline_comparison_json": self.__dump(baseline_comparison),
        }
        self.__update_row(run_id, updates)

    def __append_row(self, row: dict) -> None:
        file_exists = self.__csv_path.exists() and self.__csv_path.stat().st_size > 0
        with self.__csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k) for k in self.HEADERS})

    def __update_row(self, run_id: str, updates: dict) -> None:
        if not self.__csv_path.exists():
            raise ValueError(f"CSV file does not exist: {self.__csv_path}")

        with self.__csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        found = False
        for row in rows:
            if row["run_id"] == run_id:
                for key, value in updates.items():
                    if key in row:
                        row[key] = value
                found = True
                break

        if not found:
            raise ValueError(f"Unknown run_id: {run_id}")

        with self.__csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def __now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def __dump(value) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True)