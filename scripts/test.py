import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from app.models.jordan_cnn import JordanCNN
from app.services.dataset import ManifestImageDataset
from app.services.evaluator import Evaluator
from app.toolkits.logger import Logger
from configs.settings import CONFIGS


TESTING = CONFIGS["TRAINING"]


def build_model(model_config: dict) -> JordanCNN:
    return JordanCNN(
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config.get("dropout", TESTING["DROPOUT"])),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "datasets" / "images"
    seed = int(TESTING["SEED"])
    device = str(torch.device(TESTING["DEVICE"])) if TESTING["DEVICE"] else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Seeds are fixed here too so that any stochastic layer or data ordering
    # behaves identically to training when the same split is evaluated.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    manifest = json.loads((dataset_root / "split.json").read_text(encoding="utf-8-sig"))
    stats = json.loads((dataset_root / "split_stat.json").read_text(encoding="utf-8-sig"))
    class_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    test_samples = [sample for sample in manifest["samples"] if sample["split"] == "testing"]
    if not test_samples:
        raise ValueError("Testing samples must exist in split.json.")
    
    # include_paths=True makes the dataset yield (image, label, path) triples,
    # which the Evaluator needs to copy misclassified images for error analysis.
    # shuffle=False keeps predictions in a fixed order for reproducibility.
    test_loader = DataLoader(
        ManifestImageDataset(dataset_root, test_samples, include_paths=True),
        batch_size=int(TESTING["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(TESTING["NUM_WORKERS"]),
        pin_memory=device.startswith("cuda"),
    )
    
    # The Evaluator loads release/model.pt, rebuilds the model via build_model,
    # and writes metrics.json, confusion_matrix.png and misclassified images
    # to storage/test_results/.
    evaluator = Evaluator(
        model_builder=build_model,
        device=device,
        results_root=root / "storage" / "test_results",
    )
    logger = Logger()
    run_id = logger.start_run(
        operation_type="testing",
        model_name="JordanCNN",
        device=device,
        seed=seed,
        dataset_manifest=str(dataset_root / "split.json"),
        dataset_samples=int(stats["totals"]["testing"]),
        dataset_features=int(manifest["image_size"][0] * manifest["image_size"][1] * 3),
        dataset_classes=len(class_names),
        train_split=float(manifest["ratios"]["training"]),
        val_split=float(manifest["ratios"]["validating"]),
        test_split=float(manifest["ratios"]["testing"]),
        architecture="conv[3,32,64,128,256]-fc[128,num_classes]",
        hidden_layers=5,
        activation="ReLU",
        output_activation="raw_logits",
        loss_function="CrossEntropyLoss",
        batch_size=int(TESTING["BATCH_SIZE"]),
        config={**TESTING, "class_names": class_names, "image_size": manifest["image_size"]},
        notes="Manifest-based JordanCNN testing pipeline",
    )
    
    # If evaluation fails, the log row is still closed as "failed" with the
    # exception details so the experiment log never contains a dangling run.
    try:
        result = evaluator.evaluate(test_loader=test_loader, class_names=class_names)
    except Exception as error:
        logger.complete_run(
            run_id,
            status="failed",
            summary={"error": str(error)},
            error_analysis={"exception_type": type(error).__name__, "message": str(error)},
        )
        raise error
    
    # The test row records the full metric set (macro + weighted + top-3 +
    # inference time), plus the confusion matrix, the five most confident
    # mistakes and the majority-class baseline comparison.
    logger.complete_run(
        run_id,
        status="completed",
        accuracy=result["accuracy"],
        precision=result["macro_precision"],
        recall=result["macro_recall"],
        f1=result["macro_f1"],
        confusion_matrix=result["confusion_matrix"],
        summary={
            "class_names": result["class_names"],
            "sample_count": result["sample_count"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "top_3_accuracy": result["top_3_accuracy"],
            "average_inference_time_ms": result["average_inference_time_ms"],
            "weighted_precision": result["weighted_precision"],
            "weighted_recall": result["weighted_recall"],
            "weighted_f1": result["weighted_f1"],
            "total_parameters": result["total_parameters"],
        },
        artifacts=result["artifacts"],
        error_analysis={
            "misclassified_count": result["misclassified_count"],
            "top_5_cases": result["error_analysis"],
        },
        baseline_comparison={
            "baseline": result["baseline"],
            "task1_comparison": result["task1_comparison"],
        },
    )
    
    # Printed as JSON so the numbers can be copied directly into the report
    # (Task 5) and compared against the training log in experiments.csv.
    print(json.dumps({
        "run_id": run_id,
        "sample_count": result["sample_count"],
        "accuracy": round(result["accuracy"], 4),
        "macro_precision": round(result["macro_precision"], 4),
        "macro_recall": round(result["macro_recall"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "weighted_f1": round(result["weighted_f1"], 4),
        "top_3_accuracy": round(result["top_3_accuracy"], 4),
        "average_inference_time_ms": round(result["average_inference_time_ms"], 4),
        "misclassified_count": result["misclassified_count"],
        "artifacts": result["artifacts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
