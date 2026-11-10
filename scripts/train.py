import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from app.models.jordan_cnn import JordanCNN
from app.services.dataset import ManifestImageDataset
from app.services.trainer import Trainer
from app.toolkits.logger import Logger as ExperimentLogger
from configs.settings import CONFIGS


TRAINING = CONFIGS["TRAINING"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "datasets" / "images"
    
    seed = int(TRAINING["SEED"])
    device = str(torch.device(TRAINING["DEVICE"])) if TRAINING["DEVICE"] else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix every random source used during training so two runs with the same
    # config produce the same weights, batch order and dropout masks.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # split.json is the single source of truth for sample paths, labels and splits.
    # split_stat.json supplies dataset totals for the experiment log.
    manifest = json.loads((dataset_root / "split.json").read_text(encoding="utf-8-sig"))
    stats = json.loads((dataset_root / "split_stat.json").read_text(encoding="utf-8-sig"))
    
    class_to_idx = manifest["class_to_idx"]
    class_names = [label for label, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    train_samples = [s for s in manifest["samples"] if s["split"] == "training"]
    validation_samples = [s for s in manifest["samples"] if s["split"] == "validating"]
    if not train_samples or not validation_samples:
        raise ValueError("Training and validation samples must both exist in split.json.")
    
    batch_size = int(TRAINING["BATCH_SIZE"])
    num_workers = int(TRAINING["NUM_WORKERS"])
    pin_memory = device.startswith("cuda")
    
    # The generator is seeded so DataLoader(shuffle=True) produces the same
    # batch order across runs, which is part of reproducibility.
    generator = torch.Generator().manual_seed(seed)
    
    train_loader = DataLoader(
        ManifestImageDataset(dataset_root, train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    # Validation is never shuffled: metrics must be comparable across epochs.
    validation_loader = DataLoader(
        ManifestImageDataset(dataset_root, validation_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    num_classes = len(class_names)
    dropout = float(TRAINING["DROPOUT"])
    model = JordanCNN(num_classes=num_classes, dropout=dropout)
    
    # model_config is stored in model.pt and reused by Predictor/Evaluator to
    # rebuild the exact same architecture before loading weights.
    trainer = Trainer(
        model=model,
        device=device,
        lr=float(TRAINING["LEARNING_RATE"]),
        weight_decay=float(TRAINING["WEIGHT_DECAY"]),
        patience=int(TRAINING["PATIENCE"]),
        min_delta=float(TRAINING["MIN_DELTA"]),
        model_config={"num_classes": num_classes, "dropout": dropout, "learning_rate": float(TRAINING["LEARNING_RATE"]), "batch_size": int(TRAINING["BATCH_SIZE"]), "epochs": int(TRAINING["EPOCHS"]), "min_delta": float(TRAINING["MIN_DELTA"])},
        
        scaler_state={"transform": "rgb_to_tensor_div_255", "image_size": manifest["image_size"]},
    )
    
    # The experiment logger writes one row per run to storage/logs/experiments.csv
    # so all experiments can be compared side by side in the report.
    logger = ExperimentLogger()
    run_id = logger.start_run(
        operation_type="training",
        model_name=model.__class__.__name__,
        device=device,
        seed=seed,
        dataset_manifest=str(dataset_root / "split.json"),
        dataset_samples=int(stats["totals"]["all"]),
        dataset_features=int(manifest["image_size"][0] * manifest["image_size"][1] * 3),
        dataset_classes=num_classes,
        train_split=float(manifest["ratios"]["training"]),
        val_split=float(manifest["ratios"]["validating"]),
        test_split=float(manifest["ratios"]["testing"]),
        architecture="conv[3,32,64,128,256]-fc[128,num_classes]",
        hidden_layers=5,
        total_parameters=sum(p.numel() for p in model.parameters()),
        activation="ReLU",
        output_activation="raw_logits",
        loss_function="CrossEntropyLoss",
        optimizer="Adam",
        learning_rate=float(TRAINING["LEARNING_RATE"]),
        batch_size=batch_size,
        epochs=int(TRAINING["EPOCHS"]),
        config={**TRAINING, "class_names": class_names, "image_size": manifest["image_size"]},
        notes="Manifest-based JordanCNN training pipeline",
    )
    
    # If training crashes, the log row is still closed as "failed" with the
    # exception type and message, so the experiment log is never left dangling.
    try:
        result = trainer.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=int(TRAINING["EPOCHS"]),
        )
    except Exception as error:
        logger.complete_run(
            run_id,
            status="failed",
            summary={"error": str(error)},
            error_analysis={"exception_type": type(error).__name__, "message": str(error)},
        )
        raise
    
    # On success, only the best-epoch metrics are written to the CSV row.
    # The final metrics are printed as JSON so they can be copied into the report.
    logger.complete_run(
        run_id,
        status="completed",
        accuracy=result["best_validation_accuracy"],
        f1=result["best_validation_macro_f1"],
        best_validation_loss=result["best_validation_loss"],
        best_epoch=result["best_epoch"],
        summary={
            "class_names": class_names,
            "dataset_totals": stats["totals"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "best_validation_macro_f1": result["best_validation_macro_f1"],
        },
        artifacts=result["artifacts"],
    )
    print(json.dumps({
        "run_id": run_id,
        "best_epoch": result["best_epoch"],
        "best_validation_loss": round(result["best_validation_loss"], 4),
        "best_validation_accuracy": round(result["best_validation_accuracy"], 4),
        "best_validation_macro_f1": round(result["best_validation_macro_f1"], 4),
        "artifacts": result["artifacts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
