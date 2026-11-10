import json
import tempfile
import traceback
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from torch.utils.data import DataLoader

from app.services.dataset import ManifestImageDataset
from app.toolkits.logger import Logger
from configs.settings import CONFIGS

router = APIRouter()
TESTING = CONFIGS["TRAINING"]

_evaluator_factory = None
_predictor_factory = None


def configure(evaluator_factory, predictor_factory):
    global _evaluator_factory, _predictor_factory
    _evaluator_factory = evaluator_factory
    _predictor_factory = predictor_factory


def _build_test_loader() -> tuple[DataLoader, list[str], Path, dict, dict]:
    root = Path(__file__).resolve().parents[2]
    dataset_root = root / "datasets" / "images"
    manifest = json.loads((dataset_root / "split.json").read_text(encoding="utf-8-sig"))
    stats = json.loads((dataset_root / "split_stat.json").read_text(encoding="utf-8-sig"))
    class_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
    test_samples = [sample for sample in manifest["samples"] if sample["split"] == "testing"]
    if not test_samples:
        raise ValueError("Testing samples must exist in split.json.")
    test_loader = DataLoader(
        ManifestImageDataset(dataset_root, test_samples, include_paths=True),
        batch_size=int(TESTING["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(TESTING["NUM_WORKERS"]),
        pin_memory=False,
    )
    return test_loader, class_names, dataset_root, manifest, stats


@router.post("/api/evaluate")
def evaluate():
    if _evaluator_factory is None:
        raise HTTPException(500, "Evaluator is not configured.")
    try:
        evaluator = _evaluator_factory()
        test_loader, class_names, dataset_root, manifest, stats = _build_test_loader()
        logger = Logger()
        run_id = logger.start_run(
            operation_type="testing",
            model_name="JordanCNN",
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
            notes="Dashboard testing pipeline",
        )
        try:
            result = evaluator.evaluate(test_loader=test_loader, class_names=class_names)
        except Exception as exc:
            logger.complete_run(run_id, status="failed", summary={"error": str(exc)}, error_analysis={"exception_type": type(exc).__name__, "message": str(exc)})
            raise
        logger.complete_run(run_id, status="completed", accuracy=result["accuracy"], precision=result["macro_precision"], recall=result["macro_recall"], f1=result["macro_f1"], confusion_matrix=result["confusion_matrix"], summary={"sample_count": result["sample_count"], "top_3_accuracy": result["top_3_accuracy"], "weighted_f1": result["weighted_f1"], "total_parameters": result["total_parameters"]}, artifacts=result["artifacts"], error_analysis={"misclassified_count": result["misclassified_count"], "top_5_cases": result["error_analysis"]}, baseline_comparison={"baseline": result["baseline"], "task1_comparison": result["task1_comparison"]})
        result["run_id"] = run_id
        return JSONResponse(result)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@router.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if _predictor_factory is None:
        raise HTTPException(500, "Predictor is not configured.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Uploaded file must be an image.")
    try:
        suffix = Path(file.filename or "upload.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            predictor = _predictor_factory()
            result = predictor.predict(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return JSONResponse(result)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc