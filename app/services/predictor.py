import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from app.models.jordan_cnn import JordanCNN


class Predictor:
    def __init__(self, release_dir: Path | None = None, device: str | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.__dataset_root = root / "datasets" / "images"
        self.__release_dir = Path(release_dir) if release_dir else root / "storage" / "models" / "release"
        self.__device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Load once at construction time so each predict() call only runs inference.
        self.__model, self.__class_names, self.__image_size = self.__load()
    
    def __load(self):
        # The release directory is the single source of truth for inference:
        # it contains the trained weights (model.pt) and the metadata written by Trainer.
        metadata_path = self.__release_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Release metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        manifest = self.__load_manifest_defaults()
        
        # Rebuild the exact architecture before loading weights: model_config
        # records num_classes and dropout, which must match the checkpoint.
        checkpoint = torch.load(self.__release_dir / "model.pt", map_location="cpu", weights_only=False)
        checkpoint_model_config = dict(checkpoint.get("model_config") or {})
        model = JordanCNN(
            num_classes=int(checkpoint_model_config.get("num_classes", len(manifest["class_names"]))),
            dropout=float(checkpoint_model_config.get("dropout", 0.5)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.__device).eval()
        # .eval() disables dropout and switches BatchNorm to inference mode.
        
        # Class names and image size are resolved with a three-level fallback:
        # metadata → checkpoint model_config → split.json manifest → hard-coded defaults.
        # This keeps the predictor working even if an older checkpoint lacks these fields.
        class_names = metadata.get("model_config", {}).get("class_names") or checkpoint_model_config.get("class_names") or manifest["class_names"]
        image_size = checkpoint_model_config.get("image_size") or metadata.get("image_size") or manifest["image_size"]
        return model, class_names, tuple(image_size)
    
    def __load_manifest_defaults(self) -> dict:
        # split.json records class_to_idx and image_size produced by the Preprocessor.
        # It is only used as a fallback when the checkpoint metadata is incomplete.
        manifest_path = self.__dataset_root / "split.json"
        if not manifest_path.exists():
            return {"class_names": ["cardboard", "glass", "metal", "paper", "plastic", "trash"], "image_size": [128, 128]}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        class_names = [label for label, _ in sorted(manifest["class_to_idx"].items(), key=lambda item: item[1])]
        return {"class_names": class_names, "image_size": manifest.get("image_size", [128, 128])}
    
    def predict(self, image_path: Path, top_k: int = 3) -> dict:
        # Preprocessing must match the Preprocessor exactly: RGB, aspect-preserving
        # pad to the model's image_size with black borders, then scale to [0, 1].
        # Otherwise the model sees a different input distribution at inference time.
        with Image.open(image_path) as image:
            image = ImageOps.pad(image.convert("RGB"), self.__image_size, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
            array = np.asarray(image, dtype=np.float32) / 255.0
        # HWC → CHW and add a batch dimension: (1, 3, H, W) as expected by the CNN.
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.__device)
        
        # no_grad() skips gradient tracking during inference, saving memory and time.
        with torch.no_grad():
            logits = self.__model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # top_k is clamped to [1, num_classes] so an invalid caller value cannot crash.
        top = probs.topk(max(1, min(top_k, len(self.__class_names))))
        results = [
            {"label": self.__class_names[idx], "confidence": float(score)}
            for idx, score in zip(top.indices.tolist(), top.values.tolist())
        ]
        return {
            "label": results[0]["label"],
            "confidence": results[0]["confidence"],
            "top_3": results
        }