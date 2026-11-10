CONFIGS = {
    "PREPROCESSOR": {
        "DEFAULT_SEED": 42,
        "IMAGE_SIZE": (128, 128),
        "SPLIT_RATIOS": {
            "testing": 0.15,
            "training": 0.70,
            "validating": 0.15,
        },
        "EXTENSIONS": {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
        },
    },
    "TRAINING": {
        "SEED": 42,
        "BATCH_SIZE": 32,
        "EPOCHS": 30,
        "LEARNING_RATE": 0.0002,
        "WEIGHT_DECAY": 0.0,
        "PATIENCE": 5,
        "MIN_DELTA": 0.0,
        "DROPOUT": 0.3,
        "NUM_WORKERS": 0,
        "DEVICE": None,
    },
}
