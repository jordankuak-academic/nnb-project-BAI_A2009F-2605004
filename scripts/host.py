from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.http.routes import router, configure
from app.models.jordan_cnn import JordanCNN
from app.services.evaluator import Evaluator
from app.services.predictor import Predictor


def build_app() -> FastAPI:
    root = Path(__file__).resolve().parents[1]
    resources = root / "resources"
    release_dir = root / "storage" / "models" / "release"
    results_root = root / "storage" / "test_results"
    
    app = FastAPI(title="JordanCNN Dashboard")
    # CORS is fully open because the dashboard is a local development tool
    # served from the same host; tighten this if exposed publicly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    def evaluator_factory():
        # The Evaluator receives a model_builder instead of a pre-built model:
        # it rebuilds the architecture from the checkpoint's model_config
        # before loading state_dict, so num_classes and dropout always match.
        def model_builder(model_config: dict):
            return JordanCNN(
                num_classes=model_config["num_classes"],
                dropout=model_config.get("dropout", 0.5),
            )
        
        return Evaluator(
            model_builder=model_builder,
            release_dir=release_dir,
            results_root=results_root,
        )
    
    def predictor_factory():
        # Predictor loads release/model.pt lazily on first use and caches
        # the model in memory for subsequent predictions.
        return Predictor()
    
    # Factories are injected so routes.py stays free of model/service imports
    # and can be unit-tested with stubs.
    configure(evaluator_factory, predictor_factory)
    app.include_router(router)
    
    # StaticFiles is mounted last so /api/* routes take priority over
    # the catch-all static mount.
    if resources.exists():
        app.mount("/", StaticFiles(directory=str(resources), html=True), name="resources")
    
    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run("scripts.host:app", host="127.0.0.1", port=8000, reload=False)
