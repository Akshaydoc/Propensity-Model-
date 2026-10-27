"""MLflow helper: local sqlite backend (no server needed). If MLFLOW_TRACKING_URI is set (docker-compose) that is used."""
import logging
import os
log = logging.getLogger(__name__)


def get_mlflow(cfg):
    try:
        import mlflow
        from .config import ROOT
        uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + (ROOT / cfg["mlflow"]["tracking_db"]).as_posix()
        mlflow.set_tracking_uri(uri)
        name = cfg["mlflow"]["experiment"]
        if mlflow.get_experiment_by_name(name) is None:
            kw = {} if os.environ.get("MLFLOW_TRACKING_URI") else {"artifact_location": (ROOT / "mlartifacts").as_uri()}
            mlflow.create_experiment(name, **kw)
        mlflow.set_experiment(name)
        return mlflow
    except Exception as e:  # tracking must never break the pipeline
        log.warning("MLflow disabled: %s", e)
        return None


def log_model_and_register(mlflow, model, name):
    try:
        try:
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        except TypeError:  # mlflow 2.x
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")
        run_id = mlflow.active_run().info.run_id
        mv = mlflow.register_model(f"runs:/{run_id}/model", name)
        mlflow.tracking.MlflowClient().set_registered_model_alias(name, "champion", mv.version)
        return mv.version
    except Exception as e:
        log.warning("model registry step skipped: %s", e)
        return None
