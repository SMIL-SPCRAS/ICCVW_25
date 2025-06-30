import mlflow
import os


class MLflowLogger:
    def __init__(self, project_name: str, run_name: str, config: dict[str, any], 
                 artifact_dir: str = "artifacts", tracking_uri: str = "mlruns"):
        """Initializes MLflow tracking."""
        self.active = True
        self.run_name = run_name

        mlflow.set_tracking_uri("file://" + os.path.abspath(tracking_uri))
        mlflow.set_experiment(project_name)

        if mlflow.active_run():
            mlflow.end_run()
        
        self.run = mlflow.start_run(run_name=run_name)

        for key, value in config.items():
            mlflow.log_param(key, value)

        if os.path.isdir(artifact_dir):
            mlflow.log_artifacts(artifact_dir, artifact_path="code")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self.active:
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, filepath: str, artifact_path: str = None) -> None:
        if not self.active:
            return
        
        if os.path.exists(filepath):
            mlflow.log_artifact(filepath, artifact_path)

    def end(self) -> None:
        if self.active:
            mlflow.end_run()
            self.active = False
