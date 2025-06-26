class MetricManager:
    """Wrapper for evaluating multiple metrics."""
    def __init__(self, metrics: list[any]) -> None:
        self.metrics = metrics

    def calculate_all(self, targets: dict[str, any], predicts: dict[str, any]) -> dict[str, float]:
        """Calculate all metrics for corresponding tasks."""
        results = {}
        for metric in self.metrics:
            task = metric.task
            results[str(metric)] = metric.calc(targets[task], predicts[task])
        
        return results
