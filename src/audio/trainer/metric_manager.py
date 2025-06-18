from typing import Dict, List, Any


class MetricManager:
    """
    Wrapper for evaluating multiple metrics.

    Args:
        metrics (List[Any]): List of metric instances.
    """
    def __init__(self, metrics: List[Any]):
        self.metrics = metrics

    def calculate_all(self, targets: Dict[str, Any], predicts: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate all metrics for corresponding tasks.

        Args:
            targets (Dict[str, Any]): Ground truth labels.
            predicts (Dict[str, Any]): Model predicts.

        Returns:
            Dict[str, float]: Computed metrics.
        """
        results = {}
        for metric in self.metrics:
            task = metric.task
            results[str(metric)] = metric.calc(targets[task], predicts[task])
        
        return results
