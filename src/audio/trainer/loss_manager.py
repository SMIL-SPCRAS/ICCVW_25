from typing import Dict


class LossManager:
    """
    Manages loss tracking across multiple tasks.

    Attributes:
        task_sums (Dict[str, float]): Sum of losses per task.
        task_counts (Dict[str, int]): Count of samples per task.
    """
    def __init__(self):
        self.reset()

    def update(self, task_losses: Dict[str, float], batch_size: int):
        """
        Update loss sums for each task.

        Args:
            task_losses (Dict[str, float]): Loss values per task.
            batch_size (int): Size of the batch.
        """
        for task, value in task_losses.items():
            self.task_sums[task] += value * batch_size
            self.task_counts[task] += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute average loss for each tracked task.

        Returns:
            Dict[str, float]: Averaged loss per task.
        """
        return {
            task: self.task_sums[task] / self.task_counts[task]
            for task in self.task_sums
            if self.task_counts[task] > 0
        }

    def reset(self):
        """Reset all tracking values."""
        self.task_sums = {}
        self.task_counts = {}

    def track_task(self, task: str):
        """
        Initialize tracking for a specific task.

        Args:
            task (str): Task name.
        """
        if task not in self.task_sums:
            self.task_sums[task] = 0.0
            self.task_counts[task] = 0
