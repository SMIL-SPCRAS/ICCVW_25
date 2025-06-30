from collections import defaultdict


class LossManager:
    """Manages loss tracking across multiple tasks."""
    def __init__(self) -> None:
        self.task_sums = defaultdict(float)
        self.task_counts = defaultdict(int)
        self.reset()

    def update(self, task_losses: dict[str, float], batch_size: int) -> None:
        """Update loss sums for each task."""
        for task, value in task_losses.items():
            self.task_sums[task] += value * batch_size
            self.task_counts[task] += batch_size

    def compute(self) -> dict[str, float]:
        """Compute average loss for each tracked task."""
        return {
            task: self.task_sums[task] / self.task_counts[task]
            for task in self.task_sums
            if self.task_counts[task] > 0
        }

    def reset(self) -> None:
        """Reset all tracking values."""
        self.task_sums.clear()
        self.task_counts.clear()

    def track_task(self, task: str) -> None:
        """Initialize tracking for a specific task."""
        if task not in self.task_sums:
            self.task_sums[task] = 0.0
            self.task_counts[task] = 0
