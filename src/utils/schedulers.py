import torch
import logging
from transformers import get_scheduler

class DummyScheduler:
    def step(self, *args, **kwargs):
        pass

class SmartScheduler:
    def __init__(self, scheduler_type, optimizer, config, steps_per_epoch):
        self.scheduler_type = scheduler_type.lower()
        self.is_batch_level = False

        if self.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
            logging.info("[Scheduler] Используется ReduceLROnPlateau (по метрике).")

        elif self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=1e-6
            )
            logging.info("[Scheduler] Используется CosineAnnealingLR.")

        elif self.scheduler_type == "onecycle":
            if steps_per_epoch == 0:
                raise ValueError("train_loader пустой, OneCycle не может работать без данных.")
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.lr,
                steps_per_epoch=steps_per_epoch,
                epochs=config.num_epochs
            )
            self.is_batch_level = True
            logging.info(f"[Scheduler] Используется OneCycleLR ({steps_per_epoch} шагов на эпоху).")

        elif self.scheduler_type.startswith("huggingface_"):
            scheduler_name = self.scheduler_type.replace("huggingface_", "")

            total_steps = steps_per_epoch * config.num_epochs
            warmup_steps = int(total_steps * config.warmup_ratio)

            self.scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
    )
            self.is_batch_level = True  # HuggingFace обычно требует шагать по батчам
            logging.info(f"[Scheduler] HuggingFace: {scheduler_name} — warmup_steps={warmup_steps}, total_steps={total_steps}")

        elif self.scheduler_type == "none":
            self.scheduler = DummyScheduler()
            logging.info("[Scheduler] Нет шедулера (ручное управление lr).")

        else:
            raise ValueError(f"Неизвестный scheduler_type: {scheduler_type}")

    def step(self, metric=None, batch_level=False):
        """
        batch_level=True  ➔ шагать после батча (например, для OneCycle, HuggingFace schedulers)
        batch_level=False ➔ шагать после эпохи
        """
        if isinstance(self.scheduler, DummyScheduler):
            return

        if self.scheduler_type == "plateau":
            if not batch_level:
                self.scheduler.step(metric)
        elif self.is_batch_level:
            if batch_level:
                self.scheduler.step()
        else:
            if not batch_level:
                self.scheduler.step()
