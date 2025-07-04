import os
import torch
import numpy as np
from sklearn.metrics import recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
from models.models import JinaMultiLabelClassifier

class Trainer:
    def __init__(self, model, optimizer, criterion, config, tokenizer, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = os.path.join(config.logging.save_path, config.model.name.replace("/", "_"))
        os.makedirs(self.save_path, exist_ok=True)
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.save_path, "train"))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.save_path, "val"))
        self.tokenizer = tokenizer

        self.best_score = 0
        self.best_epoch = 0
        self.patience_counter = 0

    def train(self):
        for epoch in range(self.config.training.epochs):
            print(f"Epoch {epoch}")
            train_loss = self._train_one_epoch(epoch)
            val_metrics = self._val_one_epoch(epoch)

            # Early stopping + checkpoint
            if val_metrics["mean_men"] > self.best_score:
                self.best_score = val_metrics["mean_men"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.training.patience:
                print("Early stopping triggered.")
                break

    def _train_one_epoch(self, epoch):
        self.model.train()
        total_loss, total_size = 0.0, 0

        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_size += labels.size(0)

        avg_loss = total_loss / total_size
        self.train_writer.add_scalar("loss", avg_loss, epoch)
        return avg_loss

    def _val_one_epoch(self, epoch):
        # Initialize dataset-wise containers
        dataset_probs = {}
        dataset_targets = {}
        self.model.eval()
        total_loss, total_size = 0.0, 0
        # Collect predictions
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                ds_labels = batch["ds_labels"]  # list of strings

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                true = labels.cpu().numpy()

                for i, name in enumerate(ds_labels):
                    if name not in dataset_probs:
                        dataset_probs[name] = []
                        dataset_targets[name] = []
                    dataset_probs[name].append(probs[i])
                    dataset_targets[name].append(true[i])

                total_loss += loss.item() * labels.size(0)
                total_size += labels.size(0)

        avg_loss = total_loss / total_size
        metrics = {}
        men_scores = []

        # Compute metrics for each dataset
        for name in dataset_probs:
            preds = np.argmax(dataset_probs[name], axis=1)
            trues = np.array(dataset_targets[name])

            uar = recall_score(trues, preds, average='macro', zero_division=0)
            war = recall_score(trues, preds, average='weighted', zero_division=0)
            mf1 = f1_score(trues, preds, average='macro', zero_division=0)
            wf1 = f1_score(trues, preds, average='weighted', zero_division=0)
            men = np.mean([uar, war, mf1, wf1])

            # Log to TensorBoard
            self.val_writer.add_scalar(f"{name}/uar", uar, epoch)
            self.val_writer.add_scalar(f"{name}/war", war, epoch)
            self.val_writer.add_scalar(f"{name}/mf1", mf1, epoch)
            self.val_writer.add_scalar(f"{name}/wf1", wf1, epoch)
            self.val_writer.add_scalar(f"{name}/men", men, epoch)

            # Store for final output
            metrics[f"{name}_uar"] = uar
            metrics[f"{name}_war"] = war
            metrics[f"{name}_mf1"] = mf1
            metrics[f"{name}_wf1"] = wf1
            metrics[f"{name}_men"] = men
            men_scores.append(men)

        mean_men = np.mean(men_scores)
        self.val_writer.add_scalar("mean_men", mean_men, epoch)
        self.val_writer.add_scalar("loss", avg_loss, epoch)

        metrics["mean_men"] = mean_men
        metrics["avg_loss"] = avg_loss
        
        print(f"\nValidation metrics for epoch {epoch}:")
        print("=" * 40)
        print(f"{'Dataset':<12} | {'UAR':>6} | {'WAR':>6} | {'MF1':>6} | {'WF1':>6} | {'MEN':>6}")
        print("-" * 40)
        for name in dataset_probs:
            print(f"{name:<12} | "
                f"{metrics[f'{name}_uar']:.4f} | "
                f"{metrics[f'{name}_war']:.4f} | "
                f"{metrics[f'{name}_mf1']:.4f} | "
                f"{metrics[f'{name}_wf1']:.4f} | "
                f"{metrics[f'{name}_men']:.4f}")
        print("-" * 40)
        print(f"{'Mean MEN':<12} | "
            f"{'':>6} | {'':>6} | {'':>6} | {'':>6} | "
            f"{mean_men:.4f}")
        print(f"Avg Loss: {avg_loss:.6f}\n")

        return metrics


    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.save_path, f"best_model_epoch{epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        self.tokenizer.save_pretrained(os.path.join(self.save_path, "tokenizer"))
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save extra metadata in a .pkl
        state_dict_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        save_dict = {
            "model_name": self.config.model.name,
            "state_dict": state_dict_cpu,
            "num_labels": self.config.model.num_classes
        }

        # Handle optional metadata
        if "jina" in self.config.model.name.lower() or isinstance(self.model, JinaMultiLabelClassifier):
            save_dict["is_jina"] = True
            save_dict["jina_model_name"] = self.config.model.name
            save_dict["embed_dim"] = getattr(self.model, "embed_dim", None)  # Optional
            save_dict["lora_task"] = "classification"
        else:
            save_dict["is_jina"] = False

        pickle_path = os.path.join(self.save_path, "best_model.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(save_dict, f)

        print(f"Pickle metadata saved to {pickle_path}")