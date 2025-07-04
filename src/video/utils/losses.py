import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        Initializes the class for cross-entropy loss with optional class weighting.

        :param class_weights: Vector of class weights (optional)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        """
        Computes cross-entropy loss with (or without) class weighting.

        :param y_true: Ground truth class labels (vector or single label)
        :param y_pred: Prediction probability vector
        :return: Loss value
        """
        y_true = y_true.to(torch.long)  # Convert labels to Long type
        y_pred = y_pred.to(torch.float32)  # Convert predictions to Float32 type

        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights).float().to(y_true.device)
            loss = F.cross_entropy(y_pred, y_true, weight=class_weights)
        else:
            loss = F.cross_entropy(y_pred, y_true)
        
        return loss
    
class WeightedBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        Initializes the class for binary cross-entropy loss with optional class weighting.

        :param class_weights: Vector of class weights (optional)
        """
        super(WeightedBinaryCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        """
        Computes binary cross-entropy loss with (or without) class weighting.

        :param y_true: Ground truth class labels (vector or single label)
        :param y_pred: Prediction probability vector
        :return: Loss value
        """
        y_pred = y_pred.to(torch.float32)  # Convert predictions to Float32 type

        if self.class_weights is not None:
            class_weights = torch.tensor(self.class_weights).float().to(y_true.device)
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=class_weights)
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        
        return loss
    
class SoftCrossEntropyLoss(nn.Module):
    """
    Soft Cross Entropy Loss supporting class weights and multi-task outputs.
    """
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, outputs, targets) -> torch.Tensor:
        probs = F.log_softmax(outputs, dim=-1)

        target_probs = targets / (targets.sum(dim=-1, keepdim=True) + 1e-6)

        loss = -(target_probs * probs)
        
        weights = torch.tensor(self.class_weights).float().to(outputs.device)
        loss = loss * weights.unsqueeze(0)

        loss = loss.sum(dim=-1).mean()

        return loss