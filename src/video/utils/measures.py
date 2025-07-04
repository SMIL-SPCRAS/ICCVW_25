from sklearn.metrics import recall_score, f1_score

def uar(y_true, y_pred):
    """
    Calculate UAR metric (Unweighted Average Recall).
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: UAR (Recall across all classes without weighting)
    """
    return recall_score(y_true, y_pred, average='macro', zero_division=0)

def war(y_true, y_pred):
    """
    Calculate WAR metric (Weighted Average Recall).
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: WAR (Recall with class weighting)
    """
    return recall_score(y_true, y_pred, average='weighted', zero_division=0)

def mf1(y_true, y_pred):
    """
    Calculate MF1 metric (Macro F1 Score).
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: MF1 (F1 averaged across all classes)
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def wf1(y_true, y_pred):
    """
    Calculate WF1 metric (Weighted F1 Score).
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: WF1 (F1 with class weighting)
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=0)