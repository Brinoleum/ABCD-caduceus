import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def probabilities_to_classes(probabilities):
    """
    Use the cumulative probabilities to compute class probabilities
    and take argmax. This can be more robust in some cases.
    """
    # Convert cumulative probabilities to individual class probabilities
    batch_size, num_thresholds = probabilities.shape
    num_classes = num_thresholds + 1
    
    # Initialize class probabilities
    class_probs = np.zeros((batch_size, num_classes))
    
    for i in range(batch_size):
        cumulative = probabilities[i]
        
        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[i, 0] = 1 - cumulative[0]
        
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k = 1, ..., num_classes-2
        for k in range(1, num_classes - 1):
            class_probs[i, k] = cumulative[k-1] - cumulative[k]
        
        # P(Y = num_classes-1) = P(Y > num_classes-2)
        class_probs[i, -1] = cumulative[-1]
    
    return np.argmax(class_probs, axis=1)

def compute_ordinal_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        # If model returns multiple outputs, take the probabilities
        # This assumes your model returns (loss, logits, probabilities) during eval
        probabilities = predictions[-1]  # Take the last element (probabilities)
    else:
        probabilities = predictions
    
    # Convert torch tensors to numpy if needed
    if hasattr(probabilities, 'cpu'):
        probabilities = probabilities.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # Ensure probabilities are in the right format
    if len(probabilities.shape) == 1:
        # If 1D, reshape for single sample
        probabilities = probabilities.reshape(1, -1)
    
    # Convert probabilities to predicted classes
    predicted_classes = probabilities_to_classes(probabilities)
    
    # Compute ordinal-specific metrics
    mae = mean_absolute_error(labels, predicted_classes)
    mse = mean_squared_error(labels, predicted_classes)
    rmse = np.sqrt(mse)
    
    # Exact accuracy
    exact_accuracy = np.mean(predicted_classes == labels)
    
    # Within-k accuracy (useful for ordinal tasks)
    within_1_accuracy = np.mean(np.abs(predicted_classes - labels) <= 1)
    within_2_accuracy = np.mean(np.abs(predicted_classes - labels) <= 2)
    
    # Kendall's Tau (rank correlation)
    from scipy.stats import kendalltau
    tau, _ = kendalltau(labels, predicted_classes)
    
    # Spearman correlation
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(labels, predicted_classes)
    
    # Class-wise accuracy
    class_accuracies = {}
    for class_idx in range(5):  # Assuming 5 classes (0-4)
        mask = labels == class_idx
        if np.sum(mask) > 0:
            class_acc = np.mean(predicted_classes[mask] == class_idx)
            class_accuracies[f'class_{class_idx}_accuracy'] = class_acc
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'exact_accuracy': exact_accuracy,
        'within_1_accuracy': within_1_accuracy,
        'within_2_accuracy': within_2_accuracy,
        'kendall_tau': tau,
        'spearman_correlation': spearman_corr,
    }
    
    # Add class-wise accuracies
    metrics.update(class_accuracies)
    
    return metrics