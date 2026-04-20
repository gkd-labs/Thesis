def compute_convergence_training_efficiency(training_time, validation_mae):
    """
    Computes convergence efficiency as:
        training_time_at_patience_or_best_val / best_validation_MAE

    Args:
        training_time (float): Total minutes until patience trigger or best epoch.
        validation_mae (float): Best validation MAE achieved.

    Returns:
        float: Convergence efficiency score (lower is better).
    """
    if validation_mae <= 0:
        raise ValueError("validation_mae must be > 0")

    return training_time / validation_mae
