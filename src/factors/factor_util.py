import numpy as np

def log_transform(values: np.ndarray) -> np.ndarray:
    """
    Log transform the values.
    """
    min_value = np.min(values)
    if min_value <= 0:
        values = values - min_value + 1  # Shift to make all values positive
    values = np.log(values)
    return values

def winsorize(values: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    """
    Winsorize the values.
    """
    values = np.clip(values, lower, upper)
    return values

def z_score(values: np.ndarray) -> np.ndarray:
    """
    Z-score the values.
    """
    return (values - np.mean(values)) / np.std(values)

def normalize(values: np.ndarray) -> np.ndarray:
    """
    Normalize the values to be between 0 and 1.
    """
    return (values - np.min(values)) / (np.max(values) - np.min(values))
