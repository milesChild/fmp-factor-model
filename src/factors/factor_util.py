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
    Winsorize the values to the specified percentiles.
    
    params:
        values (np.ndarray): Array of values to winsorize
        lower (float): Lower percentile (1-100)
        upper (float): Upper percentile (1-100)
    
    returns:
        np.ndarray: Winsorized values
    """
    # Convert percentiles to actual values
    lower_value = np.percentile(values, lower)
    upper_value = np.percentile(values, upper)
    
    # Clip values to the percentiles
    values = np.clip(values, lower_value, upper_value)
    return values

def z_score(values: np.ndarray) -> np.ndarray:
    """
    Z-score the values. If all values are identical, returns an array of zeros.
    """
    std = np.std(values)
    if std == 0:  # edge case when every value is identical
        return np.zeros_like(values)
    return (values - np.mean(values)) / std

def normalize(values: np.ndarray) -> np.ndarray:
    """
    Normalize the values to be between 0 and 1.
    """
    return (values - np.min(values)) / (np.max(values) - np.min(values))
