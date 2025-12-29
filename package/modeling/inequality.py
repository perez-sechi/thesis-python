import numpy as np


def gini_coefficient(values):
    """
    Calculate the Gini coefficient for a list of values.
    
    Args:
        values: array-like of numeric values
    
    Returns:
        float: Gini coefficient between 0 and 1
    """
    # Convert to numpy array and sort
    sorted_values = np.sort(np.array(values))
    n = len(sorted_values)
    
    # Calculate cumulative sum
    cumsum = np.cumsum(sorted_values)
    
    # Calculate Gini coefficient
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    return gini