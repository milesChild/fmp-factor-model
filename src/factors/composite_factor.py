from typing import List, Optional
import pandas as pd
import numpy as np
from src.factors.characteristic import Characteristic
from src.factors.factor_util import z_score

class CompositeFactor:
    """A composite factor made up of multiple characteristics."""
    
    def __init__(self, characteristics: List[Characteristic]) -> None:
        """
        Initialize composite factor with a list of characteristics.

        params:
            characteristics (List[Characteristic]): List of characteristics to combine
        """
        if not characteristics or len(characteristics) == 0:
            raise ValueError("Must provide at least one characteristic")
        
        self._validate_characteristics(characteristics)
        self.characteristics = characteristics
        self._loadings: Optional[pd.Series] = None
        
        # Validate/normalize weights
        self._normalize_weights()
    
    def _validate_characteristics(self, characteristics: List[Characteristic]) -> None:
        """
        Validate that all characteristics share the same date vector.
        """
        date_vector = characteristics[0].date_vector
        for char in characteristics[1:]:
            if len(char.date_vector) != len(date_vector) or any(d1 != d2 for d1, d2 in zip(char.date_vector, date_vector)):
                raise ValueError(f"Characteristic {str(char)} has mismatched date vector")
    
    def _normalize_weights(self) -> None:
        """
        Normalize weights across characteristics so that they sum to 1.0. 
        
        - If all weights are None: assign equal weights
        - If some weights are None: set None weights to 0.0 and normalize the rest
        - If all weights are provided: normalize them to sum to 1.0
        
        Mutates the characteristics in place.
        
        raises:
            ValueError: If any weights are negative or all weights are zero
        """
        # Get existing weights
        weights = [char.config.weight for char in self.characteristics]
        
        # If no weights provided, use equal weights
        if all(w is None for w in weights):
            equal_weight = 1.0 / len(self.characteristics)
            for char in self.characteristics:
                char.config.weight = equal_weight
            return
        
        # If some weights are None, set them to 0.0
        for char in self.characteristics:
            if char.config.weight is None:
                char.config.weight = 0.0
        
        # Get the updated weights
        weights = [char.config.weight for char in self.characteristics]
        
        # Validate all weights are non-negative
        if any(w < 0 for w in weights):
            raise ValueError("Characteristic weights must be non-negative")
        
        # Validate not all weights are zero
        weight_sum = sum(weights)
        if np.isclose(weight_sum, 0.0):
            raise ValueError("At least one characteristic must have a non-zero weight")
        
        # Normalize weights to sum to 1.0
        normalization_factor = 1.0 / weight_sum
        for char in self.characteristics:
            char.config.weight *= normalization_factor

    def process_loadings(self) -> None:
        """
        Process the loadings for each characteristic into a composite, normalized loading 
        (z-score). Sumproducts the weights and loadings of each characteristic, then
        runs a z-score of the weighted sum.
        """
        # Process loadings for each characteristic
        for char in self.characteristics:
            char.process_loadings()
        
        # Get all characteristic loadings as Series
        char_loadings = [char.get_loadings() for char in self.characteristics]
        char_weights = [char.config.weight for char in self.characteristics]
        
        # Calculate weighted sum of loadings
        weighted_sum = pd.Series(0, index=self.characteristics[0].date_vector)
        for loading, weight in zip(char_loadings, char_weights):
            if weight is None:
                raise ValueError("Can not process loadings with a weight of None")
            if loading is None:
                raise ValueError("Can not process loadings with a loading of None")
            weighted_sum += loading * weight
            
        # Z-score the weighted sum
        self._loadings = pd.Series(
            data=z_score(weighted_sum.values),
            index=weighted_sum.index,
            name="Composite Factor"
        )

    def get_loadings(self) -> Optional[pd.Series]:
        """
        Get the processed composite factor loadings.
        If loadings have not been processed, will process them.
        
        returns:
            Optional[pd.Series]: The composite factor loadings indexed by date
        """
        if self._loadings is None:
            self.process_loadings()
        return self._loadings