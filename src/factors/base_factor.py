from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np
import datetime

class BaseFactor(ABC):
    """Abstract base class for factor implementations."""
    
    def __init__(
        self, 
        date_vector: List[datetime.date], 
        raw_vectors: List[np.ndarray], 
        raw_names: Optional[List[str]] = None, 
        raw_weights: Optional[List[np.ndarray]] = None
        ) -> None:
        """
        Initialize the factor with raw data.

        params:
            date_vector (List[datetime.date]): List of dates corresponding to the raw values
            raw_vectors (List[np.ndarray]): List of raw value arrays to process into factor loadings
            raw_names (Optional[List[str]]): Names corresponding to each raw vector
            raw_weights (Optional[List[np.ndarray]]): Optional weights for each raw vector. If None,
                equal weights will be assigned to each raw vector
        
        raises:
            ValueError: If input validation fails
        """
        self._validate_inputs(date_vector, raw_vectors, raw_names, raw_weights)
        
        self.date_vector = date_vector
        self.raw_vectors = raw_vectors
        self.raw_names = raw_names if raw_names is not None else [f"raw_{i}" for i in range(len(raw_vectors))]
        
        # If no weights provided, create equal weights for each raw vector
        if raw_weights is None:
            n_vectors = len(raw_vectors)
            equal_weight = 1.0 / n_vectors
            self.raw_weights = [
                np.full_like(raw_vectors[0], equal_weight, dtype=float) 
                for _ in range(n_vectors)
            ]
        else:
            self.raw_weights = raw_weights
        
        # this is the vector of factor loadings that is generated from raw data in the process_loadings method
        self._loadings: Optional[pd.Series] = None

        # switch for whether or not loadings have been processed
        self._processed = False
        
    def _validate_inputs(
        self,
        date_vector: List[datetime.date],
        raw_vectors: List[np.ndarray],
        raw_names: Optional[List[str]],
        raw_weights: Optional[List[np.ndarray]]
    ) -> None:
        """
        Validate input parameters.
        
        raises:
            ValueError: If any validation check fails
        """
        # Validate date vector
        if not date_vector:
            raise ValueError("date_vector cannot be empty")
        
        if not all(isinstance(d, datetime.date) and not isinstance(d, datetime.datetime) for d in date_vector):
            raise ValueError("All elements in date_vector must be datetime.date objects (not datetime.datetime)")

        # Validate raw vectors
        if not raw_vectors:
            raise ValueError("Must provide at least one raw vector")

        # Validate array dimensions and lengths
        for i, rv in enumerate(raw_vectors):
            if not isinstance(rv, np.ndarray):
                raise ValueError(f"Raw vector at index {i} must be a numpy array")
            
            if rv.ndim != 1:
                raise ValueError(f"Raw vector at index {i} must be 1-dimensional")
            
            if len(rv) != len(date_vector):
                raise ValueError(f"Raw vector at index {i} length ({len(rv)}) does not match date_vector length ({len(date_vector)})")

        # Validate raw names if provided
        if raw_names is not None:
            if not all(isinstance(name, str) for name in raw_names):
                raise ValueError("All raw_names must be strings")
            
            if len(raw_names) != len(raw_vectors):
                raise ValueError(f"Number of raw_names ({len(raw_names)}) does not match number of raw_vectors ({len(raw_vectors)})")

        # Validate weights if provided
        if raw_weights is not None:
            if len(raw_weights) != len(raw_vectors):
                raise ValueError(f"Number of weight vectors ({len(raw_weights)}) does not match number of raw_vectors ({len(raw_vectors)})")
            
            for i, weights in enumerate(raw_weights):
                if not isinstance(weights, np.ndarray):
                    raise ValueError(f"Weight vector at index {i} must be a numpy array")
                
                if weights.ndim != 1:
                    raise ValueError(f"Weight vector at index {i} must be 1-dimensional")
                
                if len(weights) != len(date_vector):
                    raise ValueError(f"Weight vector at index {i} length ({len(weights)}) does not match date_vector length ({len(date_vector)})")
            
            # Check that weights sum to 1 at each time point
            weight_sums = np.sum([w for w in raw_weights], axis=0)
            if not np.allclose(weight_sums, 1.0, rtol=1e-5):
                raise ValueError(f"Weights must sum to 1.0 at each time point. Got sums: {weight_sums}")

    @abstractmethod
    def process_loadings(self) -> None:
        """
        Process the raw values into factor loadings.
        Must be implemented by concrete factor classes.
        Should set self._loadings to a pd.Series indexed by date_vector.
        """
        pass

    def get_loadings(self) -> Optional[pd.Series]:
        """
        Get the processed factor loadings.
        If loadings have not been processed, will process them and set the switch to True.
        
        returns:
            Optional[pd.Series]: The factor loadings indexed by date, or None if not processed
        """
        if not self._processed:
            self.process_loadings()
        return self._loadings
