from typing import List, Optional
import datetime
import pandas as pd
import numpy as np
from src.factors.characteristic_config import CharacteristicConfig
from src.factors.factor_util import log_transform, winsorize, z_score

class Characteristic:
    """A characteristic is an attribute of a security that is used when creating composite factors."""

    def __init__(
        self,
        date_vector: List[datetime.date],
        raw_vector: np.ndarray,
        config: Optional[CharacteristicConfig] = CharacteristicConfig()
    ) -> None:
        """
        Initialize characteristic.

        params:
            date_vector (List[datetime.date]): List of dates corresponding to the raw values
            raw_values (List[np.ndarray]): List of raw value arrays to process into characteristic loadings
            config (CharacteristicConfig): Configuration for characteristic processing. Default is used if not provided.
        """
        self.date_vector = date_vector
        self.raw_vector = raw_vector
        self.config = config
        self._validate_characteristic()
        
        # Add loading attributes
        self._loadings: Optional[pd.Series] = None

    def _validate_characteristic(self) -> None:
        """
        Validate characteristic parameters.
        
        raises:
            ValueError: If any validation check fails
        """
        # Validate date vector
        if not self.date_vector:
            raise ValueError("date_vector cannot be empty")
        
        if not all(isinstance(d, datetime.date) and not isinstance(d, datetime.datetime) for d in self.date_vector):
            raise ValueError("All elements in date_vector must be datetime.date objects (not datetime.datetime)")

        # Validate raw vector
        if not isinstance(self.raw_vector, np.ndarray):
            raise ValueError("Raw value must be a numpy array")
        
        if self.raw_vector.size == 0:
            raise ValueError("Must provide a vector of raw values")
        
        if self.raw_vector.ndim != 1:
            raise ValueError("Raw value must be a 1-dimensional array")
        
        if len(self.raw_vector) != len(self.date_vector):
            raise ValueError(f"Raw value length ({len(self.raw_vector)}) does not match date_vector length ({len(self.date_vector)})")

    def process_loadings(self) -> None:
        """
        Process the raw values into characteristic loadings (z-scores).
        Applies transformations based on configuration (log, winsorize).
        Sets self._loadings to the processed values.
        """
        values = self.raw_vector.copy()
        
        # Apply log transformation if configured
        if self.config.log_raw_values:
            values = log_transform(values)
        
        # Apply winsorization if configured
        if self.config.winsorize_raw_values:
            values = winsorize(values)

        values = z_score(values)

        # Store as Series
        self._loadings = pd.Series(
            data=values,
            index=self.date_vector,
            name=self.config.name
        )

    def get_loadings(self) -> Optional[pd.Series]:
        """
        Get the processed characteristic loadings.
        If loadings have not been processed, will process them.
        
        returns:
            Optional[pd.Series]: The characteristic loadings indexed by date
        """
        if not isinstance(self._loadings, pd.Series):
            self.process_loadings()
        return self._loadings