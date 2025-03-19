from typing import Optional
import pandas as pd
import numpy as np

from src.factors.characteristic_config import CharacteristicConfig, NaMethod
from src.factors.factor_util import log_transform, winsorize, z_score
from src.factors.factor_interface import FactorInterface

class CrossSectionalCharacteristic(FactorInterface):
    """A cross-sectional characteristic is an attribute of a security that is used when creating composite factors."""

    def __init__(
        self,
        raw_vector: np.ndarray,
        config: Optional[CharacteristicConfig] = CharacteristicConfig()
    ) -> None:
        """
        Initialize cross-sectional characteristic.

        params:
            raw_values (np.ndarray): Array of raw values to process into characteristic loadings
            config (CharacteristicConfig): Configuration for characteristic processing. Default is used if not provided.
        """
        self.raw_vector = raw_vector
        self.config = config
        self._validate_characteristic()
        
        # Add loading attributes
        self._loadings: Optional[pd.Series] = None

    def _validate_characteristic(self) -> None:
        """
        Validate cross-sectional characteristic parameters.
        
        raises:
            ValueError: If any validation check fails
        """
        # Validate raw vector
        if not isinstance(self.raw_vector, np.ndarray):
            raise ValueError("Raw value must be a numpy array")
        
        if self.raw_vector.size == 0:
            raise ValueError("Must provide a vector of raw values")
        
        if self.raw_vector.ndim != 1:
            raise ValueError("Raw value must be a 1-dimensional array")
        
        if np.isnan(self.raw_vector).all():
            raise ValueError("Raw values must contain at least one non-NA value")

    def process_loadings(self) -> None:
        """
        Process the raw values into cross-sectional characteristic loadings (z-scores).
        Applies transformations based on configuration (log, winsorize).
        Sets self._loadings to the processed values.
        """
        # Convert to pandas Series for NA handling
        values = pd.Series(self.raw_vector.copy())

        # Handle NA according to the configuration
        # TODO: If more NaMethods are added, this should be refactored
        if self.config.na_method == NaMethod.MEAN:
            values = values.fillna(values.mean())
        elif self.config.na_method == NaMethod.MEDIAN:
            values = values.fillna(values.median())
        
        # Convert back to numpy array for transformations
        values = values.to_numpy()
        
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
            name=self.config.name
        )

    def get_loadings(self) -> Optional[pd.Series]:
        """
        Get the processed cross-sectional characteristic loadings.
        If loadings have not been processed, will process them.
        
        returns:
            Optional[pd.Series]: The cross-sectional characteristic loadings
        """
        if not isinstance(self._loadings, pd.Series):
            self.process_loadings()
        return self._loadings