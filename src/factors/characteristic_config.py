from typing import Optional
from enum import Enum

# TODO: More NaMethods
# https://chatgpt.com/share/67db1d20-8a18-8009-b403-f0a7bde82ee9
class NaMethod(Enum):
    """
    Method for handling missing values in the characteristic.
    """
    MEAN = "mean"
    MEDIAN = "median"

class CharacteristicConfig:
    def __init__(
        self,
        name: str = "Generic Characteristic",
        log_raw_values: bool = False,
        winsorize_raw_values: bool = True,
        na_method: NaMethod = NaMethod.MEDIAN,
        weight: Optional[float] = None
    ) -> None:
        """
        Initialize characteristic configuration.

        params:
            name (str): Name of the characteristic
            log_raw_values (bool): Whether to log transform raw values. Default is False.
            winsorize_raw_values (bool): Whether to winsorize raw values. Default is True.
            na_method (NaMethod): Method for handling missing values. Default is NaMethod.MEDIAN (impute with median).
            weight (Optional[float]): Weight of the characteristic in a composite. Default is None.
        """
        self.name = name
        self.log_raw_values = log_raw_values
        self.winsorize_raw_values = winsorize_raw_values
        self.na_method = na_method
        self.weight = weight
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate characteristic configuration parameters.
        
        raises:
            ValueError: If any validation check fails
        """
        # Validate basic config parameters
        if not isinstance(self.name, str) or len(self.name.strip()) == 0:
            raise ValueError("Characteristic name must be a non-empty string")
        
        if not isinstance(self.log_raw_values, bool):
            raise ValueError("log_raw_values must be a boolean")
            
        if not isinstance(self.winsorize_raw_values, bool):
            raise ValueError("winsorize_raw_values must be a boolean")
            
        if not isinstance(self.na_method, NaMethod):
            raise ValueError("na_method must be a NaMethod")
            
        if self.weight is not None:
            if not isinstance(self.weight, (int, float)):
                raise ValueError("weight must be a number if provided")
            if self.weight <= 0 or self.weight > 1:
                raise ValueError("weight must be between 0 and 1")