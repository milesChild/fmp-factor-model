from typing import Optional

class CharacteristicConfig:
    def __init__(
        self,
        name: str = "Generic Characteristic",
        log_raw_values: bool = False,
        winsorize_raw_values: bool = True,
        weight: Optional[float] = None
    ) -> None:
        """
        Initialize characteristic configuration.

        params:
            name (str): Name of the characteristic
            log_raw_values (bool): Whether to log transform raw values
            winsorize_raw_values (bool): Whether to winsorize raw values
            weight (Optional[float]): Weight of the characteristic in a composite
        """
        self.name = name
        self.log_raw_values = log_raw_values
        self.winsorize_raw_values = winsorize_raw_values
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
            
        if self.weight is not None:
            if not isinstance(self.weight, (int, float)):
                raise ValueError("weight must be a number if provided")
            if self.weight <= 0 or self.weight > 1:
                raise ValueError("weight must be between 0 and 1")