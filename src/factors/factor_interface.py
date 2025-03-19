from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class FactorInterface(ABC):
    """Abstract interface for all factors (characteristics and composites)."""
    
    @abstractmethod
    def process_loadings(self) -> None:
        """
        Process the raw values into factor loadings.
        Implementation depends on the specific factor type.
        """
        pass
    
    @abstractmethod
    def get_loadings(self) -> Optional[pd.Series]:
        """
        Get the processed factor loadings.
        If loadings have not been processed, will process them.
        
        returns:
            Optional[pd.Series]: The factor loadings
        """
        pass 