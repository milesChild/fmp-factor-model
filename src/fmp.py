import requests
from typing import Optional
from src.routes import (
    BASE_URL,
    QUOTE_PATH
)

class FMP:
    """Simplified FMP wrapper for getting factor model data. Not exhaustive impl of actual API endpoints."""
    
    def __init__(self, api_key: str):
        # try to heartbeat with fmp
        if error := self.__test_api_key(api_key):
            raise ValueError(error)
        print("[ INFO ] Successfully connected to FMP API")
        self.api_key = api_key

    def __test_api_key(self, api_key: str) -> Optional[str]:
        """
        Validates a client-provided API key. Must be a non-empty string. Must work on a basic
        price request from AAPL.

        params:
            api_key (str): the api key to validate
        returns:
            Optional[str]: error message if the api key doesn't work, otherwise None
        """
        if not isinstance(api_key, str) or not api_key or len(api_key.strip()) == 0:
            return "API key must be a non-empty string"

        try:
            response = requests.get(f"{BASE_URL}/{QUOTE_PATH}/AAPL?apikey={api_key}")
            
            if response.status_code == 401:
                return "Invalid API key"
            elif response.status_code != 200:
                return f"API request failed with status code {response.status_code}: {response.json().get('message', 'Unknown error')}"
            return None

        except requests.RequestException as e:
            return f"Could not connect to FMP API: {str(e)}"
        except Exception as e:
            return f"Unexpected error while testing API key: {str(e)}"