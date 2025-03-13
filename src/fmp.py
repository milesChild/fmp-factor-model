import requests
from typing import Optional
from src.routes import (
    BASE_URL,
    QUOTE_PATH,
    BULK_INCOME_STATEMENT_PATH,
    BULK_BALANCE_SHEET_PATH
)
from src.util import validate_year, validate_period

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
        
    def get_bulk_income_statement(self, year: str, period: str) -> dict:
        """
        Gets bulk income statement data for a given year and period.

        params:
            year (str): the year to get data for (e.g. "2024")
            period (str): the period, in quarters, to get data for (e.g. "Q1", "Q2", "Q3", "Q4")
        returns:
            dict: the bulk income statement data
        """
        # Endpoint: https://financialmodelingprep.com/stable/income-statement-bulk?year=YEAR&period=Q1
        if not validate_year(year):
            raise ValueError("Invalid year")
        if not validate_period(period):
            raise ValueError("Invalid period")
        raise NotImplementedError()
    
    def get_bulk_balance_sheet(self, year: str, period: str) -> 'pd.DataFrame':
        """
        Gets bulk balance sheet data for a given year and period.

        params:
            year (str): the year to get data for (e.g. "2024")
            period (str): the period, in quarters, to get data for (e.g. "Q1", "Q2", "Q3", "Q4")
        returns:
            pd.DataFrame: the bulk balance sheet data as a pandas DataFrame
        """
        if not validate_year(year):
            raise ValueError("Invalid year")
        if not validate_period(period):
            raise ValueError("Invalid period")
        
        url = f"{BASE_URL}/{BULK_BALANCE_SHEET_PATH}?year={year}&period={period}&apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            
            # Check if response is CSV
            content_type = response.headers.get('content-type', '')
            if 'text/csv' in content_type:
                import pandas as pd
                from io import StringIO
                return pd.read_csv(StringIO(response.text))
            
            # If not CSV, try JSON
            if response.status_code != 200:
                error_msg = response.json().get('message', 'Unknown error') if response.text else 'Empty response'
                raise ValueError(f"API request failed with status code {response.status_code}: {error_msg}")
            
            # If JSON response, convert to DataFrame
            import pandas as pd
            return pd.DataFrame(response.json())
        
        except requests.RequestException as e:
            raise ValueError(f"Could not connect to FMP API: {str(e)}")
        except ValueError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Error processing response: {str(e)}")


from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('FPM_API_KEY')
fmp = FMP(api_key)
df = fmp.get_bulk_balance_sheet("2024", "Q1")
# Save to CSV if needed
df.to_csv("bulk_balance_sheet.csv", index=False)


