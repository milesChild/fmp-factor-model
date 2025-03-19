import pytest
from unittest.mock import patch, Mock
import requests
import pandas as pd
import sys
from pathlib import Path

# Add src to Python path for testing
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from fmp.fmp import FMP
from fmp.routes import BASE_URL, QUOTE_PATH

# Tests written with the help of claude 3.5 sonnet. Not really necessary for this project

@pytest.fixture
def mock_requests():
    with patch('requests.get') as mock_get:
        yield mock_get

def test_valid_api_key(mock_requests):
    valid_key = "valid_api_key_123"
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"price": 150.0}
    mock_requests.return_value = mock_response

    fmp = FMP(valid_key)

    mock_requests.assert_called_once_with(f"{BASE_URL}/{QUOTE_PATH}/AAPL?apikey={valid_key}")
    assert fmp.api_key == valid_key

def test_invalid_api_key(mock_requests):
    invalid_key = "invalid_key_456"
    mock_response = Mock()
    mock_response.status_code = 401
    mock_requests.return_value = mock_response

    with pytest.raises(ValueError, match="Invalid API key"):
        FMP(invalid_key)
    
    mock_requests.assert_called_once_with(f"{BASE_URL}/{QUOTE_PATH}/AAPL?apikey={invalid_key}")

def test_empty_api_key():
    with pytest.raises(ValueError, match="API key must be a non-empty string"):
        FMP("")

def test_none_api_key():
    with pytest.raises(ValueError, match="API key must be a non-empty string"):
        FMP(None)

def test_whitespace_api_key():
    with pytest.raises(ValueError, match="API key must be a non-empty string"):
        FMP("   ")

def test_network_error(mock_requests):
    mock_requests.side_effect = requests.RequestException("Connection refused")

    with pytest.raises(ValueError, match="Could not connect to FMP API: Connection refused"):
        FMP("some_key")

def test_unexpected_status_code(mock_requests):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"message": "Internal server error"}
    mock_requests.return_value = mock_response

    with pytest.raises(ValueError, match="API request failed with status code 500: Internal server error"):
        FMP("some_key")

def test_malformed_response(mock_requests):
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {}  # No message in response
    mock_requests.return_value = mock_response

    with pytest.raises(ValueError, match="API request failed with status code 500: Unknown error"):
        FMP("some_key")

# Sample CSV data based on the actual response structure
MOCK_CSV_DATA = """date,symbol,reportedCurrency,cik,filingDate,acceptedDate,fiscalYear,period,cashAndCashEquivalents,shortTermInvestments,cashAndShortTermInvestments,netReceivables
2024-03-31,000001.SZ,CNY,0.0,2024-03-31,2024-03-30 20:00:00,2024,Q1,656220000000.0,-347348000000.0,308872000000.0,0.0
2024-03-31,000002.SZ,CNY,0.0,2024-03-31,2024-03-30 20:00:00,2024,Q1,83066486948.0,13806634.0,83080293583.0,280842777780.0"""

# Sample JSON data for testing JSON response handling
MOCK_JSON_DATA = [
    {
        "date": "2024-03-31",
        "symbol": "000001.SZ",
        "reportedCurrency": "CNY",
        "cik": 0.0,
        "filingDate": "2024-03-31",
        "acceptedDate": "2024-03-30 20:00:00",
        "fiscalYear": 2024,
        "period": "Q1",
        "cashAndCashEquivalents": 656220000000.0,
        "shortTermInvestments": -347348000000.0,
        "cashAndShortTermInvestments": 308872000000.0,
        "netReceivables": 0.0
    }
]

# Add these constants after the existing MOCK_CSV_DATA and MOCK_JSON_DATA

MOCK_INCOME_STATEMENT_CSV = """date,symbol,reportedCurrency,cik,filingDate,acceptedDate,fiscalYear,period,revenue,costOfRevenue,grossProfit,researchAndDevelopmentExpenses,generalAndAdministrativeExpenses,sellingAndMarketingExpenses,sellingGeneralAndAdministrativeExpenses,otherExpenses,operatingExpenses,costAndExpenses,netInterestIncome,interestIncome,interestExpense,depreciationAndAmortization,ebitda,ebit
2024-03-31,000001.SZ,CNY,0.0,2024-03-31,2024-03-30 20:00:00,2024,Q1,65991000000.0,0.0,65991000000.0,0.0,10430000000.0,0.0,10430000000.0,-11466000000.0,11466000000.0,47501000000.0,25157000000.0,53369000000.0,28212000000.0,-18555000000.0,-30000000.0,18525000000.0
2024-03-31,000002.SZ,CNY,0.0,2024-03-31,2024-03-30 20:00:00,2024,Q1,61594149065.0,54928689328.0,6665459737.0,138118210.0,1525713207.0,1633074419.0,3158787626.0,19593914.0,5828623052.0,60757312381.0,-1092247233.58,374858587.0,1467105821.0,1260339558.0,2166468481.0,2166468481.0"""

MOCK_INCOME_STATEMENT_JSON = [
    {
        "date": "2024-03-31",
        "symbol": "000001.SZ",
        "reportedCurrency": "CNY",
        "cik": 0.0,
        "filingDate": "2024-03-31",
        "acceptedDate": "2024-03-30 20:00:00",
        "fiscalYear": 2024,
        "period": "Q1",
        "revenue": 65991000000.0,
        "costOfRevenue": 0.0,
        "grossProfit": 65991000000.0,
        "operatingExpenses": 11466000000.0,
        "ebitda": -30000000.0,
        "ebit": 18525000000.0
    }
]

@pytest.fixture
def fmp_client():
    with patch('fmp.fmp.FMP._FMP__test_api_key', return_value=None):
        return FMP("dummy_api_key")

def test_get_bulk_balance_sheet_csv_success(fmp_client):
    """Test successful CSV response handling"""
    with patch('requests.get') as mock_get:
        # Mock the CSV response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = MOCK_CSV_DATA
        mock_response.headers = {'content-type': 'text/csv'}
        mock_get.return_value = mock_response

        # Call the method
        result = fmp_client.get_bulk_balance_sheet("2024", "Q1")

        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['date', 'symbol', 'reportedCurrency', 'cik', 'filingDate', 
                                      'acceptedDate', 'fiscalYear', 'period', 'cashAndCashEquivalents',
                                      'shortTermInvestments', 'cashAndShortTermInvestments', 'netReceivables']
        assert result.iloc[0]['symbol'] == '000001.SZ'
        assert result.iloc[1]['symbol'] == '000002.SZ'

def test_get_bulk_balance_sheet_json_success(fmp_client):
    """Test successful JSON response handling"""
    with patch('requests.get') as mock_get:
        # Mock the JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_JSON_DATA
        mock_response.headers = {'content-type': 'application/json'}
        mock_get.return_value = mock_response

        # Call the method
        result = fmp_client.get_bulk_balance_sheet("2024", "Q1")

        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == '000001.SZ'

def test_get_bulk_balance_sheet_invalid_year(fmp_client):
    """Test invalid year parameter"""
    with pytest.raises(ValueError, match="Invalid year"):
        fmp_client.get_bulk_balance_sheet("invalid", "Q1")

def test_get_bulk_balance_sheet_invalid_period(fmp_client):
    """Test invalid period parameter"""
    with pytest.raises(ValueError, match="Invalid period"):
        fmp_client.get_bulk_balance_sheet("2024", "Q5")

def test_get_bulk_balance_sheet_api_error(fmp_client):
    """Test API error handling"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"message": "Invalid API key"}'
        mock_response.json.return_value = {"message": "Invalid API key"}
        mock_response.headers = {'content-type': 'application/json'}
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="API request failed with status code 401: Invalid API key"):
            fmp_client.get_bulk_balance_sheet("2024", "Q1")

def test_get_bulk_balance_sheet_connection_error(fmp_client):
    """Test connection error handling"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(ValueError, match="Could not connect to FMP API: Connection error"):
            fmp_client.get_bulk_balance_sheet("2024", "Q1")

def test_get_bulk_income_statement_csv_success(fmp_client):
    """Test successful CSV response handling for income statement"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = MOCK_INCOME_STATEMENT_CSV
        mock_response.headers = {'content-type': 'text/csv'}
        mock_get.return_value = mock_response

        result = fmp_client.get_bulk_income_statement("2024", "Q1")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'revenue' in result.columns
        assert 'grossProfit' in result.columns
        assert 'ebitda' in result.columns
        assert result.iloc[0]['symbol'] == '000001.SZ'
        assert result.iloc[0]['revenue'] == 65991000000.0
        assert result.iloc[1]['symbol'] == '000002.SZ'

def test_get_bulk_income_statement_json_success(fmp_client):
    """Test successful JSON response handling for income statement"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_INCOME_STATEMENT_JSON
        mock_response.headers = {'content-type': 'application/json'}
        mock_get.return_value = mock_response

        result = fmp_client.get_bulk_income_statement("2024", "Q1")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]['symbol'] == '000001.SZ'
        assert result.iloc[0]['revenue'] == 65991000000.0

def test_get_bulk_income_statement_invalid_year(fmp_client):
    """Test invalid year parameter for income statement"""
    with pytest.raises(ValueError, match="Invalid year"):
        fmp_client.get_bulk_income_statement("invalid", "Q1")

def test_get_bulk_income_statement_invalid_period(fmp_client):
    """Test invalid period parameter for income statement"""
    with pytest.raises(ValueError, match="Invalid period"):
        fmp_client.get_bulk_income_statement("2024", "Q5")

def test_get_bulk_income_statement_api_error(fmp_client):
    """Test API error handling for income statement"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = '{"message": "Invalid API key"}'
        mock_response.json.return_value = {"message": "Invalid API key"}
        mock_response.headers = {'content-type': 'application/json'}
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="API request failed with status code 401: Invalid API key"):
            fmp_client.get_bulk_income_statement("2024", "Q1")

def test_get_bulk_income_statement_connection_error(fmp_client):
    """Test connection error handling for income statement"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.RequestException("Connection error")
        
        with pytest.raises(ValueError, match="Could not connect to FMP API: Connection error"):
            fmp_client.get_bulk_income_statement("2024", "Q1") 