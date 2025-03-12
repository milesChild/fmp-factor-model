import pytest
from unittest.mock import patch, Mock
import requests

from src.fmp import FMP
from src.routes import BASE_URL, QUOTE_PATH

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