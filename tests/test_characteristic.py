import pytest
import numpy as np
import datetime
import pandas as pd
from src.factors.characteristic import Characteristic
from src.factors.characteristic_config import CharacteristicConfig
from unittest.mock import patch, Mock

# Fixtures for common test data
@pytest.fixture
def valid_dates():
    """Create a list of valid dates."""
    return [datetime.date(2024, 1, i) for i in range(1, 6)]

@pytest.fixture
def valid_vector():
    """Create a valid numpy array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def valid_config():
    """Create a valid characteristic config."""
    return CharacteristicConfig(
        name="Test Characteristic",
        log_raw_values=True,
        winsorize_raw_values=True,
        weight=0.5
    )

@pytest.fixture
def mock_transforms():
    """Create mock transform functions."""
    with patch('src.factors.characteristic.log_transform') as mock_log, \
         patch('src.factors.characteristic.winsorize') as mock_winsorize, \
         patch('src.factors.characteristic.z_score') as mock_z:
        
        # Configure the mocks to return different arrays so we can track transformations
        mock_log.side_effect = lambda x: x + 100
        mock_winsorize.side_effect = lambda x: x + 200
        mock_z.side_effect = lambda x: x + 300
        
        yield {
            'log': mock_log,
            'winsorize': mock_winsorize,
            'z_score': mock_z
        }

class TestCharacteristicInitialization:
    """Tests for Characteristic initialization."""
    
    def test_valid_initialization(self, valid_dates, valid_vector, valid_config):
        """Test initialization with valid parameters."""
        char = Characteristic(valid_dates, valid_vector, valid_config)
        
        assert char.date_vector == valid_dates
        assert np.array_equal(char.raw_vector, valid_vector)
        assert char.config == valid_config
        assert char._loadings is None

    def test_default_config_initialization(self, valid_dates, valid_vector):
        """Test initialization with default config."""
        char = Characteristic(valid_dates, valid_vector)
        
        assert isinstance(char.config, CharacteristicConfig)
        assert char.config.name == "Generic Characteristic"
        assert char.config.log_raw_values is False
        assert char.config.winsorize_raw_values is True
        assert char.config.weight is None

class TestDateVectorValidation:
    """Tests for date vector validation."""
    
    def test_empty_date_vector(self, valid_vector):
        """Test initialization with empty date vector."""
        with pytest.raises(ValueError, match="date_vector cannot be empty"):
            Characteristic([], valid_vector)

    def test_datetime_instead_of_date(self, valid_vector):
        """Test initialization with datetime objects instead of date objects."""
        invalid_dates = [datetime.datetime.now() for _ in range(5)]
        with pytest.raises(ValueError, match="must be datetime.date objects"):
            Characteristic(invalid_dates, valid_vector)

    def test_mixed_date_types(self, valid_vector):
        """Test initialization with mixed date types."""
        mixed_dates = [
            datetime.date(2024, 1, 1),
            datetime.datetime.now(),
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4),
            datetime.date(2024, 1, 5)
        ]
        with pytest.raises(ValueError, match="must be datetime.date objects"):
            Characteristic(mixed_dates, valid_vector)

class TestRawVectorValidation:
    """Tests for raw vector validation."""
    
    def test_empty_raw_vector(self, valid_dates):
        """Test initialization with empty raw vector."""
        with pytest.raises(ValueError, match="Must provide a vector of raw values"):
            Characteristic(valid_dates, np.array([]))

    def test_list_instead_of_ndarray(self, valid_dates):
        """Test initialization with list instead of numpy array."""
        with pytest.raises(ValueError, match="Raw value must be a numpy array"):
            Characteristic(valid_dates, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_2d_array(self, valid_dates):
        """Test initialization with 2D array."""
        invalid_vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        with pytest.raises(ValueError, match="Raw value must be a 1-dimensional array"):
            Characteristic(valid_dates, invalid_vector)

    def test_mismatched_lengths(self, valid_dates):
        """Test initialization with mismatched lengths."""
        short_vector = np.array([1.0, 2.0, 3.0])  # Too short
        with pytest.raises(ValueError, match="Raw value length .* does not match date_vector length"):
            Characteristic(valid_dates, short_vector)

class TestLoadingProcessing:
    """Tests for loading processing functionality."""
    
    def test_process_loadings_all_transforms(self, valid_dates, valid_vector, mock_transforms):
        """Test processing with all transformations enabled."""
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=True,
            winsorize_raw_values=True
        )
        char = Characteristic(valid_dates, valid_vector, config)
        char.process_loadings()

        mock_transforms['log'].assert_called_once()
        mock_transforms['winsorize'].assert_called_once()
        mock_transforms['z_score'].assert_called_once()

        expected_values = valid_vector + 100 + 200 + 300
        pd.testing.assert_series_equal(
            char._loadings,
            pd.Series(data=expected_values, index=valid_dates, name="Test")
        )

    def test_process_loadings_no_transforms(self, valid_dates, valid_vector, mock_transforms):
        """Test processing with no transformations."""
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=False,
            winsorize_raw_values=False
        )
        char = Characteristic(valid_dates, valid_vector, config)
        char.process_loadings()

        mock_transforms['log'].assert_not_called()
        mock_transforms['winsorize'].assert_not_called()
        mock_transforms['z_score'].assert_called_once()

        expected_values = valid_vector + 300
        pd.testing.assert_series_equal(
            char._loadings,
            pd.Series(data=expected_values, index=valid_dates, name="Test")
        )

class TestLoadingsRetrieval:
    """Tests for loadings retrieval functionality."""
    
    def test_get_loadings_triggers_processing(self, valid_dates, valid_vector, mock_transforms):
        """Test that get_loadings triggers processing when needed."""
        char = Characteristic(valid_dates, valid_vector)
        
        assert char._loadings is None
        loadings = char.get_loadings()
        
        mock_transforms['winsorize'].assert_called_once()
        mock_transforms['z_score'].assert_called_once()
        
        assert char._loadings is not None
        assert loadings is char._loadings

    def test_get_loadings_reuses_processed(self, valid_dates, valid_vector, mock_transforms):
        """Test that get_loadings reuses existing processed values."""
        char = Characteristic(valid_dates, valid_vector)
        
        first_loadings = char.get_loadings()
        mock_transforms['winsorize'].reset_mock()
        mock_transforms['z_score'].reset_mock()
        
        second_loadings = char.get_loadings()
        
        mock_transforms['winsorize'].assert_not_called()
        mock_transforms['z_score'].assert_not_called()
        assert first_loadings is second_loadings 