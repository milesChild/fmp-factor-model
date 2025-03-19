import pytest
import numpy as np
import pandas as pd
from src.factors.cross_sectional_characteristic import CrossSectionalCharacteristic
from src.factors.characteristic_config import CharacteristicConfig
from unittest.mock import patch

# Fixtures for common test data
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
    with patch('src.factors.cross_sectional_characteristic.log_transform') as mock_log, \
         patch('src.factors.cross_sectional_characteristic.winsorize') as mock_winsorize, \
         patch('src.factors.cross_sectional_characteristic.z_score') as mock_z:
        
        # Configure the mocks to return different arrays so we can track transformations
        mock_log.side_effect = lambda x: x + 100
        mock_winsorize.side_effect = lambda x: x + 200
        mock_z.side_effect = lambda x: x + 300
        
        yield {
            'log': mock_log,
            'winsorize': mock_winsorize,
            'z_score': mock_z
        }

class TestCrossSectionalCharacteristicInitialization:
    """Tests for CrossSectionalCharacteristic initialization."""
    
    def test_valid_initialization(self, valid_vector, valid_config):
        """Test initialization with valid parameters."""
        char = CrossSectionalCharacteristic(valid_vector, valid_config)
        
        assert np.array_equal(char.raw_vector, valid_vector)
        assert char.config == valid_config
        assert char._loadings is None

    def test_default_config_initialization(self, valid_vector):
        """Test initialization with default config."""
        # Create a new config instance to avoid any shared state
        config = CharacteristicConfig()
        char = CrossSectionalCharacteristic(valid_vector, config)
        
        assert isinstance(char.config, CharacteristicConfig)
        assert char.config.name == "Generic Characteristic"
        assert char.config.log_raw_values is False
        assert char.config.winsorize_raw_values is True
        assert char.config.weight is None

class TestRawVectorValidation:
    """Tests for raw vector validation."""
    
    def test_empty_raw_vector(self):
        """Test initialization with empty raw vector."""
        with pytest.raises(ValueError, match="Must provide a vector of raw values"):
            CrossSectionalCharacteristic(np.array([]))

    def test_list_instead_of_ndarray(self):
        """Test initialization with list instead of numpy array."""
        with pytest.raises(ValueError, match="Raw value must be a numpy array"):
            CrossSectionalCharacteristic([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_2d_array(self):
        """Test initialization with 2D array."""
        invalid_vector = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        with pytest.raises(ValueError, match="Raw value must be a 1-dimensional array"):
            CrossSectionalCharacteristic(invalid_vector)

class TestLoadingProcessing:
    """Tests for loading processing functionality."""
    
    def test_process_loadings_all_transforms(self, valid_vector, mock_transforms):
        """Test processing with all transformations enabled."""
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=True,
            winsorize_raw_values=True
        )
        char = CrossSectionalCharacteristic(valid_vector, config)
        char.process_loadings()

        mock_transforms['log'].assert_called_once()
        mock_transforms['winsorize'].assert_called_once()
        mock_transforms['z_score'].assert_called_once()

        expected_values = valid_vector + 100 + 200 + 300
        pd.testing.assert_series_equal(
            char._loadings,
            pd.Series(data=expected_values, name="Test")
        )

    def test_process_loadings_no_transforms(self, valid_vector, mock_transforms):
        """Test processing with no transformations."""
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=False,
            winsorize_raw_values=False
        )
        char = CrossSectionalCharacteristic(valid_vector, config)
        char.process_loadings()

        mock_transforms['log'].assert_not_called()
        mock_transforms['winsorize'].assert_not_called()
        mock_transforms['z_score'].assert_called_once()

        expected_values = valid_vector + 300
        pd.testing.assert_series_equal(
            char._loadings,
            pd.Series(data=expected_values, name="Test")
        )

    def test_process_loadings_with_negative_values(self):
        """Test processing with negative values."""
        # Create a vector with negative values (like returns)
        vector = np.array([0.15, 0.08, -0.12, 0.25, 0.05, -0.03])
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=False,  # Don't log transform returns
            winsorize_raw_values=True
        )
        char = CrossSectionalCharacteristic(vector, config)
        char.process_loadings()
        
        # Verify no NaN values in the output
        assert not np.any(np.isnan(char._loadings.values))
        
        # Verify the values are properly scaled
        assert np.std(char._loadings.values) == pytest.approx(1.0, rel=1e-10)  # Should be z-scored
        assert np.mean(char._loadings.values) == pytest.approx(0.0, rel=1e-10)  # Should be centered

    def test_process_loadings_with_identical_values(self):
        """Test processing with identical values."""
        vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=False,
            winsorize_raw_values=True
        )
        char = CrossSectionalCharacteristic(vector, config)
        char.process_loadings()
        
        # Verify no NaN values in the output
        assert not np.any(np.isnan(char._loadings.values))
        
        # All values should be 0 after z-scoring (mean=1, std=0)
        assert np.all(char._loadings.values == 0)

    def test_process_loadings_with_extreme_values(self):
        """Test processing with extreme values."""
        vector = np.array([1e6, 1e-6, 1e6, 1e-6, 1e6])
        config = CharacteristicConfig(
            name="Test",
            log_raw_values=False,
            winsorize_raw_values=True
        )
        char = CrossSectionalCharacteristic(vector, config)
        char.process_loadings()
        
        # Verify no NaN values in the output
        assert not np.any(np.isnan(char._loadings.values))
        
        # Verify the values are properly scaled
        assert np.std(char._loadings.values) == pytest.approx(1.0, rel=1e-10)
        assert np.mean(char._loadings.values) == pytest.approx(0.0, rel=1e-10)

class TestLoadingsRetrieval:
    """Tests for loadings retrieval functionality."""
    
    def test_get_loadings_triggers_processing(self, valid_vector, mock_transforms):
        """Test that get_loadings triggers processing when needed."""
        char = CrossSectionalCharacteristic(valid_vector)
        
        assert char._loadings is None
        loadings = char.get_loadings()
        
        mock_transforms['winsorize'].assert_called_once()
        mock_transforms['z_score'].assert_called_once()
        
        assert char._loadings is not None
        assert loadings is char._loadings

    def test_get_loadings_reuses_processed(self, valid_vector, mock_transforms):
        """Test that get_loadings reuses existing processed values."""
        char = CrossSectionalCharacteristic(valid_vector)
        
        first_loadings = char.get_loadings()
        mock_transforms['winsorize'].reset_mock()
        mock_transforms['z_score'].reset_mock()
        
        second_loadings = char.get_loadings()
        
        mock_transforms['winsorize'].assert_not_called()
        mock_transforms['z_score'].assert_not_called()
        assert first_loadings is second_loadings 