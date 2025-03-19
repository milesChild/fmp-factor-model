import pytest
import numpy as np
import pandas as pd
from src.factors.cross_sectional_characteristic import CrossSectionalCharacteristic
from src.factors.characteristic_config import CharacteristicConfig, NaMethod
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

class TestNaHandling:
    """Tests for handling NA values in characteristics."""
    
    @pytest.fixture
    def vector_with_nas(self):
        """Create a vector with NA values."""
        return np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 
                        11.0, np.nan, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
                        21.0, 22.0, 23.0, np.nan, 25.0, 26.0, 27.0, 28.0])

    def test_mean_imputation(self, vector_with_nas):
        """Test that NA values are properly imputed with mean when using NaMethod.MEAN."""
        config = CharacteristicConfig(na_method=NaMethod.MEAN)
        char = CrossSectionalCharacteristic(vector_with_nas, config)
        char.get_loadings()
        
        # assert that no values are nan
        assert not np.any(np.isnan(char._loadings.values))
        # assert that all the values that were originaly nan are now equal to eachother
        na_indices = np.where(pd.isna(vector_with_nas))[0]
        assert np.all(char._loadings.values[na_indices] == char._loadings.values[na_indices[0]])

    def test_median_imputation(self, vector_with_nas):
        """Test that NA values are properly imputed with median when using NaMethod.MEDIAN."""
        
        config = CharacteristicConfig(na_method=NaMethod.MEDIAN)
        char = CrossSectionalCharacteristic(vector_with_nas, config)
        char.get_loadings()

        # assert that no values are nan
        assert not np.any(np.isnan(char._loadings.values))
        # assert that all the values that were originaly nan are now equal to eachother
        na_indices = np.where(pd.isna(vector_with_nas))[0]
        assert np.all(char._loadings.values[na_indices] == char._loadings.values[na_indices[0]])

    def test_all_na_values(self):
        """Test handling of a vector containing all NA values."""
        all_na_vector = np.array([np.nan, np.nan, np.nan])
        
        # Test MEAN/MEDIAN - should raise an error as no valid values to compute from
        config_mean = CharacteristicConfig(na_method=NaMethod.MEAN)
        with pytest.raises(ValueError, match="Raw values must contain at least one non-NA value"):
            char_mean = CrossSectionalCharacteristic(all_na_vector, config_mean)
            char_mean.get_loadings()

    def test_single_non_na_value(self):
        """Test handling of a vector containing only one non-NA value."""
        vector = np.array([np.nan, np.nan, 5.0, np.nan])
        
        # Test MEAN/MEDIAN - should work and fill all values with 5.0
        config_mean = CharacteristicConfig(na_method=NaMethod.MEAN)
        char_mean = CrossSectionalCharacteristic(vector, config_mean)
        loadings_mean = char_mean.get_loadings()
        assert len(loadings_mean) == len(vector)
        assert np.all(loadings_mean == 0)  # All values same after z-scoring

    def test_na_handling_with_transformations(self, vector_with_nas):
        """Test that NA handling works correctly with log and winsorize transformations."""
        config = CharacteristicConfig(
            na_method=NaMethod.MEAN,
            log_raw_values=True,
            winsorize_raw_values=True
        )
        char = CrossSectionalCharacteristic(vector_with_nas, config)
        
        loadings = char.get_loadings()
        # Basic checks
        assert len(loadings) == len(vector_with_nas)
        assert not np.any(pd.isna(loadings))  # Just verify no NAs remain 