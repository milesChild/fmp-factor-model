import pytest
import numpy as np
import datetime
import pandas as pd
from src.factors.composite_factor import CompositeFactor
from src.factors.characteristic import Characteristic
from src.factors.characteristic_config import CharacteristicConfig
from unittest.mock import patch

# Fixtures for common test data
@pytest.fixture
def valid_dates():
    """Create a list of valid dates."""
    return [datetime.date(2024, 1, i) for i in range(1, 6)]

@pytest.fixture
def valid_vectors():
    """Create a list of valid numpy arrays."""
    return [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        np.array([3.0, 6.0, 9.0, 12.0, 15.0])
    ]

@pytest.fixture
def valid_configs():
    """Create a list of valid characteristic configs."""
    return [
        CharacteristicConfig(name="Char1", weight=0.5),
        CharacteristicConfig(name="Char2", weight=0.3),
        CharacteristicConfig(name="Char3", weight=0.2)
    ]

@pytest.fixture
def valid_characteristics(valid_dates, valid_vectors, valid_configs):
    """Create a list of valid characteristics."""
    return [
        Characteristic(valid_dates, vec, config)
        for vec, config in zip(valid_vectors, valid_configs)
    ]

@pytest.fixture
def equal_weight_characteristics(valid_dates, valid_vectors):
    """Create characteristics with no explicit weights."""
    return [
        Characteristic(valid_dates, vec, CharacteristicConfig(name=f"Char{i+1}"))
        for i, vec in enumerate(valid_vectors)
    ]

class TestCompositeFactorInitialization:
    """Tests for CompositeFactor initialization."""
    
    def test_valid_initialization(self, valid_characteristics):
        """Test initialization with valid characteristics."""
        factor = CompositeFactor(valid_characteristics)
        
        assert factor.characteristics == valid_characteristics
        assert factor._loadings is None
        assert len(factor.characteristics) == 3
        
        # Verify weights remained unchanged (already sum to 1.0)
        weights = [char.config.weight for char in factor.characteristics]
        assert weights == [0.5, 0.3, 0.2]

    def test_empty_characteristics(self):
        """Test initialization with empty characteristics list."""
        with pytest.raises(ValueError, match="Must provide at least one characteristic"):
            CompositeFactor([])

    def test_equal_weight_assignment(self, equal_weight_characteristics):
        """Test that equal weights are assigned when none provided."""
        factor = CompositeFactor(equal_weight_characteristics)
        
        weights = [char.config.weight for char in factor.characteristics]
        expected_weight = 1.0 / len(equal_weight_characteristics)
        assert all(w == expected_weight for w in weights)
        assert sum(weights) == pytest.approx(1.0)

class TestDateVectorValidation:
    """Tests for date vector validation."""
    
    @pytest.fixture
    def mismatched_dates_char(self, valid_vectors, valid_configs):
        """Create a characteristic with different dates."""
        different_dates = [datetime.date(2024, 2, i) for i in range(1, 6)]
        return Characteristic(different_dates, valid_vectors[0], valid_configs[0])
    
    def test_mismatched_date_vectors(self, valid_characteristics, mismatched_dates_char):
        """Test initialization with mismatched date vectors."""
        invalid_chars = valid_characteristics[:1] + [mismatched_dates_char]
        with pytest.raises(ValueError, match=".*mismatched date vector*"):
            CompositeFactor(invalid_chars)

class TestWeightValidation:
    """Tests for weight validation and normalization."""
    
    @pytest.fixture
    def mock_config_class(self):
        """Create a mock CharacteristicConfig that allows any weight."""
        class MockCharacteristicConfig(CharacteristicConfig):
            def _validate_config(self):
                """Override validation to allow any weight."""
                if not isinstance(self.name, str) or len(self.name.strip()) == 0:
                    raise ValueError("Characteristic name must be a non-empty string")
                
                if not isinstance(self.log_raw_values, bool):
                    raise ValueError("log_raw_values must be a boolean")
                    
                if not isinstance(self.winsorize_raw_values, bool):
                    raise ValueError("winsorize_raw_values must be a boolean")
                    
                if self.weight is not None and not isinstance(self.weight, (int, float)):
                    raise ValueError("weight must be a number if provided")
        
        with patch('src.factors.characteristic_config.CharacteristicConfig', MockCharacteristicConfig):
            yield MockCharacteristicConfig

    @pytest.fixture
    def characteristics_with_weights(self, valid_dates, valid_vectors, mock_config_class):
        """Create characteristics with specific weights."""
        def _make_chars(weights):
            return [
                Characteristic(
                    valid_dates, 
                    valid_vectors[i], 
                    mock_config_class(name=f"Char{i+1}", weight=w)
                )
                for i, w in enumerate(weights)
            ]
        return _make_chars

    def test_normalize_valid_weights(self, characteristics_with_weights):
        """Test normalization of valid weights that don't sum to 1."""
        chars = characteristics_with_weights([0.5, 1.0, 0.5])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        assert sum(weights) == pytest.approx(1.0)
        assert weights[0] == pytest.approx(0.25)  # 0.5/2.0
        assert weights[1] == pytest.approx(0.50)  # 1.0/2.0
        assert weights[2] == pytest.approx(0.25)  # 0.5/2.0

    def test_all_weights_none(self, characteristics_with_weights):
        """Test handling of all None weights."""
        chars = characteristics_with_weights([None, None, None])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        assert all(w == pytest.approx(1/3) for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_some_weights_none(self, characteristics_with_weights):
        """Test handling of some None weights."""
        chars = characteristics_with_weights([0.5, None, 0.5])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        assert weights[0] == pytest.approx(0.5)
        assert weights[1] == pytest.approx(0.0)
        assert weights[2] == pytest.approx(0.5)
        assert sum(weights) == pytest.approx(1.0)

    def test_negative_weights(self, characteristics_with_weights):
        """Test rejection of negative weights."""
        chars = characteristics_with_weights([0.5, -0.3, 0.8])
        with pytest.raises(ValueError, match="Characteristic weights must be non-negative"):
            CompositeFactor(chars)

    def test_all_zero_weights(self, characteristics_with_weights):
        """Test rejection of all zero weights."""
        chars = characteristics_with_weights([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="At least one characteristic must have a non-zero weight"):
            CompositeFactor(chars)

    def test_single_characteristic_weight(self, characteristics_with_weights):
        """Test weight handling with a single characteristic."""
        chars = characteristics_with_weights([0.5])  # Any non-zero weight should work
        factor = CompositeFactor(chars)
        
        assert factor.characteristics[0].config.weight == pytest.approx(1.0)

    def test_very_small_weights(self, characteristics_with_weights):
        """Test handling of very small weights."""
        chars = characteristics_with_weights([1e-8, 1e-8, 1e-8])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        assert all(w == pytest.approx(1/3) for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    def test_mixed_magnitude_weights(self, characteristics_with_weights):
        """Test normalization of weights with very different magnitudes."""
        chars = characteristics_with_weights([1e-8, 1.0, 1e8])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        total = 1e-8 + 1.0 + 1e8
        assert weights[0] == pytest.approx(1e-8/total)
        assert weights[1] == pytest.approx(1.0/total)
        assert weights[2] == pytest.approx(1e8/total)
        assert sum(weights) == pytest.approx(1.0)

    def test_weight_precision(self, characteristics_with_weights):
        """Test that weight normalization maintains reasonable precision."""
        chars = characteristics_with_weights([1/3, 1/3, 1/3])
        factor = CompositeFactor(chars)
        
        weights = [char.config.weight for char in factor.characteristics]
        assert all(w == pytest.approx(1/3) for w in weights)
        assert sum(weights) == pytest.approx(1.0)