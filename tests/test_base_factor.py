import pytest
import numpy as np
import datetime
from typing import List, Optional
import pandas as pd
from src.factors.base_factor import BaseFactor

class MockFactor(BaseFactor):
    """Mock implementation of BaseFactor for testing."""
    def process_loadings(self) -> None:
        """Simple implementation that averages all raw vectors."""
        weighted_sum = np.zeros_like(self.raw_vectors[0], dtype=float)
        for vec, weight in zip(self.raw_vectors, self.raw_weights):
            weighted_sum += vec * weight
        
        self._loadings = pd.Series(
            data=weighted_sum,
            index=self.date_vector,
            name="mock_factor"
        )
        self._processed = True

@pytest.fixture
def valid_dates() -> List[datetime.date]:
    return [datetime.date(2024, 1, i) for i in range(1, 6)]

@pytest.fixture
def valid_vectors() -> List[np.ndarray]:
    return [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    ]

@pytest.fixture
def valid_names() -> List[str]:
    return ["vec1", "vec2"]

@pytest.fixture
def valid_weights() -> List[np.ndarray]:
    # Each weight vector should sum to 0.5 so that together they sum to 1.0
    weight = 0.5
    return [
        np.full(5, weight),  # [0.5, 0.5, 0.5, 0.5, 0.5]
        np.full(5, weight)   # [0.5, 0.5, 0.5, 0.5, 0.5]
    ]

def test_valid_initialization(valid_dates, valid_vectors, valid_names, valid_weights):
    """Test initialization with valid inputs."""
    factor = MockFactor(valid_dates, valid_vectors, valid_names, valid_weights)
    assert factor.date_vector == valid_dates
    assert all(np.array_equal(a, b) for a, b in zip(factor.raw_vectors, valid_vectors))
    assert factor.raw_names == valid_names
    assert all(np.array_equal(a, b) for a, b in zip(factor.raw_weights, valid_weights))
    assert factor._loadings is None
    assert not factor._processed

def test_auto_names(valid_dates, valid_vectors):
    """Test automatic name generation when names not provided."""
    factor = MockFactor(valid_dates, valid_vectors)
    assert factor.raw_names == ["raw_0", "raw_1"]

def test_auto_weights(valid_dates, valid_vectors):
    """Test automatic weight generation when weights not provided."""
    factor = MockFactor(valid_dates, valid_vectors)
    assert len(factor.raw_weights) == len(valid_vectors)
    for weights in factor.raw_weights:
        assert np.allclose(weights, 0.5)  # Equal weights for 2 vectors
        assert len(weights) == len(valid_dates)

# Input validation tests
def test_empty_date_vector(valid_vectors):
    """Test initialization with empty date vector."""
    with pytest.raises(ValueError, match="date_vector cannot be empty"):
        MockFactor([], valid_vectors)

def test_invalid_date_type(valid_vectors):
    """Test initialization with invalid date types."""
    invalid_dates = [datetime.datetime.now() for _ in range(5)]  # datetime instead of date
    with pytest.raises(ValueError, match="must be datetime.date objects"):
        MockFactor(invalid_dates, valid_vectors)

def test_empty_raw_vectors(valid_dates):
    """Test initialization with empty raw vectors."""
    with pytest.raises(ValueError, match="Must provide at least one raw vector"):
        MockFactor(valid_dates, [])

def test_invalid_vector_type(valid_dates):
    """Test initialization with non-numpy array vectors."""
    invalid_vectors = [[1, 2, 3, 4, 5]]  # List instead of numpy array
    with pytest.raises(ValueError, match="must be a numpy array"):
        MockFactor(valid_dates, invalid_vectors)

def test_invalid_vector_dimension(valid_dates):
    """Test initialization with 2D array."""
    invalid_vectors = [np.array([[1, 2], [3, 4]])]
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        MockFactor(valid_dates, invalid_vectors)

def test_mismatched_vector_lengths(valid_dates):
    """Test initialization with vectors of different lengths."""
    invalid_vectors = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.0, 2.0, 3.0])  # Too short
    ]
    with pytest.raises(ValueError, match="length .* does not match date_vector length"):
        MockFactor(valid_dates, invalid_vectors)

def test_invalid_name_type(valid_dates, valid_vectors):
    """Test initialization with non-string names."""
    invalid_names = [1, 2]  # Numbers instead of strings
    with pytest.raises(ValueError, match="must be strings"):
        MockFactor(valid_dates, valid_vectors, invalid_names)

def test_mismatched_name_count(valid_dates, valid_vectors):
    """Test initialization with wrong number of names."""
    invalid_names = ["vec1"]  # Only one name for two vectors
    with pytest.raises(ValueError, match="does not match number of raw_vectors"):
        MockFactor(valid_dates, valid_vectors, invalid_names)

def test_invalid_weight_sum(valid_dates, valid_vectors):
    """Test initialization with weights that don't sum to 1."""
    invalid_weights = [
        np.full(5, 0.7),
        np.full(5, 0.7)  # Sum > 1
    ]
    with pytest.raises(ValueError, match="Weights must sum to 1.0 at each time point"):
        MockFactor(valid_dates, valid_vectors, raw_weights=invalid_weights)

# Functionality tests
def test_process_loadings(valid_dates, valid_vectors):
    """Test that process_loadings works correctly."""
    factor = MockFactor(valid_dates, valid_vectors)
    loadings = factor.get_loadings()
    
    assert isinstance(loadings, pd.Series)
    assert len(loadings) == len(valid_dates)
    assert all(loadings.index == valid_dates)
    assert factor._processed

def test_get_loadings_triggers_processing(valid_dates, valid_vectors):
    """Test that get_loadings triggers processing when needed."""
    factor = MockFactor(valid_dates, valid_vectors)
    assert not factor._processed
    
    # First call should trigger processing
    loadings1 = factor.get_loadings()
    assert factor._processed
    
    # Second call should use cached results
    loadings2 = factor.get_loadings()
    assert loadings1 is loadings2 