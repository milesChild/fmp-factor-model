import pytest
from src.factors.characteristic_config import CharacteristicConfig

class TestCharacteristicConfigInitialization:
    """Tests for characteristic config initialization functionality."""

    def test_valid_default_initialization(self):
        """Test initialization with default values."""
        config = CharacteristicConfig()
        assert config.name == "Generic Characteristic"
        assert config.log_raw_values is False
        assert config.winsorize_raw_values is True
        assert config.weight is None

    def test_valid_custom_initialization(self):
        """Test initialization with custom values."""
        config = CharacteristicConfig(
            name="Test Characteristic",
            log_raw_values=True,
            winsorize_raw_values=False,
            weight=0.5
        )
        assert config.name == "Test Characteristic"
        assert config.log_raw_values is True
        assert config.winsorize_raw_values is False
        assert config.weight == 0.5

    # Name validation tests
    def test_empty_name(self):
        """Test initialization with empty name."""
        with pytest.raises(ValueError, match="Characteristic name must be a non-empty string"):
            CharacteristicConfig(name="")

    def test_whitespace_name(self):
        """Test initialization with whitespace name."""
        with pytest.raises(ValueError, match="Characteristic name must be a non-empty string"):
            CharacteristicConfig(name="   ")

    def test_invalid_name_type(self):
        """Test initialization with invalid name type."""
        with pytest.raises(ValueError, match="Characteristic name must be a non-empty string"):
            CharacteristicConfig(name=123)

    # Log raw values validation tests
    def test_invalid_log_raw_values_type(self):
        """Test initialization with invalid log_raw_values type."""
        with pytest.raises(ValueError, match="log_raw_values must be a boolean"):
            CharacteristicConfig(log_raw_values="True")

    # Winsorize raw values validation tests
    def test_invalid_winsorize_raw_values_type(self):
        """Test initialization with invalid winsorize_raw_values type."""
        with pytest.raises(ValueError, match="winsorize_raw_values must be a boolean"):
            CharacteristicConfig(winsorize_raw_values="True")

    # Weight validation tests
    def test_invalid_weight_type(self):
        """Test initialization with invalid weight type."""
        with pytest.raises(ValueError, match="weight must be a number if provided"):
            CharacteristicConfig(weight="0.5")

    def test_negative_weight(self):
        """Test initialization with negative weight."""
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            CharacteristicConfig(weight=-0.5)

    def test_zero_weight(self):
        """Test initialization with zero weight."""
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            CharacteristicConfig(weight=0.0)

    def test_weight_greater_than_one(self):
        """Test initialization with weight > 1."""
        with pytest.raises(ValueError, match="weight must be between 0 and 1"):
            CharacteristicConfig(weight=1.5)

    def test_valid_weight_bounds(self):
        """Test initialization with valid weight bounds."""
        # Test lower bound
        config1 = CharacteristicConfig(weight=0.001)
        assert config1.weight == 0.001

        # Test upper bound
        config2 = CharacteristicConfig(weight=1.0)
        assert config2.weight == 1.0

        # Test middle value
        config3 = CharacteristicConfig(weight=0.5)
        assert config3.weight == 0.5

    # Multiple parameter tests
    def test_multiple_invalid_parameters(self):
        """Test initialization with multiple invalid parameters."""
        with pytest.raises(ValueError, match="Characteristic name must be a non-empty string"):
            CharacteristicConfig(
                name="",
                log_raw_values="invalid",
                winsorize_raw_values=None,
                weight=2.0
            ) 