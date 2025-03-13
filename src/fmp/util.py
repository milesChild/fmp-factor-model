def validate_year(year: str) -> bool:
    """
    Validates a year. Must be a string of 4 digits.
    """
    return len(year) == 4 and year.isdigit()

def validate_period(period: str) -> bool:
    """
    Validates a period. Must be like "Q1", "Q2", "Q3", "Q4"
    """
    return period in ["Q1", "Q2", "Q3", "Q4"]