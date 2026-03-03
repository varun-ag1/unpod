"""
Currency utility module for handling currency symbols and formatting.
"""
from enum import Enum


class Currency(Enum):
    """Supported currencies with their symbols and formatting."""

    INR = {
        "symbol": "₹",  # Indian Rupee
        "code": "INR",
        "name": "Indian Rupee",
        "decimal_places": 2,
    }
    USD = {
        "symbol": "$",  # US Dollar
        "code": "USD",
        "name": "US Dollar",
        "decimal_places": 2,
    }
    EUR = {
        "symbol": "€",  # Euro
        "code": "EUR",
        "name": "Euro",
        "decimal_places": 2,
    }
    GBP = {
        "symbol": "£",  # British Pound
        "code": "GBP",
        "name": "British Pound",
        "decimal_places": 2,
    }
    # Add more currencies as needed


def get_currency_symbol(currency_code):
    """
    Get the currency symbol for the given currency code.

    Args:
        currency_code (str): ISO 4217 currency code (e.g., 'INR', 'USD')

    Returns:
        str: Currency symbol (e.g., '₹', '$')
    """
    try:
        return Currency[currency_code.upper()].value["symbol"]
    except KeyError:
        # Default to INR if currency not found
        return Currency.INR.value["symbol"]


def format_currency(amount, currency_code, include_symbol=True):
    """
    Format the amount with the appropriate currency symbol and formatting.

    Args:
        amount (float): The amount to format
        currency_code (str): ISO 4217 currency code (e.g., 'INR', 'USD')
        include_symbol (bool): Whether to include the currency symbol

    Returns:
        str: Formatted currency string
    """
    try:
        currency = Currency[currency_code.upper()].value
    except KeyError:
        # Default to INR if currency not found
        currency = Currency.INR.value

    # Format the number with proper decimal places and thousand separators
    formatted_amount = f"{amount:,.{currency['decimal_places']}f}"

    if include_symbol:
        return f"{currency['symbol']}{formatted_amount}"
    return formatted_amount
