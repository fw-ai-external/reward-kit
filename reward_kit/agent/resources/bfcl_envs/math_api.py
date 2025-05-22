"""Implementation of MathAPI."""

from typing import Dict, Any


class MathAPI:
    """A simple math API for BFCL evaluation."""

    def __init__(self) -> None:
        pass

    def _load_scenario(self, config: Any) -> None:
        # MathAPI is stateless, so no scenarios to load
        pass

    def add(self, a: float, b: float) -> Dict[str, float]:
        """Add two numbers"""
        return {"result": a + b}

    def subtract(self, a: float, b: float) -> Dict[str, float]:
        """Subtract b from a"""
        return {"result": a - b}

    def multiply(self, a: float, b: float) -> Dict[str, float]:
        """Multiply two numbers"""
        return {"result": a * b}

    def divide(self, a: float, b: float) -> Dict[str, Any]:
        """Divide a by b"""
        if b == 0:
            return {"error": "Cannot divide by zero"}
        return {"result": a / b}

    def square_root(self, a: float) -> Dict[str, Any]:
        """Calculate the square root of a number"""
        if a < 0:
            return {"error": "Cannot calculate square root of negative number"}
        return {"result": a**0.5}

    def power(self, base: float, exponent: float) -> Dict[str, float]:
        """Calculate base raised to the power of exponent"""
        return {"result": base**exponent}
