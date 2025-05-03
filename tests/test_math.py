"""
Tests for math reward functions.
"""

import pytest
from reward_kit.rewards.math import (
    extract_numbers,
    compare_numbers,
    math_reward,
    advanced_math_reward
)
from reward_kit.models import RewardOutput


class TestExtractNumbers:
    def test_extract_integers(self):
        text = "The answer is 42. Another value is -17."
        numbers = extract_numbers(text)
        
        assert len(numbers) == 2
        assert numbers[0][0] == "42"
        assert numbers[0][1] == 42.0
        assert numbers[1][0] == "-17"
        assert numbers[1][1] == -17.0
    
    def test_extract_decimals(self):
        text = "The value of pi is 3.14159."
        numbers = extract_numbers(text)
        
        assert len(numbers) == 1
        assert numbers[0][0] == "3.14159"
        assert numbers[0][1] == 3.14159
    
    def test_extract_scientific_notation(self):
        text = "Avogadro's number is approximately 6.022e23."
        numbers = extract_numbers(text)
        
        assert len(numbers) == 1
        assert numbers[0][0] == "6.022e23"
        assert numbers[0][1] == 6.022e23
    
    def test_extract_fractions(self):
        text = "One half is 1/2 and three quarters is 3/4."
        numbers = extract_numbers(text)
        
        assert len(numbers) == 2
        assert numbers[0][0] == "1/2"
        assert numbers[0][1] == 0.5
        assert numbers[1][0] == "3/4"
        assert numbers[1][1] == 0.75
    
    def test_extract_with_units(self):
        text = "The distance is 42 km and the weight is 3.5 kg."
        numbers = extract_numbers(text)
        
        assert len(numbers) == 2
        assert numbers[0][0] == "42 km"
        assert numbers[0][1] == 42.0
        assert numbers[1][0] == "3.5 kg"
        assert numbers[1][1] == 3.5
    
    def test_multiple_formats(self):
        text = "Values: 42, 3.14, 2.71e-3, 1/4, 10 m, 5.5e6 Hz"
        numbers = extract_numbers(text)
        
        assert len(numbers) == 6
        # Check that all expected numbers are extracted (may not be in order)
        extracted_values = set(n[1] for n in numbers)
        assert 42.0 in extracted_values
        assert 3.14 in extracted_values
        assert 2.71e-3 in extracted_values
        assert 0.25 in extracted_values
        assert 10.0 in extracted_values
        assert 5.5e6 in extracted_values
        
    def test_latex_boxed(self):
        text = "The solution is $\\boxed{42}$"
        numbers = extract_numbers(text)
        
        assert len(numbers) == 1
        assert numbers[0][0] == "42"
        assert numbers[0][1] == 42.0
    
    def test_latex_boxed_fraction(self):
        text = "The final answer is $\\boxed{\\frac{3}{4}}$"
        numbers = extract_numbers(text)
        
        assert len(numbers) == 1
        assert numbers[0][0] == "3/4"
        assert numbers[0][1] == 0.75
    
    def test_latex_boxed_complex_expression(self):
        text = "Using the quadratic formula: $\\boxed{x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}}$, with a=1, b=-3, c=2, we get $\\boxed{x = 2}$ or $\\boxed{x = 1}$"
        numbers = extract_numbers(text)
        
        # We should extract the values 2 and 1
        assert len(numbers) >= 2
        extracted_values = set(n[1] for n in numbers)
        assert 1.0 in extracted_values
        assert 2.0 in extracted_values
    
    def test_latex_equation(self):
        text = "The equation is $E = mc^2$ where $m = 2 \\text{ kg}$ and $c = 3 \\times 10^8 \\text{ m/s}$"
        numbers = extract_numbers(text)
        
        # Should extract 2 kg and 3×10^8 m/s
        extracted_values = set(n[1] for n in numbers)
        assert 2.0 in extracted_values
        assert 3e8 in extracted_values or 300000000.0 in extracted_values


class TestCompareNumbers:
    def test_exact_match(self):
        is_match, similarity = compare_numbers(42.0, 42.0)
        assert is_match is True
        assert similarity == 1.0
    
    def test_close_match(self):
        is_match, similarity = compare_numbers(3.14159, 3.14, relative_tolerance=0.01)
        assert is_match is True
        assert similarity == 1.0
    
    def test_not_close_match(self):
        is_match, similarity = compare_numbers(10.0, 11.0, relative_tolerance=0.01)
        assert is_match is False
        assert similarity < 1.0
    
    def test_very_different(self):
        is_match, similarity = compare_numbers(100.0, 200.0, relative_tolerance=0.01)
        assert is_match is False
        assert similarity == 0.0
    
    def test_zero_expected(self):
        is_match, similarity = compare_numbers(0.0, 0.00001, absolute_tolerance=0.0001)
        assert is_match is True
        assert similarity == 1.0
        
        is_match, similarity = compare_numbers(0.0, 0.001, absolute_tolerance=0.0001)
        assert is_match is False
        assert similarity < 1.0


class TestMathReward:
    def test_basic_match(self):
        original_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
        
        result = math_reward(generated_messages, original_messages)
        assert isinstance(result, RewardOutput)
        assert result.score == 1.0
    
    def test_close_match(self):
        original_messages = [
            {"role": "user", "content": "What is the value of pi?"},
            {"role": "assistant", "content": "Pi is approximately 3.14159."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is the value of pi?"},
            {"role": "assistant", "content": "Pi is approximately 3.14."}
        ]
        
        result = math_reward(generated_messages, original_messages, tolerance=0.01)
        assert result.score == 1.0
    
    def test_wrong_answer(self):
        original_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 5."}
        ]
        
        result = math_reward(generated_messages, original_messages)
        assert result.score < 0.5
    
    def test_units_match(self):
        original_messages = [
            {"role": "user", "content": "What is the distance from Earth to Moon?"},
            {"role": "assistant", "content": "The distance is about 384,400 km."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is the distance from Earth to Moon?"},
            {"role": "assistant", "content": "The distance is about 384,400 km."}
        ]
        
        result = math_reward(generated_messages, original_messages, require_units=True)
        assert result.score == 1.0
    
    def test_units_mismatch(self):
        original_messages = [
            {"role": "user", "content": "What is the distance from Earth to Moon?"},
            {"role": "assistant", "content": "The distance is about 384,400 km."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is the distance from Earth to Moon?"},
            {"role": "assistant", "content": "The distance is about 384,400 miles."}
        ]
        
        result = math_reward(generated_messages, original_messages, require_units=True)
        assert result.score == 0.0
    
    def test_no_answers_found(self):
        original_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "I need to add two and two."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Let me calculate that for you."}
        ]
        
        result = math_reward(generated_messages, original_messages)
        assert result.score == 0.0
        assert "error" in result.metrics


class TestAdvancedMathReward:
    def test_multiple_answers_all_match(self):
        original_messages = [
            {"role": "user", "content": "Calculate 2+2 and 3*4"},
            {"role": "assistant", "content": "2+2=4 and 3*4=12"}
        ]
        
        generated_messages = [
            {"role": "user", "content": "Calculate 2+2 and 3*4"},
            {"role": "assistant", "content": "The answers are 4 and 12."}
        ]
        
        result = advanced_math_reward(generated_messages, original_messages, match_all_answers=True)
        assert result.score == 1.0
    
    def test_multiple_answers_partial_match(self):
        original_messages = [
            {"role": "user", "content": "Calculate 2+2, 3*4, and 10/2"},
            {"role": "assistant", "content": "2+2=4, 3*4=12, and 10/2=5"}
        ]
        
        generated_messages = [
            {"role": "user", "content": "Calculate 2+2, 3*4, and 10/2"},
            {"role": "assistant", "content": "The answers are 4 and 12."}
        ]
        
        result = advanced_math_reward(generated_messages, original_messages, match_all_answers=True)
        assert result.score == 0.0  # Not all answers matched
        
        result = advanced_math_reward(generated_messages, original_messages, match_all_answers=False)
        assert result.score == 1.0  # Best match is perfect
    
    def test_answer_with_different_formats(self):
        original_messages = [
            {"role": "user", "content": "What is one half?"},
            {"role": "assistant", "content": "One half is 1/2 or 0.5."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is one half?"},
            {"role": "assistant", "content": "One half is 0.5 or 50%."}
        ]
        
        result = advanced_math_reward(generated_messages, original_messages)
        assert result.score == 1.0
    
    def test_scientific_notation_match(self):
        original_messages = [
            {"role": "user", "content": "What is Avogadro's number?"},
            {"role": "assistant", "content": "Avogadro's number is approximately 6.022×10^23 or 6.022e23."}
        ]
        
        generated_messages = [
            {"role": "user", "content": "What is Avogadro's number?"},
            {"role": "assistant", "content": "Avogadro's number is approximately 6.022e23."}
        ]
        
        result = advanced_math_reward(generated_messages, original_messages)
        assert result.score == 1.0