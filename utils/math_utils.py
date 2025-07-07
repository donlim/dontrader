# utils/math_utils.py
"""
math_utils.py

General mathematical and statistical utility functions for trading_bot.
"""
from collections import deque

def compute_smoothed_score(buffer):
    """
    Computes the mean of the scores in the buffer.
    """
    if not buffer:
        return 0.0
    return sum(buffer) / len(buffer)