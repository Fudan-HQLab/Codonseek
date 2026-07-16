"""Shared utilities for the RL pipeline.

Provides the ``timer`` decorator and other small helpers used across
multiple modules.
"""

import time
import functools


def timer(method_name: str):
    """Decorator that prints the wall-clock duration of a method call."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            print(f"{method_name}: {elapsed:.2f}s")
            return result
        return wrapper
    return decorator
