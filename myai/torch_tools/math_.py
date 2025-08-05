import typing as T
import torch
import math
import numpy as np



def as_base_digits_numpy(number:int, base):
    """
    Convert an integer into a list of digits of that integer in a given base.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        numpy.ndarray: An array of digits representing the input integer in the given base.
    """
    if number == 0: return 0
    # Convert the input numbers to their digit representation in the given base
    digits = np.array([number])
    base_digits = (digits // base**(np.arange(int(np.log(number) / np.log(base)) + 1)[::-1])) % base

    return base_digits

def as_base_digits(number:int, base):
    """
    Convert an integer into a list of digits of that integer in a given base.

    Args:
        number (int): The integer to convert.
        base (int): The base to convert the integer to.

    Returns:
        torch.Tensor: An array of digits representing the input integer in the given base.
    """
    if number == 0: return torch.tensor([0])
    # Convert the input numbers to their digit representation in the given base
    digits = torch.tensor([number])
    base_digits = (digits // base**(torch.arange(int(math.log(number) / math.log(base)), -1, -1))) % base

    return base_digits