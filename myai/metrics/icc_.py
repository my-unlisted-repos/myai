import typing as T
import numpy as np
import pandas as pd
import pingouin
import torch
from ..transforms import tonumpy

def icc(measurements):
    """Calculate intraclass correlation coefficient (ICC) for a Bartko data matrix.

    Args:
        measurements: A matrix of measurements, must have shape (N, K), where N is the number of objects and K is the number of raters.

    Returns:
        DataFrame with pingoin results.
    """
    measurements = tonumpy(measurements)
    cols: list[list] = [['object', 'rater', 'value']]
    for iobj, obj in enumerate(measurements):
        for irater, value in enumerate(obj):
            cols.append([iobj, irater, float(value)])

    data = pd.DataFrame(cols[1:], columns = cols[0])
    results = pingouin.intraclass_corr(data=data, targets = 'object', raters = 'rater', ratings = 'value')
    return results

ICCTypes = T.Literal['ICC1', 'ICC2', 'ICC3', 'ICC1k', 'ICC2k', 'ICC3k']
_ICCMAP = {"ICC1": 0, "ICC2": 1, "ICC3": 2, "ICC1k": 3, "ICC2k": 4, "ICC3k": 5}

def icc_value(measurements, icc_type: ICCTypes) -> float:
    """Calculate intraclass correlation coefficient (ICC) for a Bartko data matrix.

    Args:
        measurements: A matrix of measurements, must have shape (N, K), where N is the number of objects and K is the number of raters.
        icc_type (ICCTypes): type of ICC to use.

    Returns:
        float: ICC value.
    """
    return icc(measurements)[_ICCMAP[icc_type]][0]


