from typing import Any
import math
from decimal import Decimal

def round_significant(x: Any, nsignificant: int):
    if x == 0: return 0.0
    if math.isnan(x) or math.isinf(x): return x
    if nsignificant <= 0: raise ValueError("nsignificant must be positive")

    x = Decimal(x) # otherwise there are rounding errors
    order = Decimal(10) ** math.floor(math.log10(abs(x)))
    v = round(x / order, nsignificant - 1) * order
    return float(v)


