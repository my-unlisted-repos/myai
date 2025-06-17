from datetime import datetime, timedelta
import time

def epoch_to_datetime(epoch: float) -> datetime:
    """Seconds from the epoch to datetime"""
    return (datetime.fromtimestamp(0) + timedelta(seconds=epoch))