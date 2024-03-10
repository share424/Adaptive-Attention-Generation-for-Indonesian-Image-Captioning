# this modules is not used
# the result is different from the original code
from .cider import CIDErScore
from .bleu import BLEUScore

__all__ = ["CIDErScore", "BLEUScore", "get_metric"]


def get_metric(name: str):
    return globals()[name]