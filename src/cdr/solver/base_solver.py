from typing import AnyStr, Dict
from pandas import DataFrame
from abc import ABC, abstractmethod


class Solver(ABC):
    '''
    Abstract base class for solving tasks
    '''
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, c2c: Dict, c2w: Dict, c2s: Dict, c2b: Dict, total_budget: int, verbose=False, *kwargs):
        pass

    def __str__(self):
        return f"Solver()"