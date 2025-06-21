import pandas as pd
from typing import Tuple, Callable, List, Any
from abc import ABC, abstractmethod


class Scorer(ABC):
    '''
    Base class for scoring cells
    '''

    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
    
    @abstractmethod
    def gen_score(self, df: pd.DataFrame) :
        '''
        Generate scores for all cells in the dataframe
        '''
        pass

    def __str__(self):
        return f"Scorer()"