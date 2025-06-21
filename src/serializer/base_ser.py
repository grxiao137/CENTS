from typing import AnyStr
from pandas import DataFrame
from abc import ABC, abstractmethod

class Serializer(ABC):
    '''
    Abstract class for serializing data
    '''
    def __init__(self):
        pass

    @abstractmethod
    def serialize(self, df: DataFrame, cell_sep: AnyStr = ',', item_sep: AnyStr = '|', si: int = 0) -> str:
        pass

    def __str__(self):
        return f"Serializer()"