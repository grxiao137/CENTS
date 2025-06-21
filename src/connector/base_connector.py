from typing import AnyStr, List
from abc import ABC, abstractmethod

from cdr import MessageRole, Message

class Connector(ABC):
    '''
    TODO: Add session-related operation.
    Abstract base class for Large Language Model connectors.
    '''
    

    def __init__(self, model_name, model_api=None):
        self.model_name = model_name
        self.model_api = model_api

    @abstractmethod
    def set_params(self, **param):
        pass

    @abstractmethod
    def submit(self, msgs: List[Message], count_tokens: bool = False, verbose: bool = False, retry: int = 3,
               **kwargs):
        pass

    def __str__(self):
        return f"Connector()"
