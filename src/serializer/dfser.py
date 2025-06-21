import re
from typing import AnyStr
from pandas import DataFrame
from collections import Counter
from abc import ABC, abstractmethod

from serializer import Serializer

class DFSerializer(Serializer):
    '''
    Dataframe based serializer
    '''

    def __init__(self):
        return

    def _clean(self, s: str) -> str:
        pattern = re.compile(r"[\|\\\-<>\/]")
        return pattern.sub("", s)

    def serialize(self, df: DataFrame, cell_sep: AnyStr = ',', item_sep: AnyStr = '|', si: int = 0) -> str:
        df = df.astype(str)
        str_builder = ""
        str_builder += "pd.DataFrame({\n"
        for idx, col in enumerate(df.columns):
            values = df[col].tolist()
            values = [v for v in values if v not in {'None', 'NaN', 'none', 'nan'}]
            value_counts = Counter(values)
            filtered_values = []
            for value, count in value_counts.items():
                if count > 5:
                    filtered_values.extend([value] * 5) 
                else:
                    filtered_values.extend([value] * count)  

            ser_col = cell_sep.join(filtered_values)
            if item_sep == 'sa':
                str_builder += f"{col}: {ser_col}, \n"
            else:
                str_builder += f"Column-{si + idx}: {ser_col}, \n"

        # inlcude header for SA only
        if item_sep == 'sa':
            index = df.columns
            serialized_index = ", ".join(map(str, index))
            str_builder += f"Headers already used: [{serialized_index}]"
            str_builder += "})"
        else:
            index = df.columns
            serialized_index = ", ".join(map(str, index))
            str_builder += f"Index: [{serialized_index}]"
            str_builder += "})"

        return str_builder

    def __str__(self):
        return f"DFSerializer()"
