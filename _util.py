from typing import List, TypeVar

T = TypeVar("T")

def _flatten(a: List[List[T]]) -> List[T]:
    a_flatten = []
    for sublist in a:
        try:
            a_flatten += sublist
        except TypeError:
            a_flatten.append(sublist)
    return a_flatten