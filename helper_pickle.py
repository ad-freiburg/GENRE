from typing import Optional

import pickle


def pickle_load(path: Optional[str]):
    if path is None:
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
