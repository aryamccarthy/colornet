from pathlib import Path
import pickle
from typing import Dict

import torch as th

DEFAULT_EMBEDDINGS = Path("../../data/embeddings/subset-wb.p")

def load_embeddings(file: Path=DEFAULT_EMBEDDINGS) -> Dict[str, th.Tensor]:
    assert file.is_file()
    with open(file, 'rb') as f:
        vectors: Dict = pickle.load(f)
    for key in vectors:
        vectors[key] = th.from_numpy(vectors[key]).float()
    return vectors

if __name__ == '__main__':
    data = load_embeddings()
    print(data["blue"])
    assert len(data["blue"]) == 300
