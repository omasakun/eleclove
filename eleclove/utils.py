# %%

import json
from contextlib import contextmanager
from pathlib import Path
from time import monotonic_ns
from typing import Any

import json5
from numpy.typing import NDArray

NPArray = NDArray[Any]

@contextmanager
def timer(desc: str = "Duration"):
  start = monotonic_ns()
  yield
  end = monotonic_ns()
  print(f"{desc}: {(end - start) / 1e6:.3f} ms")

def save_json(obj: dict, path: str | Path, exist_ok=False):
  with open(path, "w" if exist_ok else "x") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path):
  with open(path, "r") as f:
    return json5.load(f)  # json5 supports comments
