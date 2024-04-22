# %%

import hashlib
from typing import Any

from numpy.typing import NDArray

NPArray = NDArray[Any]

def hex_id(obj):
  md5_hash = hashlib.md5(hex(id(obj)).encode()).hexdigest()
  return md5_hash[:6]
