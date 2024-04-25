# %%

import hashlib
from typing import Any

import jax
from jax.typing import ArrayLike
from numpy.typing import NDArray

# TODO: これらの型宣言はドキュメンテーションとオートコンプリートのためにつけているけど、型チェックには弱いかも
NPArray = NDArray[Any]
NPValue = NDArray[Any] | float | int
KeyArray = ArrayLike  # key for random number generator

def hex_id(obj):
  md5_hash = hashlib.md5(hex(id(obj)).encode()).hexdigest()
  return md5_hash[:6]

def run_on_cpu():
  jax.config.update('jax_platform_name', "cpu")
