# %%

import hashlib
from typing import Any, NoReturn

import jax
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

# TODO: これらの型宣言はドキュメンテーションとオートコンプリートのためにつけているけど、型チェックには弱いかも
NPArray = Array | NDArray[Any]
NPValue = Array | NDArray[Any] | float | int
NPBool = Array | NDArray[Any] | bool
KeyArray = ArrayLike  # key for random number generator

def hex_id(obj):
  md5_hash = hashlib.md5(hex(id(obj)).encode()).hexdigest()
  return md5_hash[:6]

def run_on_cpu():
  jax.config.update('jax_platform_name', "cpu")

def never(x: NoReturn):
  raise AssertionError(f"Unreachable code: {x}")
