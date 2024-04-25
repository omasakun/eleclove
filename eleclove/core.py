# %%

# あとで整理する
# まずは性能を無視して開発して、あとから最適化する
# TODO: ニュートン法とか使う、収束判定、可変時間ステップ

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Protocol, Sequence, TypeGuard, Union

import jax.numpy as jnp
import jax.random
import numpy as np
from jax.tree_util import register_pytree_node_class

from eleclove.utils import KeyArray, NPArray, NPValue, hex_id

EqNode = Union["VNode", "INode"]
EqNodeFull = Union["VNode", "INode", "VGround"]
VNodeFull = Union["VNode", "VGround"]

class VGround:
  """ グラウンド電圧 """
  def __str__(self):
    return "VGround"

class VNode:
  """ 電圧ノード (JS における Symbol のように使う) """
  def __init__(self, name: Optional[str] = None):
    self._name = name

  def __str__(self):
    if self._name is not None:
      return f"VNode({hex_id(self)}/{self._name})"
    else:
      return f"VNode({hex_id(self)})"

class INode:
  """ 電流ノード (JS における Symbol のように使う) """
  def __init__(self, name: Optional[str] = None):
    self._name = name

  def __str__(self):
    if self._name is not None:
      return f"INode({hex_id(self)}/{self._name})"
    else:
      return f"INode({hex_id(self)})"

class LinearEqs:
  """ EqNode でアクセスできる連立線形方程式 """
  def __init__(self):
    self._names: dict[EqNode, int] = {}  # 順序が保証された OrderedDict と等価 (Python 3.7 以降)
    self._a: list[tuple[int, int, Any]] = []
    self._b: list[tuple[int, Any]] = []

  def _node_index(self, node: EqNode) -> int:
    if node not in self._names:
      self._names[node] = len(self._names)
    return self._names[node]

  def add_a(self, i: EqNodeFull, j: EqNodeFull, value: Any):
    if isinstance(i, VGround) or isinstance(j, VGround): return
    self._a.append((self._node_index(i), self._node_index(j), value))

  def add_b(self, i: EqNodeFull, value: Any):
    if isinstance(i, VGround): return
    self._b.append((self._node_index(i), value))

  def matrices(self):
    n = len(self._names)
    a = jnp.zeros((n, n))
    b = jnp.zeros(n)
    for i, j, value in self._a:
      a = a.at[i, j].add(value)
    for i, value in self._b:
      b = b.at[i].add(value)

    return a, b

  def solve(self):
    a, b = self.matrices()
    x = jnp.linalg.solve(a, b)
    return Solution(x, list(self._names.keys()))

  def __str__(self):
    # TODO: きれいに表示する
    a, b = self.matrices()
    names = map(str, self._names.keys())
    return f"LinearEqs({list(names)}\na=\n{a}, \nb={b}, \n)"

@register_pytree_node_class
class Solution:
  """ EqNode でアクセスできる解一覧 """
  def __init__(self, values: NPArray, names: list[EqNode] | dict[EqNode, int]):
    self._values = values
    if isinstance(names, list):
      self._names = {name: i for i, name in enumerate(names)}
    else:
      self._names = names

  def to_numpy(self):
    return Solution(np.array(self._values), self._names)

  def __getitem__(self, key: EqNodeFull):
    if isinstance(key, VGround): return 0
    return self._values[self._names[key]]

  def __str__(self):
    lines: list[str] = []
    for name, value in zip(self._names, self._values):
      lines.append(f"{name}: {value}")
    return "\n".join(lines)

  def tree_flatten(self):
    return (self._values,), self._names

  @classmethod
  def tree_unflatten(cls, aux, children):
    return cls(children[0], aux)

@register_pytree_node_class
class Rand:
  def __init__(self, key: KeyArray):
    self._key = key

  @property
  def key(self):
    return self._key

  @classmethod
  def seed(cls, seed: int):
    return cls(jax.random.PRNGKey(seed))

  def next_key(self):
    _key, self._key = jax.random.split(self._key)
    return _key

  def randn(self) -> NPArray:
    return jax.random.normal(self.next_key())  # type: ignore

  def tree_flatten(self):
    return (self._key,), None

  @classmethod
  def tree_unflatten(cls, aux, children):
    return cls(children[0])

@dataclass
class TransientState:
  sol: Optional[Solution]
  dt: NPValue
  t: NPValue
  rand: Rand

class Element(Protocol):
  def stamp(self, eqs: LinearEqs, state: TransientState):
    ...

class Component(Protocol):
  def expand(self, state: TransientState) -> Sequence[Union["Element", "Component"]]:
    ...

def is_element(obj: Any) -> TypeGuard[Element]:
  return hasattr(obj, "stamp")

def is_component(obj: Any) -> TypeGuard[Component]:
  return hasattr(obj, "expand")

class Circuit:
  def __init__(self):
    self._children: list[Element | Component] = []
    self._jitted_solve = None

  def add(self, child: Element | Component):
    self._children.append(child)
    self._jitted_solve = None

  def solve(self, sol: Optional[Solution], dt: NPValue, t: NPValue, rand: Rand) -> tuple[Solution, Rand]:
    if sol is None:
      return self._solve(sol, dt, t, rand)

    if self._jitted_solve is None:
      self._jitted_solve = jax.jit(self._solve)

    # 何通りの JIT コンパイルが行われたか確認する
    # print(self._jitted_solve._cache_size())

    return self._jitted_solve(sol, dt, t, rand)

  def _solve(self, sol: Optional[Solution], dt: NPValue, t: NPValue, rand: Rand):
    eqs = LinearEqs()
    state = TransientState(sol, dt, t, rand)

    queue = self._children.copy()
    while queue:
      child = queue.pop(0)
      if is_element(child):
        child.stamp(eqs, state)
      if is_component(child):
        for child in child.expand(state):
          queue.append(child)

    # TODO: 前回の計算で使った NpArray を再利用する
    return eqs.solve(), rand
