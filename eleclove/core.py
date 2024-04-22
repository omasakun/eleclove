# %%

# あとで整理する
# まずは性能を無視して開発して、あとから最適化する
# TODO: ニュートン法とか使う、収束判定、可変時間ステップ

from typing import (Any, Optional, OrderedDict, Protocol, Sequence, TypeGuard, Union)

import numpy as np

from eleclove.utils import NPArray, hex_id

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
    self._names: OrderedDict[EqNode, int] = OrderedDict()
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

  def _prepare(self):
    n = len(self._names)
    a = np.zeros((n, n))
    b = np.zeros(n)
    for i, j, value in self._a:
      a[i, j] += value
    for i, value in self._b:
      b[i] += value

    return a, b

  def solve(self):
    a, b = self._prepare()
    x = np.linalg.solve(a, b)
    return Solution(x, list(self._names.keys()))

  def __str__(self):
    # TODO: きれいに表示する
    a, b = self._prepare()
    names = map(str, self._names.keys())
    return f"LinearEqs({list(names)}\na=\n{a}, \nb={b}, \n)"

class Solution:
  """ EqNode でアクセスできる解一覧 """
  def __init__(self, values: NPArray, names: list[EqNode]):
    assert values.ndim == 1
    assert len(values) == len(names)
    self._values = values
    self._names = OrderedDict((name, i) for i, name in enumerate(names))

  def __getitem__(self, key: EqNodeFull):
    if isinstance(key, VGround): return 0
    return self._values[self._names[key]]

  def __str__(self):
    lines: list[str] = []
    for name, value in zip(self._names, self._values):
      lines.append(f"{name}: {value}")
    return "\n".join(lines)

class Element(Protocol):
  def modify_eqs(self, eqs: LinearEqs, sol: Optional[Solution], dt: float):
    ...

class Component(Protocol):
  def expand(self, sol: Optional[Solution], dt: float) -> Sequence[Union["Element", "Component"]]:
    ...

def is_element(obj: Any) -> TypeGuard[Element]:
  return hasattr(obj, "modify_eqs")

def is_component(obj: Any) -> TypeGuard[Component]:
  return hasattr(obj, "expand")

class Circuit:
  def __init__(self):
    self._children: list[Element | Component] = []

  def add(self, child: Element | Component):
    self._children.append(child)

  def solve(self, sol: Optional[Solution], dt: float):
    eqs = LinearEqs()

    queue = self._children.copy()
    while queue:
      child = queue.pop(0)
      if is_element(child):
        child.modify_eqs(eqs, sol, dt)
      if is_component(child):
        for child in child.expand(sol, dt):
          queue.append(child)

    # TODO: 前回の計算で使った NpArray を再利用する
    return eqs.solve()
