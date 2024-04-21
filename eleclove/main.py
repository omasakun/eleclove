# %%

# あとで整理する
# まずは性能を無視して開発して、あとから最適化する

import hashlib
from copy import copy
from typing import Any, Optional, OrderedDict, Protocol, Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

NPArray = NDArray[Any]

def hex_id(obj):
  md5_hash = hashlib.md5(hex(id(obj)).encode()).hexdigest()
  return md5_hash[:6]

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

EqNode = VNode | INode
EqNodeFull = VNode | INode | VGround
VNodeFull = VNode | VGround

class Solution:
  def __init__(self, values: NPArray, names: list[EqNode]):
    assert values.ndim == 1
    assert len(values) == len(names)
    self._values = values
    self._names = names

  def __getitem__(self, key: EqNodeFull):
    if isinstance(key, VGround): return 0
    return self._values[self._names.index(key)]

  def __str__(self):
    lines: list[str] = []
    for name, value in zip(self._names, self._values):
      lines.append(f"{name}: {value}")
    return "\n".join(lines)

class LinearEqs:
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

class Element(Protocol):
  def modify_eqs(self, eqs: LinearEqs, sol: Optional[Solution], dt: float):
    ...

class Component(Protocol):
  def elements(self, sol: Optional[Solution], dt: float) -> Sequence[Element]:
    ...

class Circuit:
  def __init__(self):
    self._components: list[Component] = []

  def add(self, component: Component):
    self._components.append(component)

  def solve(self, sol: Optional[Solution], dt: float):
    eqs = LinearEqs()
    for component in self._components:
      for element in component.elements(sol, dt):
        element.modify_eqs(eqs, sol, dt)
    # TODO: 前回の計算で使った NpArray を再利用する
    return eqs.solve()

class Resistor(Element, Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    eqs.add_a(self._pos, self._pos, 1 / self._value)
    eqs.add_a(self._neg, self._neg, 1 / self._value)
    eqs.add_a(self._pos, self._neg, -1 / self._value)
    eqs.add_a(self._neg, self._pos, -1 / self._value)

  def elements(self, sol, dt):
    return [self]

class CurrentSource(Element, Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    eqs.add_b(self._pos, -self._value)
    eqs.add_b(self._neg, self._value)

  def elements(self, sol, dt):
    return [self]

class VoltageSource(Element, Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float, inode: Optional[INode] = None):
    self._pos = pos
    self._neg = neg
    self._inode = inode if inode is not None else INode()
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    eqs.add_a(self._inode, self._pos, 1)
    eqs.add_a(self._pos, self._inode, 1)
    eqs.add_a(self._inode, self._neg, -1)
    eqs.add_a(self._neg, self._inode, -1)
    eqs.add_b(self._inode, self._value)

  def elements(self, sol, dt):
    return [self]

class Capacitor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def elements(self, sol, dt):
    v_diff = 0 if sol is None else sol[self._pos] - sol[self._neg]
    g = self._value / dt
    return [
        CurrentSource(self._pos, self._neg, -g * v_diff),
        Resistor(self._pos, self._neg, 1 / g),
    ]

class Inductor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._node = VNode()
    self._inode = INode()
    self._value = value

  def elements(self, sol, dt):
    i = 0 if sol is None else sol[self._inode]
    r = self._value / dt
    return [
        VoltageSource(self._pos, self._node, -r * i, self._inode),
        Resistor(self._node, self._neg, r),
    ]

class Inductor_Ex10(Element, Component):  # example 10 再実装
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float, inode: Optional[INode] = None):
    self._pos = pos
    self._neg = neg
    self._inode = inode if inode is not None else INode()
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    i = 0 if sol is None else sol[self._inode]
    r = self._value / dt
    v = r * i
    eqs.add_a(self._inode, self._pos, 1)
    eqs.add_a(self._pos, self._inode, 1)
    eqs.add_a(self._inode, self._neg, -1)
    eqs.add_a(self._neg, self._inode, -1)
    eqs.add_a(self._inode, self._inode, -r)
    eqs.add_b(self._inode, -v)

  def elements(self, sol, dt):
    return [self]

if __name__ == "__main__":
  # 直列抵抗でためしてみる
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  circuit.add(VoltageSource(va, gnd, 1))
  circuit.add(Resistor(va, vb, 1))
  circuit.add(Resistor(vb, gnd, 3))
  sol = circuit.solve(None, 0)
  print(sol)  # A: 1.0V, B: 0.75V
  print()

  # コンデンサでためしてみる
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  circuit.add(VoltageSource(va, gnd, 1))
  circuit.add(Resistor(va, vb, 1))
  circuit.add(Capacitor(vb, gnd, 1))

  v_list = []
  sol: Optional[Solution] = None
  for _ in range(100):
    sol = circuit.solve(sol, 0.1)
    v_list.append(sol[vb])
  plt.plot(v_list)
  plt.show()

  v_list = []
  sol: Optional[Solution] = None
  for _ in range(10000):
    sol = circuit.solve(sol, 0.001)
    v_list.append(sol[vb])
  plt.plot(v_list)
  plt.show()

  # インダクタでためしてみる
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  circuit.add(VoltageSource(va, gnd, 1))
  circuit.add(Resistor(va, vb, 1))
  circuit.add(Inductor(vb, gnd, 1))

  v_list = []
  sol: Optional[Solution] = None
  for _ in range(100):
    sol = circuit.solve(sol, 0.1)
    v_list.append(sol[vb])
  plt.plot(v_list)
  plt.show()

  # ... いい感じ！
