# %%

from typing import Optional

from matplotlib import pyplot as plt

from eleclove.core import (Circuit, Component, Element, INode, Solution, VGround, VNode, VNodeFull)

class Resistor(Element):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    g = 1 / self._value
    eqs.add_a(self._pos, self._pos, g)
    eqs.add_a(self._neg, self._neg, g)
    eqs.add_a(self._pos, self._neg, -g)
    eqs.add_a(self._neg, self._pos, -g)

class CurrentSource(Element):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def modify_eqs(self, eqs, sol, dt):
    eqs.add_b(self._pos, -self._value)
    eqs.add_b(self._neg, self._value)

class VoltageSource(Element):
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

class Capacitor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: float):
    self._pos = pos
    self._neg = neg
    self._value = value

  def expand(self, sol, dt):
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

  def expand(self, sol, dt):
    i = 0 if sol is None else sol[self._inode]
    r = self._value / dt
    return [
        VoltageSource(self._pos, self._node, -r * i, self._inode),
        Resistor(self._node, self._neg, r),
    ]

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
