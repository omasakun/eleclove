# %%

from typing import Optional

from eleclove.core import (Circuit, Component, Element, INode, Rand, Solution, VGround, VNode, VNodeFull)
from eleclove.utils import NPValue, run_on_cpu

class Resistor(Element):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: NPValue):
    self._pos = pos
    self._neg = neg
    self.value = value

  def stamp(self, eqs, state):
    g = 1 / self.value
    eqs.add_a(self._pos, self._pos, g)
    eqs.add_a(self._neg, self._neg, g)
    eqs.add_a(self._pos, self._neg, -g)
    eqs.add_a(self._neg, self._pos, -g)

class CurrentSource(Element):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: NPValue):
    self._pos = pos
    self._neg = neg
    self.value = value

  def stamp(self, eqs, state):
    eqs.add_b(self._pos, -self.value)
    eqs.add_b(self._neg, self.value)

class VoltageSource(Element):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: NPValue, inode: Optional[INode] = None):
    self._pos = pos
    self._neg = neg
    self._inode = inode if inode is not None else INode()
    self.value = value

  def stamp(self, eqs, state):
    eqs.add_a(self._inode, self._pos, 1)
    eqs.add_a(self._pos, self._inode, 1)
    eqs.add_a(self._inode, self._neg, -1)
    eqs.add_a(self._neg, self._inode, -1)
    eqs.add_b(self._inode, self.value)

class Capacitor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: NPValue):
    self._pos = pos
    self._neg = neg
    self.value = value

  def expand(self, state):
    sol, dt = state.sol_prev, state.dt
    v_diff = 0 if sol is None else sol[self._pos] - sol[self._neg]
    g = self.value / dt
    return [
        CurrentSource(self._pos, self._neg, -g * v_diff),
        Resistor(self._pos, self._neg, 1 / g),
    ]

class Inductor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value: NPValue):
    self._pos = pos
    self._neg = neg
    self._node = VNode()
    self._inode = INode()
    self.value = value

  def expand(self, state):
    sol, dt = state.sol_prev, state.dt
    i = 0 if sol is None else sol[self._inode]
    r = self.value / dt
    return [
        VoltageSource(self._pos, self._node, -r * i, self._inode),
        Resistor(self._node, self._neg, r),
    ]

def _main():
  from matplotlib import pyplot as plt

  run_on_cpu()

  rand = Rand.seed(137)

  # 直列抵抗でためしてみる
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  circuit.add(VoltageSource(va, gnd, 1))
  circuit.add(Resistor(va, vb, 1))
  circuit.add(Resistor(vb, gnd, 3))
  sol, _ = circuit.solve(None, None, 0.1, 0.0, rand)
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
  for n in range(100):
    sol, _ = circuit.solve(sol, sol, 0.1, 0.1 * n, rand)
    v_list.append(sol[vb])
  plt.title(f"Capacitor (100 steps)")
  plt.plot(v_list)
  plt.show()

  v_list = []
  sol: Optional[Solution] = None
  for n in range(100):
    sol, _, converged = circuit.newton(sol, sol, 0.1, 0.1 * n, rand)
    v_list.append(sol[vb])
    assert converged
  plt.title(f"Capacitor (100 steps, Newton)")
  plt.plot(v_list)
  plt.show()

  v_list = []
  sol: Optional[Solution] = None
  for n in range(10000):
    sol, _ = circuit.solve(sol, sol, 0.001, 0.001 * n, rand)
    v_list.append(sol[vb])
  plt.title(f"Capacitor (10000 steps)")
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
  for n in range(100):
    sol, _ = circuit.solve(sol, sol, 0.1, 0.1 * n, rand)
    v_list.append(sol[vb])
  plt.title(f"Inductor (100 steps)")
  plt.plot(v_list)
  plt.show()

  # ... いい感じ！

if __name__ == "__main__": _main()
