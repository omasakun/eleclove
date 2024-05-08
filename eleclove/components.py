# %%

import math
from typing import Optional

import jax

from eleclove.core import Component, Element, INode, VNode, VNodeFull
from eleclove.utils import NPValue

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

  def stamp_ac(self, eqs, state, freq):
    self.stamp(eqs, state)

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

  def stamp_ac(self, eqs, state, freq):
    self.stamp(eqs, state)

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

  def expand_ac(self, state, freq):
    j = jax.lax.complex(0.0, 1.0)
    w = 2 * math.pi * freq
    z = 1 / (j * w * self.value)
    return [
        Resistor(self._pos, self._neg, z),
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

  def expand_ac(self, state, freq):
    j = jax.lax.complex(0.0, 1.0)
    w = 2 * math.pi * freq
    z = j * w * self.value
    return [
        Resistor(self._pos, self._neg, z),
    ]
