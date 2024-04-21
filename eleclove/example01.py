# %%

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from eleclove.main import (Capacitor, Circuit, Component, CurrentSource, Inductor, Resistor, Solution, VGround, VNode, VNodeFull)

class CustomResistor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull):
    self._pos = pos
    self._neg = neg

  def elements(self, sol, dt):
    v = 0 if sol is None else sol[self._pos] - sol[self._neg]
    i = -0.05 * v + 0.1 * v**3
    return [
        CurrentSource(self._pos, self._neg, i),
    ]

class WhiteNoise(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull):
    self._pos = pos
    self._neg = neg

  def elements(self, sol, dt):
    i = 10e-6 * np.random.randn()
    return [
        CurrentSource(self._pos, self._neg, i),
    ]

circuit = Circuit()
gnd = VGround()

va = VNode("A")

circuit.add(WhiteNoise(va, gnd))
circuit.add(CustomResistor(va, gnd))
circuit.add(Resistor(va, gnd, 10.15))  # なんでここシビアなの？
# circuit.add(Resistor(va, gnd, 100))
circuit.add(Inductor(va, gnd, 50e-12))
circuit.add(Capacitor(va, gnd, 0.005e-12))
circuit.add(CustomResistor(va, gnd))

dt = 10e-15
t_list = []
va_list = []
sol: Optional[Solution] = None
for i in range(10000):
  sol = circuit.solve(sol, dt)
  t_list.append(i * dt)
  va_list.append(sol[va])

plt.plot(t_list, va_list)
plt.show()
