# %%

# TODO: なぜかマルチコアを活かせてないみたい

import jax
import numpy as np

from eleclove.components import Capacitor, Resistor, VoltageSource
from eleclove.core import AcMode, Circuit, Rand, VGround, VNode
from eleclove.utils import NPValue, run_on_cpu

run_on_cpu()

rand = Rand.seed(137)

circuit = Circuit()
gnd = VGround()
va = VNode("A")
vb = VNode("B")
circuit.add(VoltageSource(va, gnd, 5))
circuit.add(Capacitor(va, vb, 5e-6))
circuit.add(Resistor(vb, gnd, 5))

def analyze(f: NPValue):
  sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, AcMode(f))
  return sol[vb]

f_list = np.logspace(0, 6, 10000000)
v_list = jax.vmap(analyze)(f_list)
