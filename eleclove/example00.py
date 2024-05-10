# %%

# うまく実装できているか確認するために使ったコード
# 実行速度をあまり気にしていない

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from eleclove.components import Capacitor, Inductor, Resistor, VoltageSource
from eleclove.core import (AcMode, Circuit, DcMode, Rand, Solution, VGround, VNode)
from eleclove.utils import run_on_cpu

run_on_cpu()

rand = Rand.seed(137)

# ハイパスフィルター
circuit = Circuit()
gnd = VGround()
va = VNode("A")
vb = VNode("B")
circuit.add(VoltageSource(va, gnd, 5))
circuit.add(Capacitor(va, vb, 5e-6))
circuit.add(Resistor(vb, gnd, 5))
f_list = []
v_list = []
for f in np.logspace(0, 6, 100):
  sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, AcMode(float(f)))
  f_list.append(f)
  v_list.append(sol[vb])
plt.title("High-pass filter (CR) [dB]")
plt.plot(f_list, 20 * np.log10(np.real(v_list)))
plt.xscale("log")
plt.show()

plt.title("High-pass filter (CR) [phase]")
plt.plot(f_list, (np.angle(v_list, deg=True) + 180) % 360 - 180)
plt.ylim(0, 90)
plt.xscale("log")
plt.show()

# ハイパスフィルター
circuit = Circuit()
gnd = VGround()
va = VNode("A")
vb = VNode("B")
circuit.add(VoltageSource(va, gnd, 5))
circuit.add(Resistor(va, vb, 5))
circuit.add(Inductor(vb, gnd, 50e-6))
f_list = []
v_list = []
for f in np.logspace(0, 6, 100):
  sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, AcMode(float(f)))
  f_list.append(f)
  v_list.append(sol[vb])
plt.title("High-pass filter (LR) [dB]")
plt.plot(f_list, 20 * np.log10(np.real(v_list)))
plt.xscale("log")
plt.show()

plt.title("High-pass filter (CR) [phase]")
plt.plot(f_list, (np.angle(v_list, deg=True) + 180) % 360 - 180)
plt.ylim(0, 90)
plt.xscale("log")
plt.show()

# 直列抵抗でためしてみる
circuit = Circuit()
gnd = VGround()
va = VNode("A")
vb = VNode("B")
circuit.add(VoltageSource(va, gnd, 1))
circuit.add(Resistor(va, vb, 1))
circuit.add(Resistor(vb, gnd, 3))
sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, DcMode())
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
  sol, _ = circuit.solve(sol, sol, 0.1, 0.1 * n, rand, DcMode())
  v_list.append(sol[vb])
plt.title(f"Capacitor (100 steps)")
plt.plot(v_list)
plt.show()

v_list = []
sol: Optional[Solution] = None
for n in range(100):
  sol, _, converged = circuit.newton(sol, sol, 0.1, 0.1 * n, rand, DcMode())
  v_list.append(sol[vb])
  assert converged
plt.title(f"Capacitor (100 steps, Newton)")
plt.plot(v_list)
plt.show()

v_list = []
sol: Optional[Solution] = None
for n in range(10000):
  sol, _ = circuit.solve(sol, sol, 0.001, 0.001 * n, rand, DcMode())
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
  sol, _ = circuit.solve(sol, sol, 0.1, 0.1 * n, rand, DcMode())
  v_list.append(sol[vb])
plt.title(f"Inductor (100 steps)")
plt.plot(v_list)
plt.show()

# ... いい感じ！
