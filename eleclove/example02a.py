# %%

# AC 解析とパラメーター推定の実装例
# AC 特性から、等価回路のパラメーターを推定する

# 複数の周波数に対してシミュレーションを行うためには jax.vmap か jax.lax.map を使うと並列化できる
# メモリアクセスを考えると多分 jax.vmap のほうが速い（特に GPU を使う場合）と思うので、そうすることにする

from functools import cache

import jax
import jax.numpy as jnp
import jax.scipy.optimize as optimize
from matplotlib import pyplot as plt

from eleclove.components import Capacitor, Resistor, VoltageSource
from eleclove.core import AcMode, Circuit, Rand, VGround, VNode
from eleclove.utils import NPValue, run_on_cpu

def calc_characteristics(R: NPValue, C: NPValue):
  rand = Rand.seed(137)
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  circuit.add(VoltageSource(va, gnd, 1))
  circuit.add(Capacitor(va, vb, C * 1e-6))
  circuit.add(Resistor(vb, gnd, R))

  def analyze(f: NPValue):
    sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, AcMode(f))
    return sol[vb]

  f_list = jnp.logspace(0, 6, 1000)
  return jax.vmap(analyze)(f_list)

@cache
def ground_truth():
  return calc_characteristics(1.34, 4.53)

@jax.jit
def score(x):
  return jnp.linalg.norm(calc_characteristics(x[0], x[1]) - ground_truth())

if __name__ == "__main__":
  run_on_cpu()

  # R, C の値を推定
  initial_guess = jnp.array([1.0, 1.0])
  result = optimize.minimize(score, initial_guess, method="BFGS")  # 他の方法に対応していないらしい

  print(result)
  print("r, c:    ", result.x)
  print("score:   ", result.fun)
  print("success: ", result.success)

  rs = jnp.linspace(0.1, 5.0, 50)
  cs = jnp.linspace(0.1, 10.0, 50)
  scores = jax.vmap(lambda r: jax.vmap(lambda c: score(jnp.array([r, c])))(cs))(rs)

  plt.contourf(rs, cs, scores, levels=100)
  plt.plot(result.x[0], result.x[1], "ro")
  plt.colorbar()
  plt.show()

  # ... うまくいかないのは、多分最適化方法のせい
