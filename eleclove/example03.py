# %%

from functools import cache

from attr import dataclass
import jax
import jax.numpy as jnp
import optuna
from matplotlib import pyplot as plt

from eleclove.components import Capacitor, Inductor, Resistor, VoltageSource
from eleclove.core import AcMode, Circuit, Rand, VGround, VNode
from eleclove.utils import NPValue, run_on_cpu

@dataclass
class Params:
  r_short: NPValue
  l_short: NPValue
  c_short: NPValue
  r_long: NPValue
  l_long: NPValue
  c_long: NPValue

def calc_characteristics(p: Params):
  rand = Rand.seed(137)
  circuit = Circuit()
  gnd = VGround()
  va = VNode("A")
  vb = VNode("B")
  vc = VNode("C")
  vd = VNode("D")

  circuit.add(Resistor(va, vb, p.r_short))
  circuit.add(Inductor(va, gnd, p.l_short * 1e-12))

  circuit.add(VoltageSource(vb, gnd, 1))
  circuit.add(Capacitor(vb, gnd, p.c_short * 1e-15))

  circuit.add(Inductor(vb, vc, p.l_long * 1e-12))
  circuit.add(Resistor(vc, vd, p.r_long))
  circuit.add(Capacitor(vd, gnd, p.c_long * 1e-15))

  def analyze(f: NPValue):
    sol, _ = circuit.solve(None, None, 0.1, 0.0, rand, AcMode(f))
    return sol[vb]

  f_list = jnp.logspace(11, 12, 1000)  # 0.1 THz から 1 THz まで
  return jax.vmap(analyze)(f_list)

@cache
def ground_truth():
  raise NotImplementedError("This is a placeholder for the ground truth.")

@jax.jit
def score(x: Params):
  return jnp.linalg.norm(calc_characteristics(x) - ground_truth())

def objective(trial: optuna.Trial):
  return score(
      Params(
          r_short=trial.suggest_float('r_short', 0.1, 100.0),
          l_short=trial.suggest_float('l_short', 0.1, 100.0),
          c_short=trial.suggest_float('c_short', 0.1, 100.0),
          r_long=trial.suggest_float('r_long', 0.1, 100.0),
          l_long=trial.suggest_float('l_long', 0.1, 100.0),
          c_long=trial.suggest_float('c_long', 0.1, 100.0),
      ))

# Optunaを使ってパラメーター最適化を実行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

print("Best trial:")
best_trial = study.best_trial
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
  print(f"    {key}: {value}")
