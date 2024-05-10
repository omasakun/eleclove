# %%

# Optunaを使ってパラメーター最適化を実行する例

import jax
import jax.numpy as jnp
import optuna
from matplotlib import pyplot as plt

from eleclove.example02a import score
from eleclove.utils import run_on_cpu

def objective(trial: optuna.Trial):
  R = trial.suggest_float('R', 0.1, 5.0)
  C = trial.suggest_float('C', 0.1, 10.0)
  return float(score(jnp.array([R, C])))

if __name__ == "__main__":
  run_on_cpu()

  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=200)

  print("Best trial:")
  best_trial = study.best_trial
  print("  Value: ", best_trial.value)
  print("  Params: ")
  for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

  # 結果の可視化
  rs = jnp.linspace(0.1, 5.0, 50)
  cs = jnp.linspace(0.1, 10.0, 50)
  scores = jax.vmap(lambda r: jax.vmap(lambda c: score(jnp.array([r, c])))(cs))(rs)

  plt.contourf(rs, cs, scores, levels=100)

  # plot all trials
  for trial in study.trials:
    plt.plot(trial.params['R'], trial.params['C'], "y.", markersize=2)

  plt.plot(best_trial.params['R'], best_trial.params['C'], "ro")
  plt.colorbar()
  plt.show()

  # ... Optunaでは、ありうる一つの最適に近いパラメーターが見つかった
