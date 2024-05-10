# %%

from eleclove.example01 import example01
from eleclove.utils import run_on_cpu

if __name__ == "__main__":
  run_on_cpu()
  example01(100000, 100, 50, 5, 1, False)
