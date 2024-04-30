# %%

# あとで整理する
# まずは性能を無視して開発して、あとから最適化する
# TODO: ニュートン法とか使う、収束判定、可変時間ステップ

from dataclasses import dataclass
from functools import cache
from typing import Any, Optional, Protocol, Sequence, TypeGuard, Union

import jax.numpy as jnp
import jax.random
import numpy as np
from jax.tree_util import register_pytree_node_class

from eleclove.utils import KeyArray, NPArray, NPBool, NPValue, hex_id

MAX_ITER = 100
RELTOL_V = 1e-3
ABSTOL_V = 1e-6

EqNode = Union["VNode", "INode"]
EqNodeFull = Union["VNode", "INode", "VGround"]
VNodeFull = Union["VNode", "VGround"]

class VGround:
  """ グラウンド電圧 """
  def __str__(self):
    return "VGround"

class VNode:
  """ 電圧ノード (JS における Symbol のように使う) """
  def __init__(self, name: Optional[str] = None):
    self._name = name

  def __str__(self):
    if self._name is not None:
      return f"VNode({hex_id(self)}/{self._name})"
    else:
      return f"VNode({hex_id(self)})"

class INode:
  """ 電流ノード (JS における Symbol のように使う) """
  def __init__(self, name: Optional[str] = None):
    self._name = name

  def __str__(self):
    if self._name is not None:
      return f"INode({hex_id(self)}/{self._name})"
    else:
      return f"INode({hex_id(self)})"

class LinearEqs:
  """ EqNode でアクセスできる連立線形方程式 """
  def __init__(self):
    self._names: dict[EqNode, int] = {}  # 順序が保証された OrderedDict と等価 (Python 3.7 以降)
    self._a: list[tuple[int, int, Any]] = []
    self._b: list[tuple[int, Any]] = []

  def _node_index(self, node: EqNode) -> int:
    if node not in self._names:
      self._names[node] = len(self._names)
    return self._names[node]

  def add_a(self, i: EqNodeFull, j: EqNodeFull, value: Any):
    if isinstance(i, VGround) or isinstance(j, VGround): return
    self._a.append((self._node_index(i), self._node_index(j), value))

  def add_b(self, i: EqNodeFull, value: Any):
    if isinstance(i, VGround): return
    self._b.append((self._node_index(i), value))

  def matrices(self):
    n = len(self._names)
    a = jnp.zeros((n, n))
    b = jnp.zeros(n)
    for i, j, value in self._a:
      a = a.at[i, j].add(value)
    for i, value in self._b:
      b = b.at[i].add(value)

    return a, b

  def solve(self):
    a, b = self.matrices()
    x = jnp.linalg.solve(a, b)
    return Solution(x, list(self._names.keys()))

  def __str__(self):
    # TODO: きれいに表示する
    a, b = self.matrices()
    names = map(str, self._names.keys())
    return f"LinearEqs({list(names)}\na=\n{a}, \nb={b}, \n)"

@register_pytree_node_class
class Solution:
  """ EqNode でアクセスできる解一覧 """
  def __init__(self, values: NPArray, names: list[EqNode] | dict[EqNode, int]):
    assert values.ndim == 1 or values.ndim == 2
    assert values.shape[values.ndim - 1] == len(names)
    self._values = values
    if isinstance(names, list):
      self._names = {name: i for i, name in enumerate(names)}
    else:
      self._names = names

  def to_numpy(self):
    return Solution(np.array(self._values), self._names)

  @staticmethod
  def is_converged(prev: "Solution", crnt: "Solution"):
    # TODO: 電流も判定に含めるべきだし、ノードとして設定されていない電圧・電流も含めるべき
    # TODO: 非線形特性の素子では、あらためて今の電圧から推定される電流との差を確認したほうがいい気がする
    assert prev._values.ndim == 1
    assert crnt._values.ndim == 1
    v_prev = []
    v_crnt = []
    for name in prev._names:
      if isinstance(name, VNode):
        v_prev.append(prev[name])
        v_crnt.append(crnt[name])
    return jnp.allclose(jnp.array(v_crnt), jnp.array(v_prev), rtol=RELTOL_V, atol=ABSTOL_V)

  def __getitem__(self, key: EqNodeFull):
    if isinstance(key, VGround): return 0
    if self._values.ndim == 1:
      return self._values[self._names[key]]
    if self._values.ndim == 2:
      return self._values[:, self._names[key]]
    raise ValueError("Invalid shape")

  def __str__(self):
    lines: list[str] = []
    for name, value in zip(self._names, self._values):
      lines.append(f"{name}: {value}")
    return "\n".join(lines)

  def tree_flatten(self):
    return (self._values,), self._names

  @classmethod
  def tree_unflatten(cls, aux, children):
    return cls(children[0], aux)

@register_pytree_node_class
class Rand:
  def __init__(self, key: KeyArray):
    self._key = key

  @property
  def key(self):
    return self._key

  @classmethod
  def seed(cls, seed: int):
    return cls(jax.random.PRNGKey(seed))

  def next_key(self):
    _key, self._key = jax.random.split(self._key)
    return _key

  def randn(self) -> NPArray:
    return jax.random.normal(self.next_key())

  def tree_flatten(self):
    return (self._key,), None

  @classmethod
  def tree_unflatten(cls, aux, children):
    return cls(children[0])

@dataclass
class TransientState:
  sol_prev: Optional[Solution]  # 前の時間ステップの解
  sol_crnt: Optional[Solution]  # ニュートン法で反復中の今の時間ステップの解
  dt: NPValue
  t: NPValue
  rand: Rand

class Element(Protocol):
  def stamp(self, eqs: LinearEqs, state: TransientState):
    ...  # Jax JIT のトレーシングに対応した処理を書く

class Component(Protocol):
  def expand(self, state: TransientState) -> Sequence[Union["Element", "Component"]]:
    ...  # Jax JIT のトレーシングに対応した処理を書く

def is_element(obj: Any) -> TypeGuard[Element]:
  return hasattr(obj, "stamp")

def is_component(obj: Any) -> TypeGuard[Component]:
  return hasattr(obj, "expand")

class Circuit:
  def __init__(self):
    self._children: list[Element | Component] = []

  def add(self, child: Element | Component):
    self._children.append(child)
    self._solve_jit.cache_clear()
    self._newton_jit.cache_clear()
    self._transient_jit.cache_clear()

  def solve(self, sol_prev: Optional[Solution], sol_crnt: Optional[Solution], dt: NPValue, t: NPValue, rand: Rand) -> tuple[Solution, Rand]:
    """ 線形方程式を解く。収束するまで解を更新したいときは newton メソッドを使う。 """

    # 何通りの JIT コンパイルが行われたか確認する
    # print(self._jitted_solve._cache_size())

    return self._solve_jit()(sol_prev, sol_crnt, dt, t, rand)

  @cache
  def _solve_jit(self):
    return jax.jit(self._solve)

  def _solve(self, sol_prev: Optional[Solution], sol_crnt: Optional[Solution], dt: NPValue, t: NPValue, rand: Rand):
    eqs = LinearEqs()
    state = TransientState(sol_prev, sol_crnt, dt, t, rand)

    queue = self._children.copy()
    while queue:
      child = queue.pop(0)
      if is_element(child):
        child.stamp(eqs, state)
      if is_component(child):
        for child in child.expand(state):
          queue.append(child)

    return eqs.solve(), rand

  def newton(self, sol_prev: Optional[Solution], sol_crnt: Optional[Solution], dt: NPValue, t: NPValue, rand: Rand) -> tuple[Solution, Rand, NPBool]:
    """ 非線形方程式を解く。 """

    if sol_crnt is None: sol_crnt, rand = self.solve(sol_prev, sol_crnt, dt, t, rand)
    return self._newton_jit()(sol_prev, sol_crnt, dt, t, rand)

  @cache
  def _newton_jit(self):
    return jax.jit(self._newton)

  def _newton(self, sol_prev: Optional[Solution], sol_crnt: Solution, dt: NPValue, t: NPValue, rand: Rand) -> tuple[Solution, Rand, NPBool]:
    # ニュートン法での solve 呼び出しでは、毎回同じ乱数シードを使用する

    def check(state):
      prev, crnt, iters, _, _, _ = state
      # jax.debug.print("{prev} -> {crnt}", prev=prev._values, crnt=crnt._values)
      return jnp.logical_and(jnp.logical_not(Solution.is_converged(prev, crnt)), iters < MAX_ITER)

    def step(state):
      prev, crnt, iters, dt, t, rand = state
      prev = crnt
      crnt, _ = self.solve(sol_prev, crnt, dt, t, rand)
      iters += 1
      return prev, crnt, iters, dt, t, rand

    crnt, rand_next = self.solve(sol_prev, sol_crnt, dt, t, rand)
    state = (sol_crnt, crnt, jnp.array(1), dt, t, rand)

    state = jax.lax.while_loop(check, step, state)

    _, crnt, iters, _, _, _ = state
    # jax.debug.print("Newton: {iters} iterations", iters=iters)
    return crnt, rand_next, iters < MAX_ITER

  def transient(self, sol: Optional[Solution], dt: float, t: NPArray, rand: Rand) -> tuple[Solution, Rand, NPBool]:
    if sol is None:

      def continue_transient(sol0, dt, t, rand):
        sols, rand, converged = self.transient(sol0, dt, t[1:], rand)
        sols._values = jnp.concatenate((jnp.expand_dims(sol0._values, axis=0), sols._values), axis=0)
        return sols, rand, converged

      def continue_fake(sol0, t, rand):
        # 収束した場合としなかった場合で同じ形の値を返さないといけない (jax.lax.cond) ので、ゼロで埋めた値を返す
        sol0._values = jnp.zeros((len(t), len(sol0._values)))
        return sol0, rand, False

      sol0, rand, converged = self.newton(sol, sol, dt, 0, rand)
      sols, rand, converged = jax.lax.cond(
          converged,
          lambda: continue_transient(sol0, dt, t, rand),
          lambda: continue_fake(sol0, t, rand),
      )
      return sols, rand, converged

    dt_t = jnp.stack((jnp.full(t.shape, dt), t), axis=-1)

    (_, rand, converged), sols = self._transient_jit()(sol, dt_t, rand)
    return sols, rand, converged

  @cache
  def _transient_jit(self):
    return jax.jit(self._transient)

  def _transient(self, sol: Solution, dt_t: NPArray, rand: Rand):
    def step(carry, dt_t):
      sol, rand, converged = carry
      dt, t = dt_t
      sol, rand, converged = jax.lax.cond(
          converged,
          lambda: self.newton(sol, sol, dt, t, rand),
          lambda: (sol, rand, converged),
      )
      return (sol, rand, converged), sol

    return jax.lax.scan(step, (sol, rand, True), dt_t)
