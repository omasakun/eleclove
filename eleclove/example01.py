# %%

import numpy as np
from matplotlib import pyplot as plt

from eleclove.components import Capacitor, CurrentSource, Inductor, Resistor
from eleclove.core import Circuit, Component, Rand, VGround, VNode, VNodeFull
from eleclove.utils import run_on_cpu

class CustomResistor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull):
    self._pos = pos
    self._neg = neg

  def expand(self, state):
    sol = state.sol
    v = 0 if sol is None else sol[self._pos] - sol[self._neg]
    i = -0.05 * v + 0.1 * v**3
    return [
        CurrentSource(self._pos, self._neg, i),
    ]

class WhiteNoise(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull, value=1e-6):
    self._pos = pos
    self._neg = neg
    self.value = value

  def expand(self, state):
    rand = state.rand
    i = rand.randn() * self.value
    return [
        CurrentSource(self._pos, self._neg, i),
    ]

def example01(time: float, resistor: float, capacitor: float, inductor: float, noise: float, hann: bool = False):
  circuit = Circuit()
  gnd = VGround()

  va = VNode("A")

  circuit.add(WhiteNoise(va, gnd, noise * 1e-6))
  circuit.add(CustomResistor(va, gnd))
  circuit.add(Resistor(va, gnd, resistor))
  circuit.add(Inductor(va, gnd, inductor * 1e-12))
  circuit.add(Capacitor(va, gnd, capacitor * 1e-15))
  circuit.add(CustomResistor(va, gnd))

  rand = Rand.seed(137)

  dt = 10e-15
  t = np.arange(0, time * 1e-12, dt)
  sol, rand = circuit.solve(None, dt, 0, rand)
  sol, rand = circuit.transient(sol, dt, t, rand)

  # jax array の値の取得は遅いので numpy array に変換する
  # あと、 python list から jax array への変換も遅かった
  va_list = np.array(sol[va])

  window = np.hanning(len(va_list))

  freq = np.fft.fftfreq(len(va_list), d=dt)
  fft = np.fft.fft(va_list * window if hann else va_list)
  freq = freq[:len(freq) // 2]
  fft = fft[:len(fft) // 2]

  fig1, ax = plt.subplots()
  ax.set_title(f"Waveform ({len(va_list)} samples)")
  ax.plot(t * 1e12, va_list)
  ax.set_xlabel("Time [ps]")
  ax.set_ylabel("Voltage [V]")

  fig2, ax = plt.subplots()
  ax.set_title(f"Peak: {freq[np.argmax(np.abs(fft))]*1e-12:.3f} THz")
  ax.loglog(freq, np.abs(fft))
  ax.set_xlabel("Frequency [Hz]")
  ax.set_ylabel("Magnitude")

  return fig1, fig2

def _main():
  import gradio as gr

  # use global variable 'demo'
  # see: https://www.gradio.app/guides/developing-faster-with-reload-mode
  global demo

  run_on_cpu()

  demo = gr.Interface(
      fn=example01,
      inputs=[
          gr.Slider(value=100, maximum=1000, label="Time [ps]"),
          gr.Slider(value=100, maximum=10000, label="Resistor [Ω]"),
          gr.Slider(value=50, label="Capacitor [pF]"),
          gr.Slider(value=5, label="Inductor [fH]"),
          gr.Slider(value=1, label="Noise [μV]"),
          gr.Checkbox(label="Hann window"),
      ],
      outputs=[
          gr.Plot(label="waveform"),
          gr.Plot(label="spectrogram"),
      ],
      live=True,
  )
  demo.launch()

if __name__ == "__main__": _main()
