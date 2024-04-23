# %%

from typing import Optional

import gradio as gr
import numpy as np
from matplotlib import pyplot as plt

from eleclove.components import Capacitor, CurrentSource, Inductor, Resistor
from eleclove.core import (Circuit, Component, Solution, VGround, VNode, VNodeFull)

class CustomResistor(Component):
  def __init__(self, pos: VNodeFull, neg: VNodeFull):
    self._pos = pos
    self._neg = neg

  def expand(self, sol, dt):
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

  def expand(self, sol, dt):
    i = self.value * np.random.randn()
    return [
        CurrentSource(self._pos, self._neg, i),
    ]

def simulate(time: float, resistor: float, capacitor: float, inductor: float, noise: float, hann: bool = False):
  circuit = Circuit()
  gnd = VGround()

  va = VNode("A")

  circuit.add(WhiteNoise(va, gnd, noise * 1e-6))
  circuit.add(CustomResistor(va, gnd))
  circuit.add(Resistor(va, gnd, resistor))
  circuit.add(Inductor(va, gnd, inductor * 1e-12))
  circuit.add(Capacitor(va, gnd, capacitor * 1e-15))
  circuit.add(CustomResistor(va, gnd))

  dt = 10e-15
  t_list = []
  va_list = []
  sol: Optional[Solution] = None
  for i in range(int(time * 1e-12 / dt)):
    sol = circuit.solve(sol, dt)
    t_list.append(i * dt)
    va_list.append(sol[va])

  va_list = np.array(va_list)
  window = np.hanning(len(va_list))

  freq = np.fft.fftfreq(len(va_list), d=dt)
  fft = np.fft.fft(va_list * window if hann else va_list)
  freq = freq[:len(freq) // 2]
  fft = fft[:len(fft) // 2]

  fig1, ax = plt.subplots()
  ax.plot(t_list, va_list)
  ax.set_xlabel("Time [s]")
  ax.set_ylabel("Voltage [V]")

  fig2, ax = plt.subplots()
  ax.loglog(freq, np.abs(fft))
  ax.set_xlabel("Frequency [Hz]")
  ax.set_ylabel("Magnitude")

  return fig1, fig2

def _main():
  # use global variable 'demo'
  # see: https://www.gradio.app/guides/developing-faster-with-reload-mode
  global demo

  demo = gr.Interface(
      fn=simulate,
      inputs=[
          gr.Slider(value=100, maximum=1000, label="Time [ps]"),
          gr.Slider(value=10.15, label="Resistor [Ω]"),
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
