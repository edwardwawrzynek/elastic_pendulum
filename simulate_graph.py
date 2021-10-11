#! /usr/bin/env python3
import pendulum
import numpy as np
import matplotlib.pyplot as plt
import math

# plotting utility methods

# Setup a plot of path's position on ax
def plot_pos(ax, path):
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.plot(path.x, path.y)
  # make sure y-axis includes origin at min
  ax.set_ylim(top=0.0)

# Setup a plot of path's length vs time on ax
def plot_length(ax, path):
  ax.set_xlabel('time (s)')
  ax.set_ylabel('length (m)')
  ax.plot(path.t, path.length())
  # make sure length included 0 at bottom
  ax.set_ylim(bottom=0.0)

# Setup a plot of path's angle vs time on ax
def plot_pendulum_angle(ax, path):
  ax.set_xlabel('time (s)')
  ax.set_ylabel('angle (rad)')
  ax.plot(path.t, path.angle_pendulum())


p = pendulum.ElasticPendulum(0.2, 4.00, 2.0, 1.5, -3.1415926 / 8)
#path = p.simulate(8.5, 0.001, 20)
path = pendulum.Path.from_csv("data/test.csv")

# graph
fig1, ax = plt.subplots()
fig2, ax_l = plt.subplots()
fig3, ax_p = plt.subplots()
fig4, ax_f = plt.subplots()

ax.set_title("Path")
plot_pos(ax, path)
ax_l.set_title("Spring Length vs Time")
plot_length(ax_l, path)
ax_p.set_title("Spring Angle vs Time")
plot_pendulum_angle(ax_p, path)

# find the largest frequency component of angle (using fourier transform)
w = np.fft.fft(path.angle_pendulum())
freqs = np.fft.fftfreq(len(path.angle_pendulum()))
idx = np.argmax(np.abs(w))
freq = freqs[idx]
freq_hz = abs(freq / path.dt)
print(1.0 / freq_hz)

ax_f.set_title("Fourier Transform of Spring Angle")
ax_f.set_xlabel('Frequency (hz)')
ax_f.set_ylabel('Amplitude (rad)')
ax_f.plot((freqs / path.dt)[0:math.floor(len(w)/2)], np.abs(w)[0:math.floor(len(w)/2)])

plt.show()