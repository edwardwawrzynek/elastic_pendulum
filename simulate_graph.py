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

runs = [
  {
    "mass": 0.500, 
    "start_pos": pendulum.Vec2(0.1936689155, -0.3619377580),
    "natural_length": 0.148
  },
  {
    "mass": 0.400, 
    "start_pos": pendulum.Vec2(0, 0),
    "natural_length": 0.094
  },
  {
    "mass": 0.700,
    "start_pos": pendulum.Vec2(0, 0),
    "natural_length":0.0
  },
  {
    "mass": 0.900,
    "start_pos": pendulum.Vec2(0, 0),
    "natural_length": 0.148
  }
]

# Natural length of the spring (m)
NATURAL_LENGTH = 0.148
# Mass of weight (kg)
MASS = 0.500

SPRING_CONSTANT = 30.2970062

p = pendulum.ElasticPendulum(MASS, SPRING_CONSTANT, NATURAL_LENGTH, pendulum.Vec2(0.1936689155, -0.3619377580))
path = p.simulate(20.0, 0.001, 20)
#path = pendulum.Path.from_csv("data/m500g/track.csv")

# graph
fig1, ax = plt.subplots()
fig2, ax_l = plt.subplots()
fig3, ax_p = plt.subplots()
#fig4, ax_f = plt.subplots()

ax.set_title("Path")
plot_pos(ax, path)
ax_l.set_title("Spring Length vs Time")
plot_length(ax_l, path)
ax_p.set_title("Spring Angle vs Time")
plot_pendulum_angle(ax_p, path)

# find the largest frequency component of angle (using fourier transform)
#w = np.fft.fft(path.angle_pendulum())
#freqs = np.fft.fftfreq(len(path.angle_pendulum()))
#idx = np.argmax(np.abs(w))
#freq = freqs[idx]
#freq_hz = abs(freq / path.dt)
#print(1.0 / freq_hz)

#ax_f.set_title("Fourier Transform of Spring Angle")
#ax_f.set_xlabel('Frequency (hz)')
#ax_f.set_ylabel('Amplitude (rad)')
#ax_f.plot((freqs / path.dt)[0:math.floor(len(w)/2)], np.abs(w)[0:math.floor(len(w)/2)])

plt.show()