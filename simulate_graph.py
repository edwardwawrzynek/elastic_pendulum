#! /usr/bin/env python3
import pendulum
import numpy as np
import matplotlib.pyplot as plt
import math
import samples

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

# plot and save three path graphs
def plot_sample_path(path, directory):
  fig1, ax_p = plt.subplots()
  fig2, ax_l = plt.subplots()
  fig3, ax_a = plt.subplots()

  ax_p.set_title("Path")
  plot_pos(ax_p, path)
  ax_l.set_title("Spring Length vs Time")
  plot_length(ax_l, path)
  ax_a.set_title("Spring Angle vs Time")
  plot_pendulum_angle(ax_a, path)

  fig1.savefig(directory + "/path.png")
  fig2.savefig(directory + "/length.png")
  fig3.savefig(directory + "/angle.png")


# plot simulated + real data for a sample
def analyze_sample(sample):
  # load recorded data
  recorded_path = pendulum.Path.from_csv("data/" + sample["name"] + "/track.csv")
  # get starting pos from recorded sample
  start_pos = pendulum.Vec2(recorded_path.x[0], recorded_path.y[0])
  # simulate path
  p = pendulum.ElasticPendulum(sample["mass"], samples.SPRING_CONSTANT, sample["natural_length"], start_pos)
  simulated_path = p.simulate(recorded_path.t.size * recorded_path.dt, recorded_path.dt / 50, 20)
  # plot and save figures
  plot_sample_path(recorded_path, "figures/" + sample["name"] + "/recorded")
  plot_sample_path(simulated_path, "figures/" + sample["name"] + "/simulated")

for sample in samples.samples:
  analyze_sample(sample)

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