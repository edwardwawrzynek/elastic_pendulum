#! /usr/bin/env python3
import pendulum
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import math
import samples

# plotting utility methods

# plot an x, y graph with uncertainty bands
def plot_xy(ax, x, y, uncertain = np.ndarray(0)):
  if uncertain.size == 0:
    ax.plot(x, y)
  else:
    # FROM matplotlib error bands example
    # calculate normals via derivatives of splines
    tck, u = splprep([x, y], s=0)
    dx, dy = splev(u, tck, der=1)
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * uncertain
    yp = y + ny * uncertain
    xn = x - nx * uncertain
    yn = y - ny * uncertain

    vertices = np.block([[xp, xn[::-1]],
                        [yp, yn[::-1]]]).T
    codes = Path.LINETO * np.ones(len(vertices), dtype=Path.code_type)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)

    patch = PathPatch(path, facecolor='C0', edgecolor='none', alpha=0.3)

    ax.plot(x, y, linewidth=1)
    ax.add_patch(patch)

# plot a function of time with uncertainty
def plot_t_func(ax, t, val, err = np.ndarray(0)):
  if err.size != 0:
    ax.fill_between(t, val - err, val + err, alpha = 0.3)
  ax.plot(t, val, linewidth=1)

# Setup a plot of path's position on ax
def plot_pos(ax, path):
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  plot_xy(ax, path.x, path.y, path.err)
  # make sure y-axis includes origin at min
  ax.set_ylim(top=0.0)

# Setup a plot of path's length vs time on ax
def plot_length(ax, path):
  ax.set_xlabel('time (s)')
  ax.set_ylabel('length (m)')
  length, err = path.length()
  plot_t_func(ax, path.t, length, err)
  # make sure length included 0 at bottom
  ax.set_ylim(bottom=0.0)

# Setup a plot of path's angle vs time on ax
def plot_pendulum_angle(ax, path):
  ax.set_xlabel('time (s)')
  ax.set_ylabel('angle (rad)')
  angle, err = path.angle_pendulum()
  plot_t_func(ax, path.t, angle, err)

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

  fig2.set_size_inches(8.0, 4.8)
  fig2.savefig(directory + "/length.png")

  fig3.set_size_inches(8.0, 4.8)
  fig3.savefig(directory + "/angle.png")


# plot simulated + real data for a sample
def analyze_sample(sample):
  # load recorded data
  recorded_path = pendulum.Path.from_csv("data/" + sample["name"] + "/track.csv")
  recorded_path.err = np.repeat(samples.POS_UNCERTAINTY, recorded_path.t.size)
  # get starting pos from recorded sample
  start_pos = pendulum.Vec2(recorded_path.x[0], recorded_path.y[0])
  # simulate path
  p = pendulum.ElasticPendulum(sample["mass"], samples.SPRING_CONSTANT, sample["natural_length"], start_pos)
  simulated_path = p.simulate(recorded_path.t.size * recorded_path.dt, recorded_path.dt / 50, 20)
  # plot and save figures
  plot_sample_path(recorded_path, "figures/" + sample["name"] + "/recorded")
  plot_sample_path(simulated_path, "figures/" + sample["name"] + "/simulated")

  return recorded_path, simulated_path

# convert samples to np arrays of mass, angle frequency, length frequency
def samples_to_freqs(samples):
  mass = []
  angle_freq = []
  length_freq = []

  for sample in samples:
    mass.append(sample["mass"])
    path = sample["recorded"]
    angle, _ = path.angle_pendulum()
    angle_freq.append(path.largest_freq(angle))
    # in order to do a fourier transform on length, we need it centered on 0
    length, _ = path.length()
    length_zero_mean = length - np.mean(length)
    length_freq.append(path.largest_freq(length_zero_mean))
    freq, w = path.fourier(length_zero_mean)

    fig, ax = plt.subplots()
    ax.plot(freq, np.abs(w))
    ax.set_xlabel("Frequency (hz)")
    ax.set_ylabel("Magnitude (m)")
    fig.savefig("figures/ft_length" + str(sample["mass"]) + ".png")
  
  return np.array(mass), np.array(angle_freq), np.array(length_freq)

# plot mass vs frequency of angle oscillation
def analyze_samples(samples):
  mass, angle_freq, length_freq = samples_to_freqs(samples)
  
  # plot angular period
  fig_a, ax_a = plt.subplots()
  ax_a.set_title("Angle Period vs Mass")
  ax_a.set_xlabel("Pendulum Mass (kg)")
  ax_a.set_ylabel("Period of Angular Oscillation (s)")

  ax_a.plot(mass, 1.0 / angle_freq, 'o')
  fig_a.savefig("figures/angle_period.png")

  # plot length period
  fig_l, ax_l = plt.subplots()
  ax_a.set_title("Length Period vs Mass")
  ax_a.set_xlabel("Pendulum Mass (kg)")
  ax_a.set_ylabel("Period of Length Oscillation (s)")

  ax_a.plot(mass, 1.0 / length_freq, 'o')
  fig_a.savefig("figures/length_period.png")

def main():
  paths = []
  # load + analyze sample
  for sample in samples.samples:
    recorded, simulated = analyze_sample(sample)
    paths.append({
      "mass": sample["mass"],
      "recorded": recorded,
      "simulated": simulated
    })
  # do analysis on all paths
  analyze_samples(paths)

main()