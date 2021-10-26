import math
import numpy as np

# Physical model of a pendulum

# 2d vector
class Vec2:
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  def __add__(self, o):
    return Vec2(self.x + o.x, self.y + o.y)
  
  def __sub__(self, o):
    return self + (o * -1.0)
  
  def __mul__(self, o):
    return Vec2(self.x * o, self.y * o)
  
  def __truediv__(self, o):
    return self * (1.0 / o)
  
  def mag(self):
    return math.sqrt(self.x ** 2 + self.y ** 2)
  
  def normalize(self):
    return self / self.mag()
  
  def __str__(self):
    return "<{:.3f}, {:.3f}>".format(self.x, self.y)

# The path (simulated or real) of a pendulum
class Path:
  # t - time
  # x, y - position
  # err - error radius on position
  def __init__(self, dt, t, x, y, err = np.ndarray(0)):
    self.dt = dt
    self.t = t
    self.x = x
    self.y = y
    self.err = err

  # Get a Path from data stored in a csv file
  # CSV should have three columns (with two columns of headers).
  # For example:

  # mass A
  # time, x, y
  # 0, 0.0, 0.0
  # 0.1, 0.1, -0.25
  @staticmethod
  def from_csv(path):
    data = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=2, unpack=True, encoding=None)
    # calculate time interval
    dt = data[0][1] - data[0][0]
    return Path(dt, data[0], data[1], data[2])
  
  # Get the magnitude of position -- the length of pendulum (m)
  def length(self):
    # magnitude of pos
    length_sq = np.square(self.x) + np.square(self.y)
    length = np.sqrt(length_sq)
    if self.err.size == 0:
      return length, self.err
    # relative err in x^2 and y^2
    rel_err_x = 2.0 * np.abs(self.err / self.x)
    rel_err_y = 2.0 * np.abs(self.err / self.y)
    sum_err = (np.square(self.x) * rel_err_x) + (np.square(self.y) * rel_err_y)
    # relative error in sqrt(x^2 + y^2)
    rel_err = 0.5 * (sum_err / length_sq)
    new_err = rel_err * length
    return length, new_err
  
  # Get the angle of the position from the horizontal (rad)
  def angle_from_horizontal(self):
    return np.arctan2(self.y, self.x)
  
  # get the angle of the position from the negative vertical: the angle the pendulum is inclined from its resting position (rad)
  def angle_pendulum(self):
    # note: x and y flipped because we are doing angle to negative vertical
    div_term = self.x / -self.y
    angle = np.arctan2(self.x, -self.y)
    if self.err.size == 0:
      return angle, self.err
    # relative err in x, y
    rel_err_x = np.abs(self.err / self.x)
    rel_err_y = np.abs(self.err / self.y)
    # error in division
    err_div = (rel_err_x + rel_err_y) * np.abs(div_term)
    # error in arctan
    # err = err(div_term) * d/dx arctan(div_term) = err(div_term) * 1/(1 + err(div_term))
    new_err = err_div * (1.0 / (1.0 + div_term * div_term))
    return angle, new_err
  
  # Get velocity as t, dx/dt, dy/dt
  # length of returned values is one less than path length (due to numerical differentiation)
  def velocity(self):
    return self.t[1:], np.diff(self.x) * self.dt, np.diff(self.y) * self.dt
  
  # Get speed as t, |v|
  def velocity_mag(self):
    t, dx, dy = self.velocity()
    return t, np.sqrt(np.square(dx), np.square(dy))
  
  # do a fourier transform on the given data
  def fourier(self, data):
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data)) / self.dt
    return freqs[0:math.floor(len(data)/2)], w[0:math.floor(len(data)/2)]
  
  # get the primary frequency of a signal in data using a fourier transform
  def largest_freq(self, data):
    freqs, w = self.fourier(data)
    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    return freq

# A point with a mass, position, and velocity
class PointMass:
  # m - mass (kg)
  # x - position (m)
  # v - velocity (m/s)
  def __init__(self, m, x, v):
    self.m = m
    self.x = x
    self.v = v
  
  # apply a force (N) for a given amount of time (s)
  def apply_force(self, f, t):
    a = f / self.m
    # add accel to vel
    self.v += a * t
    self.x += self.v * t
  
  def __str__(self):
    return "mass: {} kg, x: {}, v: {}".format(self.m, self.x, self.v)

# acceleration due to gravity on earth (m/s^2)
ACCEL_GRAVITY = 9.80665

# An elastic pendulum system
class ElasticPendulum:
  # m - mass (kg)
  # le - spring equilibrium length (m)
  # k - spring constant (N/m)
  # l0 - initial spring starting length (m)
  # theta0 - initial starting angle from the vertical (rad)
  def __init__(self, m, k, le, pos):
    self.obj = PointMass(m, pos, Vec2(0.0, 0.0))
    self.k = k
    self.le = le
  
  def __str__(self):
    return "Elastic pendulum with k = {:.2f} N/m, equilibrium length le = {:.2f} m. Current state:\n{}".format(self.k, self.le, self.obj)

  # calculate force from gravity
  def force_gravity(self):
    return Vec2(0.0, -self.obj.m * ACCEL_GRAVITY)
  
  # calculate force from the spring
  def force_spring(self):
    # amount of compression
    dx = self.obj.x.mag() - self.le
    # Hooke's law
    return self.obj.x.normalize() * -self.k * dx
  
  def force_net(self):
    return self.force_gravity() + self.force_spring()
  
  # run a discrete time period
  def run(self, t):
    f = self.force_net()
    self.obj.apply_force(f, t)
  
  # simulate the pendulum's motion and return a Path
  # simulates for time [0, total_time)
  # pendulum motion is calculated at dt intervals
  # undersample_rate controls which calculated samples get added to return values
  #   1 = return every calculated sample, 2 = return every other sample, 3 = return every third sample, etc
  def simulate(self, total_time, dt, undersample_rate = 1):
    times = []
    x = []
    y = []
    t = 0.0
    i = 0
    while t < total_time:
      if i % undersample_rate == 0:
        times.append(t)
        x.append(self.obj.x.x)
        y.append(self.obj.x.y)
      self.run(dt)
      t += dt
      i += 1
  
    return Path(dt * undersample_rate, np.array(times), np.array(x), np.array(y))
  
