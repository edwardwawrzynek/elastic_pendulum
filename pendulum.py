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
  def __init__(self, dt, t, x, y):
    self.dt = dt
    self.t = t
    self.x = x
    self.y = y

  # Get a Path from data stored in a csv file
  # CSV should have three columns (with headers).
  # For example:

  # time, x, y
  # 0, 0.0, 0.0
  # 0.1, 0.1, -0.25
  @staticmethod
  def from_csv(path):
    data = np.genfromtxt(path, dtype=None, delimiter=',', skip_header=1, unpack=True)
    # calculate time interval
    dt = data[0][1] - data[0][0]
    return Path(dt, data[0], data[1], data[2])
  
  # convert a time to the nearest index
  def time_to_index(self, time):
    return math.floor(time / self.dt)

  # get position at a given sample index
  def sample_pos(self, index):
    return Vec2(self.x[index], self.y[index])

  # get velocity at a given sample index
  def sample_velocity(self, index):
    return self.sample_pos(index) - self.sample_pos(index - 1)
  
  # get velocity at a given time
  def velocity(self, t):
    return self.sample_velocity(self.time_to_index(t))


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
  def __init__(self, m, k, le, l0, theta0):
    # calculate initial position (spring starts at origin)
    pos = (Vec2(math.sin(theta0), -math.cos(theta0)) * l0)

    self.obj = PointMass(m, pos, Vec2(0.0, 0.0))
    self.k = k
    self.le = le
  
  def __str__(self):
    return "Pendulum with k = {:.2f} N/m, equilibrium length le = {:.2f} m. Current state:\n{}".format(self.k, self.le, self.obj)

  # calculate force from gravity
  def force_gravity(self):
    return Vec2(0.0, -self.obj.m * ACCEL_GRAVITY)
  
  # calculate force from the spring
  def force_spring(self):
    # amount of compression
    dx = self.obj.x.mag() - self.le
    # Hooke's law
    return self.obj.x.normalize() * (-self.k * dx)
  
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
  
