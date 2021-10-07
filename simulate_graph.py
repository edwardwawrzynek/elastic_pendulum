#! /usr/bin/env python3
import pendulum
import numpy as np
import matplotlib.pyplot as plt

p = pendulum.ElasticPendulum(0.2, 4, 1.0, 1.3, -0.5)
print(p)
path = p.simulate(10.0, 0.001, 20)

# graph
fig, ax = plt.subplots()
ax.grid(linestyle='--', linewidth=0.5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

# draw path
ax.plot(path.x, path.y)
# mark start point
#ax.plot(x[0], y[0], 'o')
ax.text(path.x[0], path.y[0], "$t_0$", ha = 'center', va = 'bottom')

plt.show()