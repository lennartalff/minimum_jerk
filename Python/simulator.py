import quadrocoptertrajectory as quadtraj
import math
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def rotation_between_vectors(v1, v2):
    n = np.cross(v1, v2)
    return Quaternion(w=np.dot(v1, v2), x=n[0], y=n[1], z=n[2])


class Ball():

    def __init__(self, p0, v0) -> None:
        self.p0 = p0
        self.v0 = v0
        self.gravity = np.array([0, 0, -9.81])

    def floor_collision_time(self):
        p_half = self.v0[2] / self.gravity[2]
        q = 2 * self.p0[2] / self.gravity[2]
        root = math.sqrt(p_half**2 - q)
        t1 = -p_half + root
        t2 = -p_half - root
        if t1 * t2 < 0.0:
            return max(t1, t2)
        if t1 > 0.0 and t2 > 0.0:
            raise RuntimeError("Bad initial condition for ball.")
        if t1 < 0.0 and t2 < 0.0:
            raise RuntimeError("Bad initial condition for ball.")

    def position(self, t):
        return 0.5 * self.gravity * t**2 + self.v0 * t + self.p0

    def velocity(self, t):
        return self.gravity * t + self.v0


class Body():

    def __init__(self, p0, v0, a0):
        self.p = p0
        self.v = v0
        self.a = a0
        self.q = Quaternion()
        self.gravity = np.array([0, 0, -9.81])

    def update_state(self, dt, thruster_force):
        self.a = thruster_force + self.gravity
        v_prev = self.v
        self.v = self.v + dt * self.a
        self.s = 0.5 (v_prev + self.v) * dt

    def position(self):
        return np.copy(self.p)

    def velocity(self):
        return np.copy(self.v)

    def acceleration(self):
        return np.copy(self.a)


ball = Ball(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
T_f = ball.floor_collision_time()

n_points = 100
t = np.linspace(0.0, T_f, n_points)
p_ball = np.zeros([n_points, 3])
for i in range(n_points):
    p_ball[i, :] = ball.position(t[i])

for i in range(3):
    plt.plot(t, p_ball[:, i])
plt.show()

# Define the trajectory starting state:
pos0 = [0, 0, 0.5]  #position
vel0 = [0, 0, 0]  #velocity
acc0 = [0, 0, 0]  #acceleration
