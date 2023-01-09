import quadrocoptertrajectory as quadtraj
import math
import numpy as np
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
import time
from typing import List

N_TIMESTEPS = 20
N_FORCESTEPS = 5
N_NORMALS_RADIAL = 3
RADIAL_ANGLE = math.radians(60.0)  # half the opening angle
N_NORMALS_TANG = 10
TANG_ANGLE = 2 * math.pi
F_MIN = 5.0
F_MAX = 25.0
OMEGA_MAX = 20.0
GRAVITY = np.array([0.0, 0.0, 0.0])


@dataclass
class FeasibilityCounter:
    feasible: int = 0
    indeterminable: int = 0
    thrust_low: int = 0
    thrust_high: int = 0
    body_rate: int = 0
    unknown: int = 0

    def count(self, feasibility):
        if feasibility == quadtraj.InputFeasibilityResult.Feasible:
            self.feasible += 1
        elif feasibility == quadtraj.InputFeasibilityResult.Indeterminable:
            self.indeterminable += 1
        elif feasibility == quadtraj.InputFeasibilityResult.InfeasibleThrustLow:
            self.thrust_low += 1
        elif feasibility == quadtraj.InputFeasibilityResult.InfeasibleThrustHigh:
            self.thrust_high += 1
        else:
            self.unknown += 1

    def print(self):
        total = self.feasible + self.indeterminable + self.thrust_low + self.thrust_high + self.body_rate + self.unknown
        if total == 0:
            print('Nothing counted.')
        else:
            total /= 100.0
            print(
                f'Feasible: {self.feasible} ({self.feasible/total:.2f} %)\n'
                f'Indeterminable: {self.indeterminable} ({self.indeterminable/total:.2f} %)\n'
                f'Thrust Low: {self.thrust_low} ({self.thrust_low / total:.2f} %)\n'
                f'Thrust High: {self.thrust_high} ({self.thrust_high / total:.2f} %)\n'
                f'Body Rate: {self.body_rate} ({self.body_rate / total:.2f} %)\n'
                f'Unknown: {self.unknown} ({self.unknown / total:.2f} %)')


@dataclass
class Trajectory:
    trajectory: quadtraj.RapidTrajectory
    q: Quaternion  # rotation from ball oriented frame to map
    t0_abs: float
    tf: float
    ball_position: np.array
    feasible: bool = False

    def t_vec(self, n_samples):
        return np.linspace(0.0, self.tf, n_samples)

    def t_vec_abs(self, n_samples):
        return np.linspace(self.t0_abs, self.t0_abs + self.tf, n_samples)

    def p_vec(self, t_vec):
        p_vec = np.zeros([len(t_vec), 3], dtype=float)
        for i, t in enumerate(t_vec.tolist()):
            p_vec[i, :] = self.q.inverse.rotate(
                self.trajectory.get_position(t))
        return p_vec

    def v_vec(self, t_vec):
        v_vec = np.zeros([len(t_vec), 3], dtype=float)
        for i, t in enumerate(t_vec.tolist()):
            v_vec[i, :] = self.q.inverse.rotate(
                self.trajectory.get_velocity(t))
        return v_vec

    def a_vec(self, t_vec):
        a_vec = np.zeros([len(t_vec), 3], dtype=float)
        for i, t in enumerate(t_vec.tolist()):
            a_vec[i, :] = self.q.inverse.rotate(
                self.trajectory.get_acceleration(t))
        return a_vec

    def p_ball(self):
        return self.q.inverse.rotate(self.ball_position)

    def thrust(self, t_abs):
        return self.q.inverse.rotate(
            self.trajectory.get_thrust_vector(t_abs - self.t0_abs))


def rotation_between_vectors(v1, v2) -> Quaternion:
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    n = n / n_norm
    angle = math.atan2(n_norm, np.dot(v1, v2))
    c = math.cos(angle * 0.5)
    s = math.sin(angle * 0.5)
    n *= s
    q = Quaternion(w=c, x=n[0], y=n[1], z=n[2])
    return q

class Ring():
    def __init__(self, p0, v0):
        self.p0 = p0
        self.v0 = v0
        self.z_floor = -1.5
    
    def velocity(self, t):
        return self.v0

    def position(self, t):
        return self.p0 + self.v0 * t
    
    def floor_collision_time(self, z_offset):
        return (self.z_floor + z_offset + self.p0[2]) / self.v0[2]


class Ball():

    def __init__(self, p0, v0) -> None:
        self.p0 = p0
        self.v0 = v0
        self.m = 0.0027
        self.d = 0.04
        self.A = 1 / 4 * math.pi * self.d**2
        self.C_d = 0.5
        self.air_density = 1.2041
        self.drag_coeff = 0.5 * self.air_density * self.A * self.C_d
        self.gravity = np.array([0, 0, 9.81])

        self.v_inf = self.init_v_inf()
        self.t_up = self.init_t_up()
        self.z_up = self.position_z(self.t_up)

    def init_v_inf(self):
        return math.sqrt(self.gravity[2] * self.m / self.drag_coeff)

    def init_t_up(self):
        t_up = self.v_inf / self.gravity[2] * math.atan(
            self.v0[2] / self.v_inf)
        return t_up

    def velocity_z(self, t):
        if t <= self.t_up:
            return self.v_inf * math.tan(
                math.atan((self.v0[2] - self.gravity[2] * t) / self.v_inf))
        else:
            p = 2 * self.drag_coeff / self.m * self.v_inf
            e_term = math.exp(p * (t - self.t_up))
            return -self.v_inf * (e_term - 1) / (e_term + 1)

    def position_z(self, t):
        if t <= self.t_up:
            ln1 = math.log(
                math.cos(self.gravity[2] * (self.t_up - t) / self.v_inf))
            ln2 = math.log(math.cos(self.gravity[2] * self.t_up / self.v_inf))
            return self.v_inf**2 / self.gravity[2] * (ln1 - ln2)
        else:
            p = 2 * self.drag_coeff / self.m * self.v_inf
            ln = math.log(0.5 * math.exp(-p * (t - self.t_up)) + 0.5)
            return self.z_up - self.v_inf * (
                t - self.t_up) - self.m / self.drag_coeff * ln

    def velocity_x(self, t):
        return 1 / (self.drag_coeff / self.m * t + 1 / self.v0[0])

    def position_x(self, t):
        ln = math.log(self.drag_coeff / self.m * self.v0[0] * t + 1)
        return self.m / self.drag_coeff * ln

    def floor_collision_time(self, z_distance):

        def NewtonRaphson(f, df, xi):
            x = xi
            while abs(f(x)) > 1e-6:
                fx = f(x)
                dfx = df(x)
                x = x - (fx / dfx)
            return x

        f = lambda t: self.position(t)[2] - z_distance
        df = lambda t: self.velocity(t)[2]
        t = NewtonRaphson(f, df, 100.0)
        return t
        # p_half = -self.v0[2] / self.gravity[2]
        # q = -2 * self.p0[2] / self.gravity[2]
        # root = math.sqrt(p_half**2 - q)
        # t1 = -p_half + root
        # t2 = -p_half - root
        # if t1 * t2 < 0.0:
        #     return max(t1, t2)
        # if t1 > 0.0 and t2 > 0.0:
        #     raise RuntimeError("Bad initial condition for ball.")
        # if t1 < 0.0 and t2 < 0.0:
        #     raise RuntimeError("Bad initial condition for ball.")

    def position(self, t):
        return np.array([self.position_x(t), 0.0,
                         self.position_z(t)]) + self.p0

    def velocity(self, t):
        return np.array([self.velocity_x(t), 0.0, self.velocity_z(t)])


class Body():

    def __init__(self, p0, v0, a0, m_rb=0.0, m_added=0.0, damping=0.0):
        self.p = p0
        self.v = v0
        self.a = a0
        self.m_rb = m_rb
        self.m_added = m_added
        self.damping = damping
        self.q = Quaternion()
        self.gravity = GRAVITY
        self.thrust = np.array([0.0, 0.0, 0.0])

    def set_thrust(self, thrust):
        self.thrust = thrust
        print(f'Setting thrust to {self.thrust}')

    def update_state(self, dt):
        self.a = self.thrust + self.gravity
        v_prev = self.v
        self.v += dt * self.a
        dp = 0.5 * (v_prev + self.v) * dt
        self.p += dp

    def position(self):
        return np.copy(self.p)

    def velocity(self):
        return np.copy(self.v)

    def acceleration(self):
        return np.copy(self.a)


def q_map_to_ball(velocity_ball) -> Quaternion:
    z_unit = np.array([0.0, 0.0, 1.0])
    v_normalized = velocity_ball / np.linalg.norm(velocity_ball)
    return rotation_between_vectors(z_unit, v_normalized)


def generate_normals():
    base_normal = np.array([0.0, 0.0, -1.0])
    radial_angle = RADIAL_ANGLE
    delta_radial = radial_angle / N_NORMALS_RADIAL
    radial_angles = np.linspace(delta_radial,
                                radial_angle,
                                N_NORMALS_RADIAL,
                                dtype=float)
    tang_angle = TANG_ANGLE
    delta_tang = tang_angle / N_NORMALS_TANG
    tang_angles = np.linspace(delta_tang, tang_angle, N_NORMALS_TANG)
    q_rad = [Quaternion(angle=x, axis=[1.0, 0.0, 0.0]) for x in radial_angles]
    q_tang = [Quaternion(angle=x, axis=[0.0, 0.0, 1.0]) for x in tang_angles]

    normals = np.zeros([len(q_rad) * len(q_tang), 3], dtype=float)
    for i, q1 in enumerate(q_rad):
        for j, q2 in enumerate(q_tang):
            normals[i * len(q_tang) + j] = q2.rotate(q1.rotate(base_normal))

    return normals


def generate_trajectories(p, v, a, t_now, dt_left, ball: Ball):
    t_start = t_now + dt_left / N_TIMESTEPS
    t_stop = t_now + dt_left
    t_vec = np.linspace(t_start, t_stop, N_TIMESTEPS)
    f_vec = np.linspace(F_MIN, F_MAX, N_FORCESTEPS)
    trajectories = []

    min_cost = float('inf')

    for t in t_vec:
        p_ball = ball.position(t)
        v_ball = ball.velocity(t)
        t_final = t - t_now

        # rotate gravity in the frame, that's z-axis is aligned with the ball's velocity
        q = q_map_to_ball(v_ball).inverse
        p_ball_rotated = q.rotate(p_ball)
        gravity_rotated = q.rotate(GRAVITY)
        p_rotated = q.rotate(p)
        v_rotated = q.rotate(v)
        a_rotated = q.rotate(a)
        for f in f_vec:
            # compute normal in the opposite direction of the ball's velocity
            # this is trivial since we are in the rotated frame, in which the
            # z-axis is aligned with the ball's velocity. So the reversed normal
            # vector is always -unit_z
            normals = generate_normals()
            for i in range(len(normals)):
                trajectory = quadtraj.RapidTrajectory(p_rotated, v_rotated,
                                                      a_rotated,
                                                      gravity_rotated, 1.1, 1.4, 5.4)
                a_final = normals[i, :] * f + gravity_rotated
                p_final = p_ball_rotated - normals[i, :] * 0.2
                trajectory.set_goal_acceleration(a_final)
                # final velocity in z direction can be arbitrary -> will not be specified.
                # both other direction x and y should be zero
                trajectory.set_goal_velocity_in_axis(0, 0.0)
                trajectory.set_goal_velocity_in_axis(1, 0.0)
                trajectory.set_goal_position(p_final)
                trajectory.generate(t_final)
                cost = trajectory.get_cost()
                traj = Trajectory(trajectory, Quaternion(q), t_now, t_final,
                                  p_ball_rotated)
                trajectories.append(traj)

    return trajectories


def select_best_trajectory(trajectories: List[Trajectory]):
    min_cost = float('inf')
    selected_trajectory: Trajectory = None
    solution_found = False
    for t in trajectories:
        cost = t.trajectory.get_cost()
        # no need to check feasibilty if we already have a cheaper solution
        if solution_found and cost >= min_cost:
            continue
        # is cheaper than current solution. If feasible, select this one.
        feasibility = t.trajectory.check_input_feasibility(
            F_MIN, F_MAX, OMEGA_MAX, 0.02)
        if feasibility != quadtraj.InputFeasibilityResult.Feasible:
            continue
        solution_found = True
        selected_trajectory = t
        min_cost = cost
    return selected_trajectory


def test_plot_normals():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    normals = generate_normals()
    for i in range(len(normals)):
        normal = normals[i]
        plt.quiver(0.0,
                   0.0,
                   0.0,
                   normal[0],
                   normal[1],
                   normal[2],
                   length=0.1,
                   normalize=True)

class UuvSimulator():
    def __init__(self, p0, v0, a0, ring: Ring, time_horizon_us) -> None:
        self.ring = ring
        self.time_horizon_us = time_horizon_us
        self.dt_us = 1000
        self.t_last_trajectory_update_us = -1000000000
        self.trajectory_update_period_us = 20000
        self.t_now_us = 0
        self.body = Body(p0, v0, a0)

class QuadSimulator():

    def __init__(self, p0, v0, a0, ball: Ball, time_horizon_us) -> None:
        self.ball = ball
        self.time_horizon_us = time_horizon_us
        self.dt_us = 1000
        self.t_last_trajectory_update_us = -1000000000
        self.trajectory_update_period_us = 20000
        self.t_now_us = 0
        self.body = Body(p0, v0, a0)
        self.selected_trajectory: Trajectory = None
        self.drawn_selected: List(Trajectory) = []
        self.trajecories: List(Trajectory) = []
        self.p_history = []
        self.t_history = []
        self.thrust_factor = 1.0
        self.thrust_offset = 0.0

    def should_regenerate_trajectory(self):
        return (self.t_now_us - self.t_last_trajectory_update_us >=
                self.trajectory_update_period_us)

    def simulate_section(self, n_loops):
        self.p_history = [self.body.position()]
        self.t_history = [self.t_now_us * 1e-6]
        for i in range(n_loops):
            self.t_now_us += self.dt_us
            if self.t_now_us >= self.time_horizon_us:
                return False
            if self.should_regenerate_trajectory():
                self.t_last_trajectory_update_us = self.t_now_us
                t_now = self.t_now_us * 1e-6
                dt_left = (self.time_horizon_us - self.t_now_us) * 1e-6
                self.trajecories = generate_trajectories(
                    self.body.position(), self.body.velocity(),
                    self.body.acceleration(), t_now, dt_left, self.ball)
                t_new = select_best_trajectory(self.trajecories)
                if t_new is not None:
                    self.selected_trajectory = t_new
                thrust = self.selected_trajectory.thrust(t_now + 0.02)
                thrust = thrust * self.thrust_factor + thrust / np.linalg.norm(
                    thrust) * self.thrust_offset
                self.body.set_thrust(thrust * self.thrust_factor)
            self.body.update_state(self.dt_us * 1e-6)
            self.p_history.append(self.body.position())
            self.t_history.append(self.t_now_us * 1e-6)
        return True

    def plot_trajectories(self, ax=None):
        for t in self.drawn_selected:
            self.plot_trajectory(t, color='blue', ax=ax)
        self.drawn_selected.append(self.selected_trajectory)
        self.plot_start_position(self.selected_trajectory, ax=ax)
        # for t in self.trajecories:
        #     self.plot_trajectory(t, color='lightgrey', ax=ax)
        self.plot_trajectory(self.selected_trajectory, color='lime', ax=ax)
        self.plot_movement(ax=ax)
        self.plot_ball(ax=ax)

    def plot_trajectory(self, trajectory: Trajectory, color, ax=None):
        p = trajectory.p_vec(trajectory.t_vec(50))
        if ax is None:
            ax = plt.gca()
        ax.plot(p[:, 0], p[:, 2], color=color)

    def plot_start_position(self, trajectory: Trajectory, ax=None):
        p = trajectory.p_vec(trajectory.t_vec(2))
        if ax is None:
            ax = plt.gca()
        ax.scatter(p[0, 0],
                   p[0, 2],
                   c='grey',
                   edgecolors='black',
                   marker='o',
                   s=20**2)

    def plot_movement(self, ax=None):
        p = np.array(self.p_history)
        t = np.array(self.t_history)
        if ax is None:
            ax = plt.gca()
        ax.plot(p[:, 0], p[:, 2], color='red')
        # plt.figure()
        # for i in range(3):
        #     plt.plot(t, p[:, i])
    def plot_ball(self, ax=None):
        p = self.ball.position(self.t_now_us * 1e-6)
        if ax is None:
            ax = plt.gca()
        ax.scatter(
            p[0],
            p[2],
            c='red',
            # edgecolors='black',
            marker='x',
            s=20**2)


def simulate_figure7_scenario():
    # ball initial conditions
    p0_ball = np.array([-0.5, 0.0, 3.5])
    v0_ball = np.array([2.5, 0.0, 6.0])

    # vehicle initial conditions
    p0 = [0.0, 0.0, 1.0]
    v0 = [0.0, 0.0, 0.0]
    a0 = [0.0, 0.0, 0.0]

    ball = Ball(p0_ball, v0_ball)
    time_horizon_us = int(ball.floor_collision_time(0.3) * 1e6)
    sim = QuadSimulator(p0, v0, a0, ball, time_horizon_us)

    # sim.thrust_factor = 1.1 # works quite okay
    # sim.thrust_factor = 1.2 # around the limit
    # sim.thrust_factor = 1.5 # does not work
    # sim.thrust_factor = 0.9 # does not work
    sim.thrust_factor = 1.0
    sim.simulate_section(1)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim([-0.1, 3])
    ax.set_ylim([-0.5, 5.5])

    sim.plot_trajectories()
    try:
        while sim.simulate_section(40):
            plt.figure()
            sim.plot_trajectories()
            ax = plt.gca()
            ax.set_xlim([-0.1, 3])
            ax.set_ylim([-0.5, 5.5])
    except KeyboardInterrupt:
        pass

    plt.figure()
    sim.plot_trajectories()
    ax = plt.gca()
    ax.set_xlim([-0.1, 3])
    ax.set_ylim([-0.5, 5.5])


def figure7_scenario():
    global N_TIMESTEPS
    N_TIMESTEPS = 3
    # ball initial conditions
    p0_ball = np.array([-0.5, 0.0, 3.5])
    v0_ball = np.array([2.5, 0.0, 6.0])

    # vehicle initial conditions
    p0 = [0.0, 0.0, 1.0]
    v0 = [0.0, 0.0, 0.0]
    a0 = [0.0, 0.0, 0.0]

    # n sampling points for plotting
    n_points = 50

    ball = Ball(p0_ball, v0_ball)
    # calculate final time as the time when the ball is 0.3m above the ground
    T_f = ball.floor_collision_time(0.3)

    # plot the ball's xy-curve
    t = np.linspace(0.0, T_f, n_points)
    p_ball = np.zeros([n_points, 3])
    v_ball = np.zeros_like(p_ball)
    for i in range(n_points):
        p_ball[i, :] = ball.position(t[i])
        v_ball[i, :] = ball.velocity(t[i])

    fig2d = 'figure7_2d'
    fig3d = 'figure7+3d'

    plt.figure(fig2d)
    plt.plot(p_ball[:, 0],
             p_ball[:, 2],
             linestyle='dashdot',
             color='black',
             zorder=1)

    plt.figure(fig3d)
    ax_3d = plt.axes(projection='3d')
    ax_3d.plot3D(p_ball[:, 0],
                 p_ball[:, 1],
                 p_ball[:, 2],
                 linestyle='dashdot',
                 color='black')

    t1 = time.time()
    trajectories = generate_trajectories(p0, v0, a0, 0.2, T_f - 0.2, ball)
    t2 = time.time()
    print(f'Trajectories computed: {t2-t1:.3f} s')
    print(f'{(t2-t1)/len(trajectories)*1e3:.3f}ms per trajectory.\n')

    i_best = 0
    best_cost = float('inf')
    all_costs = []
    counter = FeasibilityCounter()
    tf = -1.0
    ball_positions = []
    for i, traj in enumerate(trajectories):
        t_vec = traj.t_vec(n_points)
        t_vec_abs = traj.t_vec_abs(n_points)
        p = traj.p_vec(t_vec)
        v = traj.v_vec(t_vec)
        a = traj.a_vec(t_vec)
        if traj.tf > tf:
            tf = traj.tf
            ball_positions.append(traj.p_ball())
        feasibility = traj.trajectory.check_input_feasibility(
            F_MIN, F_MAX, OMEGA_MAX, 0.02)
        counter.count(feasibility)
        if feasibility == quadtraj.InputFeasibilityResult.Feasible:
            color = 'lightgrey'
        elif feasibility == quadtraj.InputFeasibilityResult.Indeterminable:
            color = 'cyan'
        else:
            color = 'red'
        plt.figure(fig2d)
        plt.plot(p[:, 0], p[:, 2], color, zorder=5)
        ax_3d.plot3D(p[:, 0], p[:, 1], p[:, 2])
        cost = traj.trajectory.get_cost()
        all_costs.append(cost)
        if cost < best_cost:
            best_cost = cost
            i_best = i
        # a_arrow = np.array([a[-1, 0], a[-1, 2]])
        # a_arrow = a_arrow / np.linalg.norm(a_arrow) * 0.1
        # plt.arrow(p[-1, 0], p[-1, 2], a_arrow[0], a_arrow[1])
    print(f'Plotting time: {t2-t1:.1f}s')
    traj = trajectories[i_best]
    p = traj.p_vec(t_vec)
    plt.plot(p[:, 0], p[:, 2], 'lime', zorder=10)
    ball_positions = np.array(ball_positions)
    plt.scatter(ball_positions[:, 0],
                ball_positions[:, 2],
                c='grey',
                edgecolors='black',
                marker='o',
                s=20**2,
                zorder=10)
    ax_3d.scatter(ball_positions[:, 0],
                  ball_positions[:, 1],
                  ball_positions[:, 2],
                  c='grey',
                  edgecolors='black',
                  marker='o',
                  s=20**2,
                  zorder=10)
    counter.print()
    ax_3d.margins(0.1, 0.1, 0.1)
    ax_3d.set_box_aspect([1, 1, 1])
    ax_3d.set_aspect('equal')


def main():
    figure7_scenario()
    # simulate_figure7_scenario()
    plt.show()


if __name__ == '__main__':
    main()
