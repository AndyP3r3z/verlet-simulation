import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random as rd
import math

# ---------------------------
# Generador sin solapamientos
# ---------------------------
def generate_non_overlapping_positions(N, R, min_dist, max_attempts=5000):
    """
    Genera N posiciones aleatorias dentro de un disco de radio R,
    con separación mínima min_dist entre partículas y respecto de la frontera.
    Retorna un array de forma (N,2).
    """
    positions = []
    safe_radius = R - min_dist

    for i in range(N):
        attempts = 0
        while attempts < max_attempts:
            r = safe_radius * math.sqrt(np.random.rand())
            theta = 2 * math.pi * np.random.rand()
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            candidate = np.array([x, y])

            too_close = False
            for p in positions:
                if np.linalg.norm(candidate - p) < min_dist:
                    too_close = True
                    break

            if not too_close:
                positions.append(candidate)
                break

            attempts += 1

        if attempts >= max_attempts:
            raise RuntimeError(
                f"No se pudo colocar la partícula {i} sin solapamientos. "
                "Reduce min_dist o reduce N o aumenta R."
            )

    return np.array(positions)


# ---------------------------
# Clase Particle
# ---------------------------
class Particle:
    def __init__(self, m, x, y, vx, vy, r, dt=0.002):
        self.m = m
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = r

        self.dt = dt

        # aceleraciones actuales (a) y antiguas (a_old)
        self.ax = 0.0
        self.ay = 0.0

    def __sub__(self, other):
        """Calculates the distance between this and another particle."""
        return math.sqrt( ((self.x - other.x)**2) + ((self.y - other.y)**2) )

# ---------------------------
# Caja con LJ, Verlet y Berendsen
# ---------------------------
class Box:
    def __init__(self, radio, sigma=1.0, epsilon=1.0, cutoff=None, T0=0.1, tau=0.5):
        self.radio = radio
        self.particles = []

        # Lennard-Jones params
        self.sigma = sigma
        self.epsilon = epsilon
        # cutoff en unidades de distancia
        self.cutoff = cutoff

        # Termostato
        self.T0 = T0
        self.tau = tau

    def add_particle(self, p: Particle):
        self.particles.append(p)

    def average_min_distance(self) -> float:
        """Calculates the average minimum distance between all particles."""
        distances: list[list[float]] = []
        mins: list[float] = []
        n_particles: int = len(self.particles)
        # Populate the distances list.
        for i, particle in enumerate(self.particles):
            all_distances: list[float] = [
                particle - self.particles[j]
                # Calculates distances without repetitions.
                for j in range(i+1, n_particles)]
            distances.append(all_distances)
        # Populate the mins.
        for i, diffs in enumerate(distances):
            # Appends previous calculated distances to min.
            to_append: list[float] = [
                distances[j][i-j-1]
                for j in range(i)]
            # Calculates the min.
            mins.append(min(diffs+to_append))
        # Returns average.
        return sum(mins)/len(mins)

    # Fuerza LJ entre pares (componentes)
    def lj_force_components(self, dx, dy, r):
        """Devuelve Fx, Fy (sin dividir por masa). Maneja r->0 con un pequeño eps."""
        if r == 0:
            return 0.0, 0.0
        if (self.cutoff is not None) and (r > self.cutoff):
            return 0.0, 0.0

        sigma = self.sigma
        eps = self.epsilon

        inv_r = 1.0 / r
        inv_r6 = (sigma * inv_r) ** 6
        inv_r12 = inv_r6 * inv_r6
        # Magnitud de la fuerza (derivada de LJ)
        F = 24 * eps * (2 * inv_r12 - inv_r6) * inv_r
        Fx = F * dx
        Fy = F * dy
        return Fx, Fy

    def compute_forces(self):
        """
        Calcula fuerzas y devuelve lista de aceleraciones (ax, ay) para cada partícula.
        Uso O(N^2) simple.
        """
        N = len(self.particles)
        accs = [(0.0, 0.0)] * N  # lista de tuplas (ax, ay), se llenará

        # Inicializa acumuladores
        fx = np.zeros(N)
        fy = np.zeros(N)

        for i in range(N):
            pi = self.particles[i]
            for j in range(i + 1, N):
                pj = self.particles[j]

                dx = pi.x - pj.x
                dy = pi.y - pj.y
                r2 = dx * dx + dy * dy
                if r2 == 0:
                    continue
                r = math.sqrt(r2)

                Fx, Fy = self.lj_force_components(dx, dy, r)

                fx[i] += Fx
                fy[i] += Fy
                fx[j] -= Fx
                fy[j] -= Fy

        for i in range(N):
            ax = fx[i] / self.particles[i].m
            ay = fy[i] / self.particles[i].m
            accs[i] = (ax, ay)

        return accs

    def apply_box_collision_position_fix(self, p: Particle):
        """Coloca la partícula justo en la frontera si sobrepasa (después de mover)."""
        dist_sq = p.x * p.x + p.y * p.y
        limit_sq = (self.radio - p.r) ** 2
        if dist_sq >= limit_sq:
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-12
            nx = p.x / dist
            ny = p.y / dist
            # reposicionar en la frontera
            p.x = (self.radio - p.r) * nx
            p.y = (self.radio - p.r) * ny
            # reflejar velocidad normal (conservativo)
            vn = p.vx * nx + p.vy * ny
            p.vx -= 2 * vn * nx
            p.vy -= 2 * vn * ny

    def compute_temperature(self):
        """Temperatura instantánea proporcional a velocidad cuadrática (k_B=1). 2D."""
        v2 = 0.0
        for p in self.particles:
            v2 += p.vx * p.vx + p.vy * p.vy
        if len(self.particles) == 0:
            return 0.0
        # T = <v^2> / (2) in 2D? We use T = (1/(2N)) sum m v^2 with m=1.
        T = v2 / (2.0 * len(self.particles))
        return T

    def apply_berendsen(self):
        """Reescalado de velocidades para alcanzar self.T0 con tiempo tau."""
        T = self.compute_temperature()
        if T <= 0:
            return
        dt = self.particles[0].dt
        lam = math.sqrt(max(0.0, 1.0 + (dt / self.tau) * (self.T0 / T - 1.0)))
        for p in self.particles:
            p.vx *= lam
            p.vy *= lam

    def simulate_step(self):
        """
        Un paso de Velocity-Verlet completo:
         1) mover posiciones usando a(t)
         2) aplicar corrección si sale de caja (posición)
         3) calcular a(t+dt)
         4) actualizar velocidades usando (a(t)+a(t+dt))/2
         5) aplicar termostato (Berendsen)
        """
        N = len(self.particles)
        if N == 0:
            return

        # 0) guardar aceleraciones actuales como a_old
        a_old = [(p.ax, p.ay) for p in self.particles]

        dt = self.particles[0].dt

        # 1) actualizar posiciones
        for p in self.particles:
            p.x += p.vx * dt + 0.5 * p.ax * dt * dt
            p.y += p.vy * dt + 0.5 * p.ay * dt * dt

        # 2) corregir posiciones que salieron del contenedor (colisión posicional)
        for p in self.particles:
            self.apply_box_collision_position_fix(p)

        # 3) calcular nuevas aceleraciones a_new
        accs = self.compute_forces()  # lista de (ax, ay)
        for i, (ax_new, ay_new) in enumerate(accs):
            self.particles[i].ax = ax_new
            self.particles[i].ay = ay_new

        # 4) actualizar velocidades usando promedio de aceleraciones
        for i, p in enumerate(self.particles):
            ax_old, ay_old = a_old[i]
            p.vx += 0.5 * (ax_old + p.ax) * dt
            p.vy += 0.5 * (ay_old + p.ay) * dt

        # 5) Después de actualizar velocidades, también aplicar colisión por frontera
        # (ya hicimos corrección posicional; aquí solo por si la velocidad las empuja)
        for p in self.particles:
            # Si la posición está correcta, reflejar la velocidad si está saliendo
            dist = math.sqrt(p.x * p.x + p.y * p.y)
            if dist > (self.radio - p.r) - 1e-12:
                nx = p.x / dist
                ny = p.y / dist
                vn = p.vx * nx + p.vy * ny
                if vn > 0:  # si se mueve hacia afuera, reflejar
                    p.vx -= 2 * vn * nx
                    p.vy -= 2 * vn * ny

        # 6) aplicar termostato Berendsen (opcional según configuración)
        self.apply_berendsen()

    def total_energy(self):
        """Energía cinética + potencial LJ aproximada."""
        # Cinética
        K = 0.0
        for p in self.particles:
            K += 0.5 * p.m * (p.vx * p.vx + p.vy * p.vy)
        # Potencial
        U = 0.0
        N = len(self.particles)
        for i in range(N):
            for j in range(i + 1, N):
                pi = self.particles[i]
                pj = self.particles[j]
                dx = pi.x - pj.x
                dy = pi.y - pj.y
                r = math.hypot(dx, dy)
                if r == 0:
                    continue
                sigma = self.sigma
                eps = self.epsilon
                inv_r = sigma / r
                inv_r6 = inv_r ** 6
                inv_r12 = inv_r6 * inv_r6
                U += 4 * eps * (inv_r12 - inv_r6)
        return K + U, K, U


# ---------------------------
# Parámetros y setup
# ---------------------------

def graph_params_evolution(timestep: float, **kwargs: list[float]) -> None:
    """Creates a graph with the parameter evolutions given."""
    labels: list[str] = list(kwargs.keys())
    n: int = len(kwargs)
    length: int = len(next(iter(kwargs.values())))
    t: np.ndarray = np.arange(length)*timestep
    _, axes = plt.subplots(n, 1, sharex=True, figsize=(8, 2.5*n))
    if n == 1: axes = [axes]
    colors: list[str] = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, lbl, color in zip(axes, labels, colors):
        ax.plot(t, kwargs[lbl], color=color)
        ax.legend([lbl])
        ax.grid(True)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    return

def main() -> None:
    # Parámetros
    N = 15
    particle_radius = 0.5
    # A mayor `k`, mayor será el área total en relación con el área ocupada con
    # las esferas (se verá más vacío).
    k: float = 4
    radio = math.sqrt(k*N)*particle_radius # cálculo del radio del contenedor
    min_dist = 1.8 * particle_radius  # separación inicial
    dt = 0.002

    # LJ params
    sigma = 1.0
    epsilon = 1.0
    cutoff = 3.0 * sigma  # Mejora rendimiento y evita fuerzas muy lejanas

    # Termostato
    T0 = 0.05   # temperatura objetivo (baja)
    tau = 1.0   # tiempo de acoplamiento (mayor = más lento)

    # Crear caja
    box = Box(radio=radio, sigma=sigma, epsilon=epsilon, cutoff=cutoff, T0=T0, tau=tau)

    # Generar posiciones iniciales sin solapamientos
    positions = generate_non_overlapping_positions(N=N, R=radio, min_dist=min_dist)

    # Crear partículas con velocidades aleatorias iniciales (escalar velocidad para temperatura inicial)
    initial_T = 1.0
    for pos in positions:
        vx = rd.uniform(-1, 1)
        vy = rd.uniform(-1, 1)
        p = Particle(m=1.0, x=float(pos[0]), y=float(pos[1]), vx=vx, vy=vy, r=particle_radius, dt=dt)
        box.add_particle(p)

    # Calcular aceleraciones iniciales (a(t=0))
    accs0 = box.compute_forces()
    for i, (ax0, ay0) in enumerate(accs0):
        box.particles[i].ax = ax0
        box.particles[i].ay = ay0

    # Reescalar velocidades para una temperatura inicial aproximada
    # (para empezar más frío o más caliente)
    def set_initial_temperature(box_obj, T_target):
        # calcula T actual
        v2 = sum(p.vx**2 + p.vy**2 for p in box_obj.particles)
        if v2 == 0:
            return
        T_current = v2 / (2.0 * len(box_obj.particles))
        lam = math.sqrt(T_target / T_current)
        for p in box_obj.particles:
            p.vx *= lam
            p.vy *= lam

    set_initial_temperature(box, initial_T)

    # ---------------------------
    # Animación
    # ---------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-radio, radio)
    ax.set_ylim(-radio, radio)
    ax.set_aspect('equal')
    ax.set_title("Lennard-Jones + Velocity-Verlet + Berendsen (local)")

    # dibujar frontera circular
    container = plt.Circle((0, 0), radio, fill=False, color='black', lw=1.2)
    ax.add_patch(container)

    particle_artists = []
    for p in box.particles:
        c = plt.Circle((p.x, p.y), p.r, fc='red', alpha=0.9)
        ax.add_patch(c)
        particle_artists.append(c)

    # Graficar temperatura en esquina (texto)
    temp_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    frames = 20000
    steps_per_frame = 1

    energies: list[float] = []
    potentials: list[float] = []
    temperatures: list[float] = []
    r_mins: list[float] = []

    def animate(frame):
        for _ in range(steps_per_frame):
            box.simulate_step()

        E_total, K, U = box.total_energy()
        T_inst = box.compute_temperature()
        parameter = box.average_min_distance()
        energies.append(E_total)
        potentials.append(U)
        temperatures.append(T_inst)
        r_mins.append(parameter)

        temp_text.set_text(f"T={T_inst:.4f}  E={E_total:.4f}\nr_min = {parameter:.4f}")

        for artist, p in zip(particle_artists, box.particles):
            artist.set_center((p.x, p.y))

        return particle_artists + [temp_text]

    ani = FuncAnimation(fig, animate, frames=frames, interval=20, blit=False)
    plt.show()

    graph_params_evolution(
        dt,
        E=energies,
        U=potentials,
        T=temperatures,
        r_min=r_mins)

if __name__ == "__main__": main()
