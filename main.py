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
# Caja con LJ, Verlet, Berendsen o Langevin
# ---------------------------
class Box:
    def __init__(self, radio, sigma=1.0, epsilon=1.0, cutoff=None, T0=0.1, gamma=5.0, tau=0.5):
        self.radio = radio
        self.particles = []
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.T0 = T0
        
        # Parámetros para ambos termostatos
        self.gamma = gamma  # Para Langevin (Fricción)
        self.tau = tau      # Para Berendsen (Tiempo de relajación)

    def add_particle(self, p: Particle):
        self.particles.append(p)

    def average_min_distance(self) -> float:
        """Calcula la distancia mínima promedio (necesario para las gráficas)."""
        distances = []
        mins = []
        n_particles = len(self.particles)
        # Poblar lista de distancias
        for i, particle in enumerate(self.particles):
            all_distances = [particle - self.particles[j] for j in range(i+1, n_particles)]
            distances.append(all_distances)
        # Calcular mínimos
        for i, diffs in enumerate(distances):
            to_append = [distances[j][i-j-1] for j in range(i)]
            full_list = diffs + to_append
            if full_list:
                mins.append(min(full_list))
            else:
                mins.append(0.0)
        
        if not mins: return 0.0
        return sum(mins)/len(mins)

    def compute_forces(self):
        """
        Calcula fuerzas vectorizadas. Puede usar Langevin o solo LJ conservativo.
        """
        N = len(self.particles)
        if N == 0: return []
        
        # 1. Preparar datos (Numpy)
        pos = np.array([[p.x, p.y] for p in self.particles])
        vel = np.array([[p.vx, p.vy] for p in self.particles])
        masses = np.array([p.m for p in self.particles]).reshape(-1, 1)

        # 2. Matriz de distancias
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] 
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf) # Evitar división por cero

        # 3. Fuerzas LJ (Conservativas)
        mask = np.ones_like(r2, dtype=bool)
        if self.cutoff is not None:
            mask = r2 < self.cutoff**2

        inv_r2 = np.zeros_like(r2)
        inv_r2[mask] = 1.0 / r2[mask]
        sr2 = (self.sigma**2) * inv_r2
        sr6 = sr2 ** 3
        
        # F = 24 * eps * inv_r2 * sr6 * (2*sr6 - 1)
        factor = (24.0 * self.epsilon * inv_r2) * sr6 * (2.0 * sr6 - 1.0)
        f_matrix = factor[:, :, np.newaxis] * diff
        f_conservative = np.sum(f_matrix, axis=1)

        #--------------------------- Elección del termostato--------------------------------
        
        # --- OPCION A: LANGEVIN (Descomentar para usar) ---
        dt = self.particles[0].dt
        noise_std = np.sqrt(2.0 * self.T0 * self.gamma * masses / dt)
        f_random = noise_std * np.random.normal(size=(N, 2))
        f_drag = -self.gamma * vel 
        f_total = f_conservative + f_drag + f_random
        
        # --- OPCION B: SOLO LJ (Para usar Berendsen o NVE) ---
        # Si usas Berendsen, comenta las 4 lineas de arriba (Option A) y descomenta esta:
        # f_total = f_conservative 

        acc_array = f_total / masses
        return [tuple(a) for a in acc_array]

    def apply_berendsen(self):
        """Termostato de Berendsen (Reescalado de velocidades)."""
        T = self.compute_temperature()
        if T <= 0: return
        dt = self.particles[0].dt
        lam = math.sqrt(max(0.0, 1.0 + (dt / self.tau) * (self.T0 / T - 1.0)))
        for p in self.particles:
            p.vx *= lam
            p.vy *= lam

    def apply_box_collision_position_fix(self, p: Particle):
        """Corrección de posición en la frontera circular."""
        dist_sq = p.x * p.x + p.y * p.y
        limit_sq = (self.radio - p.r) ** 2
        if dist_sq >= limit_sq:
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 1e-12
            nx = p.x / dist; ny = p.y / dist
            p.x = (self.radio - p.r) * nx
            p.y = (self.radio - p.r) * ny
            vn = p.vx * nx + p.vy * ny
            p.vx -= 2 * vn * nx
            p.vy -= 2 * vn * ny

    def compute_temperature(self):
        v2 = sum(p.vx**2 + p.vy**2 for p in self.particles)
        if not self.particles: return 0.0
        return v2 / (2.0 * len(self.particles))

    def simulate_step(self):
        N = len(self.particles)
        if N == 0: return
        dt = self.particles[0].dt
        a_old = [(p.ax, p.ay) for p in self.particles]

        # 1. Verlet: Posición
        for p in self.particles:
            p.x += p.vx * dt + 0.5 * p.ax * dt * dt
            p.y += p.vy * dt + 0.5 * p.ay * dt * dt
        
        # 2. Colisión Pared
        for p in self.particles: self.apply_box_collision_position_fix(p)

        # 3. Verlet: Fuerzas (Langevin entra aquí si está activo)
        accs = self.compute_forces()
        for i, (ax, ay) in enumerate(accs):
            self.particles[i].ax = ax
            self.particles[i].ay = ay

        # 4. Verlet: Velocidad
        for i, p in enumerate(self.particles):
            p.vx += 0.5 * (a_old[i][0] + p.ax) * dt
            p.vy += 0.5 * (a_old[i][1] + p.ay) * dt

        # ----- ----Velocidades para el termostato ----------

        # --- OPCION A: LANGEVIN ---
        # Si usas Langevin, NO descomentes Berendsen.
        pass 

        # --- OPCION B: BERENDSEN (Descomentar para usar) ---
        # self.apply_berendsen() 
        
    def total_energy(self):
        K = sum(0.5 * p.m * (p.vx**2 + p.vy**2) for p in self.particles)
        U = 0.0
        N = len(self.particles)
        # Cálculo simple de potencial para las gráficas
        for i in range(N):
            for j in range(i + 1, N):
                pi = self.particles[i]; pj = self.particles[j]
                dx = pi.x - pj.x; dy = pi.y - pj.y
                r = math.hypot(dx, dy)
                if r == 0: continue
                inv_r = self.sigma / r
                inv_r6 = inv_r ** 6
                U += 4 * self.epsilon * (inv_r6**2 - inv_r6)
        return K + U, K, U



# ---------------------------Parámetros y setup---------------------------

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
    N = 100  # número de partículas
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
    gamma = 5.0  # coeficiente de fricción para Langevin 
    #gamma bajo = más agitado, gamma alto = más viscoso

    # tau = 1.0   # tiempo de acoplamiento (mayor = más lento) Brandensen

    # Crear caja
    box = Box(radio=radio, sigma=sigma, epsilon=epsilon, cutoff=cutoff, T0=T0, gamma=gamma) #Quitar gamma si se deasea utilizar Brendensen en vz de Langevin

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
    # ax.set_title("Lennard-Jones + Velocity-Verlet + Berendsen (local)")
    ax.set_title("Lennard-Jones + Velocity-Verlet + Langevin")

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
