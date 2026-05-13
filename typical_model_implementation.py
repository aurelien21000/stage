import numpy as np
from numba import njit


@njit(nogil=True)
def executer_simulation_complete_numba(
        T_fin, rayon_min,
        pos_agents, theta, pos_obs, pos_anchors, M, N, L, dt, u0,
        r_R, r_A, nu, tau, zeta, eta, d_s, d_o,
        nc, taille_cellule, C_phi_factor, mu_factor, fact1, fact2,
        grille_A, compteur_A, grille_O, compteur_O,
        f_phi_X, f_phi_Z, f_rep_aa, cos_sum, sin_sum
):
    temps_physique = 0.0

    while temps_physique < T_fin - 1e-9:

        # Réinitialisation of all arrays
        compteur_A.fill(0)
        compteur_O.fill(0)
        f_phi_X.fill(0.0)
        f_phi_Z.fill(0.0)
        f_rep_aa.fill(0.0)

        for k in range(M):
            cos_sum[k] = np.cos(theta[k])
            sin_sum[k] = np.sin(theta[k])

        # agent's positions in the grid
        for k in range(M):
            cx = int(pos_agents[k, 0] / taille_cellule) % nc
            cy = int(pos_agents[k, 1] / taille_cellule) % nc
            idx = compteur_A[cx, cy]
            grille_A[cx, cy, idx] = k
            compteur_A[cx, cy] += 1

        # obstacle's positions in the grid (using modulo to project into periodic domain)
        for i in range(N):
            ox_mod = pos_obs[i, 0] % L
            oy_mod = pos_obs[i, 1] % L
            cx = int(ox_mod / taille_cellule) % nc
            cy = int(oy_mod / taille_cellule) % nc
            idx = compteur_O[cx, cy]
            grille_O[cx, cy, idx] = i
            compteur_O[cx, cy] += 1

        # Répulsion / Attraction Agent-Obstacle
        for cx in range(nc):
            for cy in range(nc):
                for idx_k in range(compteur_A[cx, cy]):
                    k = grille_A[cx, cy, idx_k]
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx = (cx + dx) % nc
                            ny = (cy + dy) % nc
                            for idx_i in range(compteur_O[nx, ny]):
                                i = grille_O[nx, ny, idx_i]

                                # Using modulo for obstacle position to interact correctly within periodic bounds
                                dx_pos = (pos_obs[i, 0] % L) - pos_agents[k, 0]
                                dy_pos = (pos_obs[i, 1] % L) - pos_agents[k, 1]
                                dx_pos = dx_pos - L * np.round(dx_pos / L)
                                dy_pos = dy_pos - L * np.round(dy_pos / L)
                                dist = np.sqrt(dx_pos ** 2 + dy_pos ** 2)

                                if 1e-12 <= dist < tau:
                                    f_scal = C_phi_factor * (1.0 - dist / tau)
                                    fx = (dx_pos / dist) * f_scal
                                    fy = (dy_pos / dist) * f_scal
                                    f_phi_X[i, 0] += fx
                                    f_phi_X[i, 1] += fy
                                    f_phi_Z[k, 0] -= fx
                                    f_phi_Z[k, 1] -= fy

        # Agent-Agent Repulsion
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1))
        for cx in range(nc):
            for cy in range(nc):
                for idx_k in range(compteur_A[cx, cy]):
                    k = grille_A[cx, cy, idx_k]
                    for off_x, off_y in offsets:
                        nx = (cx + off_x) % nc
                        ny = (cy + off_y) % nc
                        for idx_l in range(compteur_A[nx, ny]):
                            l = grille_A[nx, ny, idx_l]

                            if off_x == 0 and off_y == 0 and l <= k:
                                continue

                            dx_pos = pos_agents[k, 0] - pos_agents[l, 0]
                            dy_pos = pos_agents[k, 1] - pos_agents[l, 1]
                            dx_pos = dx_pos - L * np.round(dx_pos / L)
                            dy_pos = dy_pos - L * np.round(dy_pos / L)
                            dist = np.sqrt(dx_pos ** 2 + dy_pos ** 2)

                            if 0 < dist < r_R:
                                f_scal = mu_factor * (1.0 - dist / r_R)
                                fx = (dx_pos / dist) * f_scal
                                fy = (dy_pos / dist) * f_scal
                                f_rep_aa[k, 0] += fx
                                f_rep_aa[k, 1] += fy
                                f_rep_aa[l, 0] -= fx
                                f_rep_aa[l, 1] -= fy

                            if dist <= r_A:
                                cos_sum[k] += np.cos(theta[l])
                                sin_sum[k] += np.sin(theta[l])
                                cos_sum[l] += np.cos(theta[k])
                                sin_sum[l] += np.sin(theta[k])

        #adaptative timestep
        v_max = 1e-9
        N_safe = N if N > 0 else 1
        M_safe = M if M > 0 else 1

        for k in range(M):
            f_tot_x = (1.0 / (zeta * N_safe)) * f_phi_Z[k, 0] + (1.0 / (zeta * M_safe)) * f_rep_aa[k, 0]
            f_tot_y = (1.0 / (zeta * N_safe)) * f_phi_Z[k, 1] + (1.0 / (zeta * M_safe)) * f_rep_aa[k, 1]
            v_agent = np.sqrt(f_tot_x ** 2 + f_tot_y ** 2) + u0
            if v_agent > v_max: v_max = v_agent

        for i in range(N):
            f_rep_obs_x = (1.0 / (eta * M_safe)) * f_phi_X[i, 0]
            f_rep_obs_y = (1.0 / (eta * M_safe)) * f_phi_X[i, 1]
            v_obs = np.sqrt(f_rep_obs_x ** 2 + f_rep_obs_y ** 2)
            if v_obs > v_max: v_max = v_obs

        dt_ideal = (0.1 * rayon_min) / v_max
        K = int(np.ceil(dt / dt_ideal))
        if u0 > 1e-9:
            K_max = int(np.ceil(dt / ((0.5 * rayon_min) / u0)))
        else:
            K_max = 1

        if K_max > K: K = K_max
        if K > 10: K = 10
        if K < 1: K = 1

        dt_adapt = dt / float(K)
        std_agents_adapt = np.sqrt(2.0 * d_s * dt_adapt)
        std_obs_adapt = np.sqrt(2.0 * d_o * dt_adapt)  # Brownian motion properties

        if fact1 < 1.0:
            a = -np.log(fact1) / dt
            fact1_adapt = np.exp(-a * dt_adapt)
            fact2_adapt = (1.0 - fact1_adapt) / a if a > 1e-9 else dt_adapt
        else:
            fact1_adapt = 1.0
            fact2_adapt = dt_adapt

        # Update of obstacle positions then agents
        for i in range(N):
            # Absolute space calculation for the Hookean spring - NO np.round, NO modulo
            dx_res = pos_obs[i, 0] - pos_anchors[i, 0]
            dy_res = pos_obs[i, 1] - pos_anchors[i, 1]

            f_rep_obs_x = (1.0 / (eta * M_safe)) * f_phi_X[i, 0]
            f_rep_obs_y = (1.0 / (eta * M_safe)) * f_phi_X[i, 1]

            dep_x = dx_res * (fact1_adapt - 1.0) + f_rep_obs_x * fact2_adapt
            dep_y = dy_res * (fact1_adapt - 1.0) + f_rep_obs_y * fact2_adapt

            bruit_x = np.random.normal(0.0, std_obs_adapt)
            bruit_y = np.random.normal(0.0, std_obs_adapt)

            # calculating noise after choosing the time-step
            # NO MODULO L HERE: the obstacle evolves in absolute space
            pos_obs[i, 0] = pos_obs[i, 0] + dep_x + bruit_x
            pos_obs[i, 1] = pos_obs[i, 1] + dep_y + bruit_y

        for k in range(M):
            theta_bar = np.arctan2(sin_sum[k], cos_sum[k])
            diff_angle = np.sin(theta_bar - theta[k])
            # calculating final orientation (explanation in the TeX)
            bruit_theta = np.random.normal(0.0, std_agents_adapt)
            theta[k] = theta[k] + nu * diff_angle * dt_adapt + bruit_theta

            v_act_x = u0 * np.cos(theta[k])
            v_act_y = u0 * np.sin(theta[k])

            f_tot_x = (1.0 / (zeta * N_safe)) * f_phi_Z[k, 0] + (1.0 / (zeta * M_safe)) * f_rep_aa[k, 0]
            f_tot_y = (1.0 / (zeta * N_safe)) * f_phi_Z[k, 1] + (1.0 / (zeta * M_safe)) * f_rep_aa[k, 1]

            dep_x = (v_act_x + f_tot_x) * dt_adapt
            dep_y = (v_act_y + f_tot_y) * dt_adapt

            pos_agents[k, 0] = (pos_agents[k, 0] + dep_x) % L
            pos_agents[k, 1] = (pos_agents[k, 1] + dep_y) % L

        temps_physique += dt_adapt

    return temps_physique


class SimulationSwimmers:
    def __init__(self, M_agents=3000, N_obstacles=3000, espace_taille=1.0, pattern_type="bandes"):
        self.L = espace_taille
        self.params = {
            "u0": 1.0, "r_R": 0.075, "r_A": 0.1, "nu": 2.0, "tau": 0.15,
            "C_phi": 5.0, "d_s": 0.02, "eta": 1.0, "d_o": 0.0
        }
        presets = {
            "bandes": {"kappa": 1000.0, "mu": 0.004, "zeta": 5.0},
            "amas": {"kappa": 100.0, "mu": 0.02, "zeta": 0.5},
            "pistes": {"kappa": 100.0, "mu": 0.04, "zeta": 0.5},
            "nid_abeille": {"kappa": 10.0, "mu": 0.05, "zeta": 0.5},
            "stable": {"kappa": 1000.0, "mu": 0.007, "zeta": 5.0},
            "cas_limite": {"kappa": 1000.0, "mu": 0.006, "zeta": 0.2}
        }
        if pattern_type in presets: self.params.update(presets[pattern_type])

        self.M = M_agents
        self.N = N_obstacles
        self.dt = 0.005
        self.temps_physique = 0.0

        rng = np.random.default_rng()
        self.pos_agents = rng.uniform(0, self.L, (self.M, 2))
        self.theta = np.full(self.M, np.pi / 4)
        self.pos_obs = rng.uniform(0, self.L, (self.N, 2))
        self.pos_anchors = self.pos_obs.copy()

        rayon_max = max(self.params['r_A'], self.params['tau'], self.params['r_R'])
        self.nc = max(3, int(self.L / rayon_max))
        self.taille_cellule = self.L / self.nc

        self.C_phi_factor = -3.0 * self.params['C_phi'] / (np.pi * self.params['tau'] ** 2)
        self.mu_factor = 12.0 * self.params['mu'] / (np.pi * self.params['r_R'] ** 3)

        a = self.params['kappa'] / self.params['eta']
        if a > 1e-6:
            self.fact1 = np.exp(-a * self.dt)
            self.fact2 = (1.0 - self.fact1) / a
        else:
            self.fact1 = 1.0
            self.fact2 = self.dt

        self.grille_A = np.full((self.nc, self.nc, self.M), -1, dtype=np.int32)
        self.compteur_A = np.zeros((self.nc, self.nc), dtype=np.int32)
        self.grille_O = np.full((self.nc, self.nc, self.N), -1, dtype=np.int32)
        self.compteur_O = np.zeros((self.nc, self.nc), dtype=np.int32)

        self.f_phi_X = np.zeros((self.N, 2))
        self.f_phi_Z = np.zeros((self.M, 2))
        self.f_rep_aa = np.zeros((self.M, 2))
        self.cos_sum = np.zeros(self.M)
        self.sin_sum = np.zeros(self.M)

    def lancer_simulation(self, T_fin):
        if self.M == 0 or self.N == 0: return False
        rayons_actifs = [r for r in [self.params['r_R'], self.params['r_A'], self.params['tau']] if r > 0]
        rayon_min = min(rayons_actifs) if rayons_actifs else 1.0

        self.temps_physique = executer_simulation_complete_numba(
            T_fin, rayon_min,
            self.pos_agents, self.theta, self.pos_obs, self.pos_anchors,
            self.M, self.N, self.L, self.dt, self.params['u0'],
            self.params['r_R'], self.params['r_A'], self.params['nu'], self.params['tau'],
            self.params['zeta'], self.params['eta'], self.params['d_s'], self.params['d_o'],
            self.nc, self.taille_cellule, self.C_phi_factor, self.mu_factor, self.fact1, self.fact2,
            self.grille_A, self.compteur_A, self.grille_O, self.compteur_O,
            self.f_phi_X, self.f_phi_Z, self.f_rep_aa, self.cos_sum, self.sin_sum
        )
        return True

    def mettre_a_jour_parametres(self, kappa=None, zeta=None, mu=None, **kwargs):
        if kappa is not None: self.params['kappa'] = kappa
        if zeta is not None: self.params['zeta'] = zeta
        if mu is not None: self.params['mu'] = mu

        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value

        #adaptating all constants
        if self.params['tau'] > 0:
            self.C_phi_factor = -3.0 * self.params['C_phi'] / (np.pi * self.params['tau'] ** 2)
        else:
            self.C_phi_factor = 0.0

        if self.params['r_R'] > 0:
            self.mu_factor = 12.0 * self.params['mu'] / (np.pi * self.params['r_R'] ** 3)
        else:
            self.mu_factor = 0.0

        a = self.params['kappa'] / self.params['eta']
        if a > 1e-6:
            self.fact1 = np.exp(-a * self.dt)
            self.fact2 = (1.0 - self.fact1) / a
        else:
            self.fact1 = 1.0
            self.fact2 = self.dt