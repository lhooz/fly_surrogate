import taichi as ti
import numpy as np

# Initialize Taichi
# CRITICAL FOR COLAB: We reduced fraction from 0.7 to 0.4 
# to leave room for JAX (which we also capped at 40%).
try:
    ti.init(arch=ti.gpu, device_memory_fraction=0.8)
except:
    print("Warning: Taichi GPU init failed. Falling back to CPU.")
    ti.init(arch=ti.cpu)

@ti.data_oriented
class TaichiFluidEngine:
    def __init__(self):
        # ==========================================
        #        1. PHYSICAL CONSTANTS (SI)
        # ==========================================
        self.PHYS_SIZE = 0.20
        self.NU_SI = 1.5e-5
        self.RHO_SI = 1.225
        self.WING_LEN = 0.01
        
        self.NX = 500
        self.NY = 500
        
        # ==========================================
        #        2. UNIT CONVERSION
        # ==========================================
        self.DX_SI = self.PHYS_SIZE / self.NX 
        self.DT_SI = 3.0e-6
        self.DT = self.DT_SI
        
        # LBM Params
        self.NU_LB = 0.02
        self.TAU = 3.0 * self.NU_LB + 0.5
        self.OMEGA = 1.0 / self.TAU
        self.C_SMAGO = 0.10 
        self.C_VEL = self.DT_SI / self.DX_SI
        
        # ==========================================
        #        3. STRUCTURAL PARAMS (CPU)
        # ==========================================
        self.N_PTS = 20
        self.THICKNESS = 0.0015
        
        # Physics Coefficients
        """
        self.K_STRETCH = 100000.0
        self.K_BEND_MAX = 10.0
        self.K_BEND_MIN = 10.0
        self.K_COUPLE_MAX = 100.0
        self.K_COUPLE_MIN = 100.0
        self.B_COUPLE_MAX = 10.0 
        self.B_COUPLE_MIN = 10.0
        self.MASS_MAX = 0.00001
        self.MASS_MIN = 0.00001
        self.DAMPING = 0.001
        self.SUBSTEPS = 30
        """
        self.K_STRETCH = 100000.0
        self.K_BEND_MAX = 10.0
        self.K_BEND_MIN = 0.1
        self.K_COUPLE_MAX = 10.0
        self.K_COUPLE_MIN = 0.1
        self.B_COUPLE_MAX = 1.0  
        self.B_COUPLE_MIN = 0.01
        self.MASS_MAX = 0.00001
        self.MASS_MIN = 0.000001
        self.DAMPING = 0.001
        self.SUBSTEPS = 30
        
        self.FORCE_SCALE = 5.0 
        self.MAX_NODE_FORCE = 0.2
        self.ALPHA_PENALTY = 0.1
        self.MAX_LATTICE_FORCE = 0.05
        self.max_f_observed = ti.field(dtype=ti.f32, shape=())

        # ==========================================
        #        4. ALLOCATE MEMORY
        # ==========================================
        
        # --- GPU FIELDS ---
        self.f = ti.Vector.field(9, dtype=ti.f32, shape=(self.NX, self.NY))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.NX, self.NY))
        self.rho = ti.field(dtype=ti.f32, shape=(self.NX, self.NY))
        self.u = ti.Vector.field(2, dtype=ti.f32, shape=(self.NX, self.NY))
        
        # Interface Buffers
        self.s_pos_field = ti.Vector.field(2, dtype=ti.f32, shape=self.N_PTS)
        self.s_vel_field = ti.Vector.field(2, dtype=ti.f32, shape=self.N_PTS)
        self.s_force_aero_field = ti.Vector.field(2, dtype=ti.f32, shape=self.N_PTS)

        # LBM Constants
        self.w = ti.field(dtype=ti.f32, shape=9)
        self.cx = ti.field(dtype=ti.i32, shape=9)
        self.cy = ti.field(dtype=ti.i32, shape=9)

        # --- CPU ARRAYS ---
        self.s_pos = np.zeros((self.N_PTS, 2), dtype=np.float32)
        self.s_vel = np.zeros((self.N_PTS, 2), dtype=np.float32)
        self.s_force_coupling = np.zeros((self.N_PTS, 2), dtype=np.float32)
        
        # --- NONLINEAR RAMP SETUP ---
        x = np.linspace(0.0, 1.0, self.N_PTS)
        k = -np.log(0.5) / 0.15
        exponential_ramp = np.exp(-k * x)

        self.mass_arr = self.MASS_MIN + (self.MASS_MAX - self.MASS_MIN) * exponential_ramp
        self.k_bend_arr = self.K_BEND_MIN + (self.K_BEND_MAX - self.K_BEND_MIN) * exponential_ramp
        self.k_couple_arr = self.K_COUPLE_MIN + (self.K_COUPLE_MAX - self.K_COUPLE_MIN) * exponential_ramp
        self.b_couple_arr = self.B_COUPLE_MIN + (self.B_COUPLE_MAX - self.B_COUPLE_MIN) * exponential_ramp
        
        self.L0 = self.WING_LEN / (self.N_PTS - 1)

        self.init_constants_gpu()
        self.reset(0.0, 0.0, np.pi/2)

    # ==========================================
    #               GPU KERNELS
    # ==========================================

    @ti.kernel
    def init_constants_gpu(self):
        weights = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
        c_x = [0, 1, 0, -1, 0, 1, -1, -1, 1]
        c_y = [0, 0, 1, 0, -1, 1, 1, -1, -1]
        for i in ti.static(range(9)):
            self.w[i] = weights[i]
            self.cx[i] = c_x[i]
            self.cy[i] = c_y[i]

    @ti.kernel
    def reset_fluid_gpu(self):
        for I in ti.grouped(self.f):
            for i in range(9):
                self.f[I][i] = self.w[i]
            self.rho[I] = 1.0
            self.u[I] = [0.0, 0.0]

    @ti.kernel
    def step_lbm_collision(self):
        for x, y in ti.ndrange(self.NX, self.NY):
            f_curr = self.f[x, y]
            rho_val = 0.0
            u_val = ti.Vector([0.0, 0.0])
            
            for i in range(9):
                rho_val += f_curr[i]
                u_val += ti.Vector([float(self.cx[i]), float(self.cy[i])]) * f_curr[i]
            
            u_val /= (rho_val + 1e-9)
            
            self.rho[x, y] = rho_val
            self.u[x, y] = u_val
            
            u_sq = u_val.norm_sqr()
            
            # --- Smagorinsky LES ---
            Pxx = 0.0; Pxy = 0.0; Pyy = 0.0
            for i in range(9):
                cx = float(self.cx[i])
                cy = float(self.cy[i])
                eu = cx*u_val[0] + cy*u_val[1]
                f_eq = self.w[i] * rho_val * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                f_neq = f_curr[i] - f_eq
                Pxx += cx*cx*f_neq
                Pxy += cx*cy*f_neq
                Pyy += cy*cy*f_neq
            
            Pi_norm = ti.sqrt(Pxx**2 + 2.0*Pxy**2 + Pyy**2)
            tau_turb = 0.5 * (ti.sqrt(self.TAU**2 + 18.0 * (self.C_SMAGO**2) * (Pi_norm / (rho_val + 1e-9))) - self.TAU)
            if tau_turb < 0.0: tau_turb = 0.0 
            
            omega_eff = 1.0 / (self.TAU + tau_turb)
            
            # --- Collision ---
            for i in range(9):
                eu = float(self.cx[i])*u_val[0] + float(self.cy[i])*u_val[1]
                f_eq = self.w[i] * rho_val * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                self.f_new[x, y][i] = f_curr[i] - omega_eff * (f_curr[i] - f_eq)

    @ti.kernel
    def compute_fsi_and_apply_force(self):
        # [Image of Immersed Boundary Method grid interaction]
        # This kernel maps Lagrangian structure points to the Eulerian fluid grid.
        
        self.max_f_observed[None] = 0.0  # Reset

        for i in range(self.N_PTS):
            self.s_force_aero_field[i] = [0.0, 0.0]
            
        for i in range(self.N_PTS):
            p = self.s_pos_field[i]
            v = self.s_vel_field[i]
            
            ix = (p.x + self.PHYS_SIZE/2.0) / self.DX_SI
            iy = (p.y + self.PHYS_SIZE/2.0) / self.DX_SI
            base_x, base_y = int(ix), int(iy)
            
            for ox in range(-3, 4): # Reduced stencil from 6 to 3 for speed (usually sufficient)
                for oy in range(-3, 4):
                    grid_x = (base_x + ox + self.NX) % self.NX
                    grid_y = (base_y + oy + self.NY) % self.NY
                    
                    pos_grid_x = (float(grid_x) * self.DX_SI) - (self.PHYS_SIZE / 2.0)
                    pos_grid_y = (float(grid_y) * self.DX_SI) - (self.PHYS_SIZE / 2.0)
                    
                    dist_sq = (pos_grid_x - p.x)**2 + (pos_grid_y - p.y)**2
                    weight = ti.exp(-dist_sq / (2.0 * (self.THICKNESS/2.0)**2))
                    
                    if weight > 1e-6:
                        rho_local = self.rho[grid_x, grid_y]
                        u_fluid = self.u[grid_x, grid_y]
                        u_solid_lb = v * self.C_VEL
                        
                        # Brinkman Penalization
                        f_lb = -self.ALPHA_PENALTY * rho_local * (u_fluid - u_solid_lb)
                        # Track the absolute maximum for monitoring
                        ti.atomic_max(self.max_f_observed[None], ti.abs(f_lb[0]))
                        ti.atomic_max(self.max_f_observed[None], ti.abs(f_lb[1]))

                        f_lb[0] = ti.max(ti.min(f_lb[0], self.MAX_LATTICE_FORCE), -self.MAX_LATTICE_FORCE)
                        f_lb[1] = ti.max(ti.min(f_lb[1], self.MAX_LATTICE_FORCE), -self.MAX_LATTICE_FORCE)
                        
                        # Force Transfer
                        self.s_force_aero_field[i] -= f_lb * self.FORCE_SCALE * weight
                        
                        # Fluid Feedback
                        for k in range(9):
                            f_dot_c = f_lb[0]*float(self.cx[k]) + f_lb[1]*float(self.cy[k])
                            source = 3.0 * self.w[k] * rho_local * f_dot_c * weight
                            self.f_new[grid_x, grid_y][k] += source

    @ti.kernel
    def step_lbm_streaming(self):
        for x, y in ti.ndrange(self.NX, self.NY):
            for i in range(9):
                src_x = (x - self.cx[i] + self.NX) % self.NX
                src_y = (y - self.cy[i] + self.NY) % self.NY
                self.f[x, y][i] = self.f_new[src_x, src_y][i]

    # ==========================================
    #           PYTHON METHODS (CPU)
    # ==========================================
    def get_max_lattice_force(self):
        return self.max_f_observed[None]

    def reset(self, start_bx, start_bz, start_angle):
        self.reset_fluid_gpu()
        c = np.cos(start_angle)
        s = np.sin(start_angle)
        for i in range(self.N_PTS):
            x_local = (self.WING_LEN / 2.0) - (float(i) / (self.N_PTS - 1)) * self.WING_LEN
            self.s_pos[i, 0] = start_bx + x_local * c
            self.s_pos[i, 1] = start_bz + x_local * s
            self.s_vel[i, :] = 0.0
        self.s_pos_field.from_numpy(self.s_pos)
        self.s_vel_field.from_numpy(self.s_vel)

    def step(self, ghost_pos, ghost_angle, ghost_vel_lin, ghost_vel_ang):
        # 1. GPU LBM Step
        self.s_pos_field.from_numpy(self.s_pos)
        self.s_vel_field.from_numpy(self.s_vel)
        self.step_lbm_collision()
        self.compute_fsi_and_apply_force()
        self.step_lbm_streaming()
        
        # 2. Retrieve Aero Forces
        f_aero_np = self.s_force_aero_field.to_numpy()
        
        # 3. CPU Structural Step
        self.step_structure_cpu(ghost_pos, ghost_angle, ghost_vel_lin, ghost_vel_ang, f_aero_np)

    def step_structure_cpu(self, ghost_pos, ghost_angle, ghost_vel_lin, ghost_vel_ang, aero_forces):
        bx, bz = ghost_pos
        vbx, vbz = ghost_vel_lin
        c, s = np.cos(ghost_angle), np.sin(ghost_angle)
        dt_sub = self.DT / self.SUBSTEPS
        
        self.s_force_coupling.fill(0)
        
        for _ in range(self.SUBSTEPS):
            f_total = np.zeros_like(self.s_pos)
            
            # Spring Stretch
            diff_prev = self.s_pos[:-1] - self.s_pos[1:] 
            dist_prev = np.linalg.norm(diff_prev, axis=1, keepdims=True) + 1e-9
            dir_prev = diff_prev / dist_prev
            f_spring = self.K_STRETCH * (dist_prev - self.L0) * dir_prev
            f_total[1:] += f_spring
            f_total[:-1] -= f_spring
            
            # Bending
            if self.N_PTS > 2:
                mid = 0.5 * (self.s_pos[:-2] + self.s_pos[2:])
                curve = mid - self.s_pos[1:-1]
                f_bend = self.k_bend_arr[1:-1, None] * curve
                f_total[1:-1] += f_bend
            
            # Ghost Coupling
            x_local_arr = np.linspace(self.WING_LEN/2, -self.WING_LEN/2, self.N_PTS)
            gx = bx + x_local_arr * c
            gy = bz + x_local_arr * s
            ghost_pts = np.stack([gx, gy], axis=1)
            
            rx, ry = gx - bx, gy - bz
            vgx = vbx - ghost_vel_ang * ry
            vgy = vbz + ghost_vel_ang * rx
            ghost_vels = np.stack([vgx, vgy], axis=1)
            
            f_couple_inst = self.k_couple_arr[:, None] * (ghost_pts - self.s_pos) + \
                            self.b_couple_arr[:, None] * (ghost_vels - self.s_vel)
            
            self.s_force_coupling += f_couple_inst / self.SUBSTEPS
            f_total += f_couple_inst
            
            # Aero + Damping
            aero_clamped = np.clip(aero_forces, -self.MAX_NODE_FORCE, self.MAX_NODE_FORCE)
            f_total += aero_clamped
            f_total -= self.DAMPING * self.s_vel
            
            accel = f_total / self.mass_arr[:, None]
            self.s_vel += accel * dt_sub
            self.s_pos += self.s_vel * dt_sub

    def get_observation(self):
        return (
            self.s_pos.copy(),
            self.s_vel.copy(),
            self.s_force_aero_field.to_numpy(),
            self.s_force_coupling.copy(),
            self.max_f_observed[None]
        )