import os

# ---------------------------------------------------------
# COLAB GPU CONFIGURATION
# ---------------------------------------------------------
# 1. Stop JAX from pre-allocating 90% of VRAM at startup.
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 2. Alternatively, force JAX to take only a specific fraction (e.g., 40%)
#    leaving the rest for Taichi.
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

# 3. Ensure JAX sees the GPU (Remove the "cpu" forcing)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import pickle
import random
import time
import matplotlib

# Essential for headless servers/Colab to prevent display errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 4. Initialize Taichi with memory limits
import taichi as ti
try:
    # Limit Taichi to ~40% of GPU memory so it doesn't fight JAX
    ti.init(arch=ti.gpu, device_memory_fraction=0.4)
except:
    print("Warning: Taichi failed to initialize on GPU. Falling back to CPU.")
    ti.init(arch=ti.cpu)

# Verify devices
print(f"JAX Devices: {jax.devices()}")

# --- IMPORT TEACHER ---
try:
    from environment_engine import TaichiFluidEngine
except ImportError:
    print("Error: 'environment_engine.py' not found.")
    exit()

# ==========================================
# 1. BODY KINEMATICS (Global Jitter)
# ==========================================
class BodyKinematics:
    """
    Simulates the global motion of the insect body (The "Hinge").
    Includes translation (X, Z) and Body Pitch (Theta).
    """
    def __init__(self):
        # JIT-compile the state function for performance
        self._jit_body_fn = jax.jit(self._compute_body_state)

    @staticmethod
    def _compute_body_state(t, Ax, fx, phx, Az, fz, phz, Ap, fp, php):
        """
        Calculates Global Body Position and Pitch.
        Simple sinusoidal perturbations (random drift/jitter).
        """
        # Global Translation (X, Z)
        bx = Ax * jnp.sin(2.0 * jnp.pi * fx * t + phx)
        bz = Az * jnp.sin(2.0 * jnp.pi * fz * t + phz)
        
        # Body Pitch (Theta) - centered at 0
        theta = Ap * jnp.sin(2.0 * jnp.pi * fp * t + php)
        
        return jnp.stack([bx, bz, theta])

    def get_body_state(self, t, body_controls):
        """
        Returns (Pos, Vel) for the body center/hinge.
        Pos = [x, z, theta]
        Vel = [vx, vz, omega]
        """
        # Use JAX JVP (Jacobian-Vector Product) for exact velocity
        primals, tangents = jax.jvp(
            lambda time_val: self._jit_body_fn(time_val, *body_controls),
            (t,), (1.0,)
        )
        
        pos = np.array(primals)   # [x, z, theta]
        vel = np.array(tangents)  # [vx, vz, omega]
        return pos, vel

# ==========================================
# 2. WING KINEMATICS (Local Flapping)
# ==========================================
class WingKinematics:
    """
    Simulates the flapping motion relative to the hinge.
    Uses JAX for exact derivatives of the Tanh pitch switching.
    """
    def __init__(self):
        self._jit_wing_fn = jax.jit(self._compute_raw_wing_state)

    @staticmethod
    def _compute_raw_wing_state(t, freq, stroke_amp, pitch_phase, dev_amp, f_dev, dev_phase, aoa_down, aoa_up):
        """
        Calculates UN-RAMPED local wing state.
        Returns [Stroke, Dev, Pitch].
        """
        phi_flap = 2.0 * jnp.pi * freq * t
        phi_dev  = 2.0 * jnp.pi * f_dev * t + dev_phase
        
        # A. Stroke (Sinusoidal)
        stroke_val = stroke_amp * jnp.sin(phi_flap)
        
        # B. Deviation (Figure-8)
        dev_val = -dev_amp * jnp.sin(phi_dev)
        
        # C. Pitch (Tanh Switching)
        stroke_vel_sign = jnp.cos(phi_flap + pitch_phase)
        # Sharpness = 3.0 provides smooth but fast switching
        switch = 0.5 * (1.0 + jnp.tanh(3.0 * stroke_vel_sign))
        current_mid_aoa = aoa_up + (aoa_down - aoa_up) * switch
        pitch_val = -current_mid_aoa * jnp.cos(phi_flap + pitch_phase)
        
        return jnp.stack([stroke_val, dev_val, pitch_val])

    def get_local_pose_and_velocity(self, t, controls, ramp_val=1.0, ramp_rate=0.0):
        """
        Calculates Local Wing Pose and Velocity (relative to hinge).
        Applies Product Rule for lag-free ramping.
        """
        # Unpack exact controls matching training loop
        # [freq, stroke_amp, aoa_down, aoa_up, pitch_phase, dev_amp, f_dev, dev_phase]
        args = tuple(controls)

        # 1. Get Oscillating State (Pos) and Derivative (Vel) via JAX
        primals, tangents = jax.jvp(
            lambda s: self._jit_wing_fn(s, *args), (t,), (1.0,)
        )
        
        p_osc = np.array(primals) # [str, dev, pitch]
        v_osc = np.array(tangents)

        # 2. Apply Ramp (Product Rule: d(uv) = u'v + uv')
        # Pos = P * R
        # Vel = V*R + P*R'
        p_final = p_osc * ramp_val
        v_final = v_osc * ramp_val + p_osc * ramp_rate
        
        return p_final, v_final

# ==========================================
# 3. COMPOSITE KINEMATICS (Rigid Body Transform)
# ==========================================
def compute_global_kinematics(t, wing_gen, body_gen, wing_ctrl, body_ctrl, ramp_val, ramp_rate):
    """
    Combines Body Jitter + Local Flapping into Global State.
    Returns the exact inputs needed for 'teacher.step()'.
    """
    
    # 1. Get State Components
    # Body: [bx, bz, b_theta], [b_vx, b_vz, b_omega]
    b_pos_raw, b_vel_raw = body_gen.get_body_state(t, body_ctrl)

    b_pos = b_pos_raw * ramp_val
    b_vel = b_vel_raw * ramp_val + b_pos_raw * ramp_rate
    
    # Wing: [str, dev, w_pitch], [v_str, v_dev, w_omega_local]
    w_pos, w_vel = wing_gen.get_local_pose_and_velocity(t, wing_ctrl, ramp_val, ramp_rate)
    
    # 2. Unpack
    bx, bz, b_theta = b_pos
    b_vx, b_vz, b_omega = b_vel
    
    l_str, l_dev, l_pitch = w_pos
    v_str, v_dev, v_pitch = w_vel
    
    # 3. Rigid Body Transformation
    # We rotate the Stroke Plane by the Body Pitch
    c, s = np.cos(b_theta), np.sin(b_theta)
    
    # Global Position (Hinge + Rotated Stroke Translation)
    # 
    # P_global = P_body + R_body * P_local
    g_x = bx + (l_str * c - l_dev * s)
    g_z = bz + (l_str * s + l_dev * c)
    
    # Global Angle (Add Body Pitch + Wing Pitch)
    # + Pi/2 to align with vertical (standard LBM convention)
    g_ang = b_theta + l_pitch + (np.pi / 2.0)
    
    # Global Velocity
    # V_global = V_body + (Omega x R*P_local) + (R * V_local)
    
    # Term 1: Rotated Local Velocity (R * V_local)
    rv_x = v_str * c - v_dev * s
    rv_z = v_str * s + v_dev * c
    
    # Term 2: Tangential Velocity (Omega x Radius)
    # Radius vector (in global frame) = (g_x - bx, g_z - bz)
    rx, rz = (g_x - bx), (g_z - bz)
    tan_x = -b_omega * rz
    tan_z =  b_omega * rx
    
    g_vx = b_vx + rv_x + tan_x
    g_vz = b_vz + rv_z + tan_z
    
    # Global Angular Velocity
    g_omega = b_omega + v_pitch
    
    # 4. Return format for teacher.step()
    # teacher.step(pos=(x,y), ang=val, vel_lin=(vx,vy), vel_ang=val)
    return (g_x, g_z), g_ang, (g_vx, g_vz), g_omega

# ==========================================
# 1. VISUALIZATION HELPER
# ==========================================
def visualize_trajectory(cycle, teacher, wing_gen, body_gen, wing_ctrl, body_ctrl, 
                         ramp_duration, steps_warmup, steps_collect, save_dir):
    filename = os.path.join(save_dir, f"viz_cycle_{cycle:04d}.gif")
    print(f"--> Generating Visualization: {filename}")
    
    NX, NY = teacher.NX, teacher.NY
    phys_size = teacher.PHYS_SIZE
    half = phys_size / 2.0
    extent = [-half, half, -half, half]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    dummy_field = np.zeros((NX, NY))
    im = ax.imshow(dummy_field.T, cmap='bwr', vmin=-0.01, vmax=0.01, 
                   origin='lower', extent=extent, interpolation='bilinear')
    
    ax.grid(True, alpha=0.3)
    ghost_line, = ax.plot([], [], 'r:', linewidth=1.0, alpha=0.5, label='Ghost')
    real_line, = ax.plot([], [], 'k-', linewidth=1.5, alpha=0.6, label='Centerline')
    q_force = ax.quiver([0], [0], [0], [0], color='black', scale=1.0, scale_units='xy')
    
    # Info Text
    info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f"Cycle {cycle} | Warmup + Training")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    
    # --- SIMULATION RESET (Updated for Global Kinematics) ---
    g_pos, g_ang, _, _ = compute_global_kinematics(
        0.0, wing_gen, body_gen, wing_ctrl, body_ctrl, 0.0, 0.0
    )
    teacher.reset(g_pos[0], g_pos[1], g_ang)
    t = 0.0
    
    N, L = teacher.N_PTS, teacher.WING_LEN
    x_base, y_base = np.linspace(L/2, -L/2, N), np.zeros(N)
    
    # Define exact ramp function
    def get_ramp_state(time_val):
        if time_val >= ramp_duration: return 1.0, 0.0
        phase = (np.pi / 2.0) * (time_val / ramp_duration)
        val = np.sin(phase)**2
        rate = 2.0 * np.sin(phase) * np.cos(phase) * ((np.pi / 2.0) / ramp_duration)
        return val, rate

    # Run viz integration at roughly LBM step size for smoothness
    dt_lbm = teacher.DT
    total_time = ramp_duration * 2.0 # Warmup + Collect
    total_lbm_steps = int(total_time / dt_lbm)
    
    render_skip = max(1, total_lbm_steps // 60)
    total_frames = total_lbm_steps // render_skip
    
    def update(frame):
        nonlocal t
        current_force = np.zeros(2)
        
        for _ in range(render_skip):
            r_val, r_rate = get_ramp_state(t)
            
            # [FIX] Use compute_global_kinematics
            g_pos, g_ang, g_lin, g_rot = compute_global_kinematics(
                t, wing_gen, body_gen, wing_ctrl, body_ctrl, r_val, r_rate
            )
            teacher.step(g_pos, g_ang, g_lin, g_rot)
            
            _, _, forces, _, _ = teacher.get_observation()
            current_force = np.sum(forces, axis=0)
            
            t += teacher.DT

        pts, _, _, _, _ = teacher.get_observation()
        u_field = teacher.u.to_numpy()
        
        ux, uy = u_field[:, :, 0], u_field[:, :, 1]
        curl = np.gradient(uy, axis=0) - np.gradient(ux, axis=1)
        im.set_array(curl.T)
        
        real_line.set_data(pts[:, 0], pts[:, 1])
        
        # [FIX] Ghost Line Drawing
        r_val, r_rate = get_ramp_state(t)
        g_pos, g_ang, _, _ = compute_global_kinematics(
            t, wing_gen, body_gen, wing_ctrl, body_ctrl, r_val, r_rate
        )
        c, s = np.cos(g_ang), np.sin(g_ang)
        gx = g_pos[0] + x_base * c - y_base * s
        gy = g_pos[1] + x_base * s + y_base * c
        ghost_line.set_data(gx, gy)
        
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        q_force.set_offsets([cx, cy])
        q_force.set_UVC(current_force[0] * 0.05, current_force[1] * 0.05)
        
        phase_name = "WARMUP" if t < ramp_duration else "TRAIN"
        info_text.set_text(f"Time: {t:.4f} s\nPhase: {phase_name}\nFx: {current_force[0]:.3f} N\nFy: {current_force[1]:.3f} N")
        
        return im, real_line, ghost_line, q_force, info_text

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    ani.save(filename, writer='pillow', fps=15)
    plt.close('all') 
    print(f"--> Saved {filename}")

# ==========================================
# 2. THE STUDENT (ResNet + NaN Safe + Accel)
# ==========================================
class ResNetBlock(hk.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

    def __call__(self, x):
        shortcut = x
        x = hk.Conv1D(output_channels=self.channels, kernel_shape=self.kernel_size, padding='SAME')(x)
        x = jax.nn.gelu(x)
        x = hk.Conv1D(output_channels=self.channels, kernel_shape=self.kernel_size, padding='SAME')(x)
        return jax.nn.gelu(x + shortcut)

class FluidSurrogateResNet(hk.Module):
    def __init__(self, n_points, hidden_dim=64):
        super().__init__()
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        
    def __call__(self, x):
        batch_size = x.shape[0]
        # Input has 6 channels: Pos(2), Vel(2), Acc(2)
        n2 = self.n_points * 2
        
        pos_flat = x[:, :n2]
        vel_flat = x[:, n2 : n2*2]
        acc_flat = x[:, n2*2 :] # New input slice
        
        pos = jnp.reshape(pos_flat, (batch_size, self.n_points, 2))
        vel = jnp.reshape(vel_flat, (batch_size, self.n_points, 2))
        acc = jnp.reshape(acc_flat, (batch_size, self.n_points, 2))
        
        # Concatenate along channel dimension
        h = jnp.concatenate([pos, vel, acc], axis=-1)
        
        h = hk.Conv1D(output_channels=self.hidden_dim, kernel_shape=3, padding='SAME')(h)
        h = jax.nn.gelu(h)
        for _ in range(3):
            h = ResNetBlock(channels=self.hidden_dim, kernel_size=3)(h)
        pred_forces = hk.Conv1D(output_channels=2, kernel_shape=3, padding='SAME')(h)
        return jnp.reshape(pred_forces, (batch_size, -1))

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_online():
    print("=== STARTING JAX ONLINE TRAINING (SUB-SAMPLED + BACKWARD FD) ===")
    
    ckpt_dir = "checkpoints"
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pkl")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    teacher = TaichiFluidEngine()
    kinematics_gen = WingKinematics()
    body_gen = BodyKinematics()
    
    # --- TIME STEP CONFIGURATION ---
    DT_LBM = teacher.DT              # 3e-6
    SUBSAMPLE_RATE = 10              # Sample every 10 LBM steps
    DT_MODEL = DT_LBM * SUBSAMPLE_RATE # 3e-5 (Target for model)
    print(f"--> LBM DT: {DT_LBM:.2e} | Model DT: {DT_MODEL:.2e} | Ratio: {SUBSAMPLE_RATE}")

    # Constants
    NORM_POS = teacher.PHYS_SIZE 
    NORM_VEL = 10.0
    NORM_ACC = 1000.0 # High value due to 1/dt division
    NORM_FORCE = 100.0
    N_PTS = teacher.N_PTS
    
    # Input: Pos(2) + Vel(2) + Acc(2) = 6 channels
    INPUT_DIM = N_PTS * 6 
    OUTPUT_DIM = N_PTS * 2
    
    # Init Student
    def forward_fn(x):
        model = FluidSurrogateResNet(n_points=N_PTS, hidden_dim=64)
        return model(x)

    model = hk.without_apply_rng(hk.transform(forward_fn))
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.zeros([1, INPUT_DIM])
    
    # --- OPTIMIZER ---
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-4)
    )
    
    # --- RESUME / INIT ---
    if os.path.exists(ckpt_path):
        print(f"--> Found checkpoint at {ckpt_path}. Resuming...")
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
            params = data['params']
            opt_state = data['opt_state']
            start_cycle = data['cycle'] + 1
    else:
        print("--> No checkpoint found. Starting fresh.")
        params = model.init(rng, dummy_input)
        opt_state = optimizer.init(params)
        start_cycle = 0

    # Ring Buffer
    BUFFER_SIZE = 200_000
    buffer_x = np.zeros((BUFFER_SIZE, INPUT_DIM), dtype=np.float32)
    buffer_y = np.zeros((BUFFER_SIZE, OUTPUT_DIM), dtype=np.float32)
    buffer_ptr = 0
    buffer_len = 0
    
    # --- JIT TRAIN STEP ---
    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(p, x, y):
            pred_y = model.apply(p, x)
            return jnp.mean(jnp.square(pred_y - y))
        
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_x, batch_y)
        
        is_nan = jnp.isnan(loss)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        final_params = jax.tree_util.tree_map(lambda n, o: jnp.where(is_nan, o, n), new_params, params)
        final_opt_state = jax.tree_util.tree_map(lambda n, o: jnp.where(is_nan, o, n), new_opt_state, opt_state)
        
        return final_params, final_opt_state, loss

    start_time = time.time()
    
    # --- MAIN LOOP ---
    for cycle in range(start_cycle, 6001): 
        
        # =====================================================================
        # 1. RANDOMIZE CONTROLS
        # =====================================================================
        
        # --- A. WING KINEMATICS (Original Distro) ---
        raw_freq_in    = np.random.uniform(-1.0, 1.0)
        raw_str_in     = np.random.uniform(-2.0, 2.0)
        raw_aoa_d_in   = np.random.uniform(-1.3, 1.2)
        raw_aoa_u_in   = np.random.uniform(-1.3, 1.2)
        raw_pitch_ph   = np.random.uniform(-1.0, 1.0)
        raw_dev_amp_in = np.random.uniform(-1.0, 1.0)
        raw_f_dev_in   = np.random.uniform(-1.0, 1.0)
        raw_dev_ph_in  = np.random.uniform(-1.0, 1.0)

        # Derived Wing Params
        freq = np.clip(20.0 + raw_freq_in * 10.0, 10.0, 30.0)
        stroke_amp = np.clip(0.02 + raw_str_in * 0.005, 0.01, 0.03)
        aoa_down = np.clip(0.8 + raw_aoa_d_in, -0.5, 2.0)
        aoa_up   = np.clip(0.8 + raw_aoa_u_in, -0.5, 2.0)
        pitch_phase = np.clip(raw_pitch_ph * 0.8, -0.8, 0.8)
        dev_amp = np.clip(raw_dev_amp_in * 0.008, -0.008, 0.008)
        
        # Deviation Frequency (Original Logic)
        f_dev_ratio = np.clip(2.0 + raw_f_dev_in * 0.3, 1.7, 2.3)
        f_dev = freq * f_dev_ratio
        dev_phase = np.clip(raw_dev_ph_in * 1.57, -1.57, 1.57)

        # Pack Wing Controls
        # CRITICAL: Must match the order in WingKinematics._compute_raw_wing_state
        # Signature: (t, freq, stroke_amp, pitch_phase, dev_amp, f_dev, dev_phase, aoa_down, aoa_up)
        wing_controls = np.array([
            freq, stroke_amp, pitch_phase, 
            dev_amp, f_dev, dev_phase, 
            aoa_down, aoa_up
        ])

        # --- B. BODY KINEMATICS (Global Jitter) ---
        # Simulating hovering drift and slight body rotations
        b_Ax  = np.random.uniform(-0.01, 0.01) # +/- 10mm translation (X)
        b_fx  = np.random.uniform(0.0, 40.0)      # 0-40 Hz
        b_phx = np.random.uniform(-3.14, 3.14)
        
        b_Az  = np.random.uniform(-0.01, 0.01) # +/- 10mm translation (Z)
        b_fz  = np.random.uniform(0.0, 40.0)
        b_phz = np.random.uniform(-3.14, 3.14)
        
        b_Ap  = np.random.uniform(-1.0, 1.0)   # +/- ~57 deg pitch
        b_fp  = np.random.uniform(0.0, 40.0)
        b_php = np.random.uniform(-3.14, 3.14)

        body_controls = np.array([
            b_Ax, b_fx, b_phx,
            b_Az, b_fz, b_phz,
            b_Ap, b_fp, b_php
        ])

        # =====================================================================
        # 2. LOGGING
        # =====================================================================
        print("-" * 60)
        print(f"CYCLE {cycle} CONFIGURATION:")
        print(">>> WING (Local):")
        print(f"    Freq: {freq:.2f} Hz | Stroke: {stroke_amp*1000:.1f} mm | Pitch Ph: {pitch_phase:.2f}")
        print(f"    AoA: [{aoa_down:.2f}, {aoa_up:.2f}] | Dev: {dev_amp*1000:.1f} mm ({f_dev_ratio:.2f}x)")
        print(">>> BODY (Global):")
        print(f"    X-Jitter: {b_Ax*1000:.1f}mm @ {b_fx:.1f}Hz")
        print(f"    Z-Jitter: {b_Az*1000:.1f}mm @ {b_fz:.1f}Hz")
        print(f"    Pitching: {np.degrees(b_Ap):.1f}deg @ {b_fp:.1f}Hz")
        print("-" * 60)
        
        # 2. Calculate Timing
        period = 1.0 / freq
        
        # Warmup uses raw LBM steps
        warmup_steps_lbm = int(period / DT_LBM)
        
        # Collection uses MODEL steps (subsampled)
        collect_steps_model = int(period / DT_MODEL)
        
        ramp_duration = period 
        
        def get_ramp_state(t_val):
            if t_val >= ramp_duration: return 1.0, 0.0
            phase = (np.pi / 2.0) * (t_val / ramp_duration)
            val = np.sin(phase)**2
            rate = 2.0 * np.sin(phase) * np.cos(phase) * ((np.pi / 2.0) / ramp_duration)
            return val, rate
        
        # 3. Visualization Check
        if cycle % 100 == 0:
            # UPDATED CALL: Pass body_gen, wing_controls, and body_controls
            visualize_trajectory(
                cycle, 
                teacher, 
                kinematics_gen, 
                body_gen,          # Added body generator
                wing_controls,     # Renamed from 'controls'
                body_controls,     # Added body controls
                ramp_duration, 
                warmup_steps_lbm, 
                collect_steps_model * SUBSAMPLE_RATE, 
                ckpt_dir
            )

        # ==========================================
        # 4. Simulation Execution
        # ==========================================
        try:
            # A. RESET
            # Calculate initial Global State (t=0, ramp=0)
            g_pos, g_ang, _, _ = compute_global_kinematics(
                0.0, kinematics_gen, body_gen, wing_controls, body_controls, 0.0, 0.0
            )
            teacher.reset(g_pos[0], g_pos[1], g_ang)
            t = 0.0
            
            # Helper to store velocity from previous *Model Step* (for Finite Difference)
            prev_model_vels = np.zeros_like(teacher.s_vel)

            # B. WARMUP (Raw LBM Steps)
            for _ in range(warmup_steps_lbm):
                r_val, r_rate = get_ramp_state(t)
                
                # Update Simulation with GLOBAL Kinematics (Body + Wing)
                g_pos, g_ang, g_lin, g_rot = compute_global_kinematics(
                    t, kinematics_gen, body_gen, wing_controls, body_controls, r_val, r_rate
                )
                teacher.step(g_pos, g_ang, g_lin, g_rot)
                t += DT_LBM
            
            # Capture state at end of warmup
            _, prev_model_vels, _, _, _ = teacher.get_observation()

            # C. DATA COLLECTION (Sub-sampled Loop)
            traj_x = np.zeros((collect_steps_model, INPUT_DIM), dtype=np.float32)
            traj_y = np.zeros((collect_steps_model, OUTPUT_DIM), dtype=np.float32)
            
            # Pre-calc span offsets for reconstructing body-frame points
            span_offsets = np.linspace(teacher.WING_LEN/2.0, -teacher.WING_LEN/2.0, teacher.N_PTS)

            max_f_in_cycle = 0.0

            for k in range(collect_steps_model):
                
                # Inner Loop: Run LBM 'SUBSAMPLE_RATE' times
                for _ in range(SUBSAMPLE_RATE):
                    r_val, r_rate = get_ramp_state(t)
                    g_pos, g_ang, g_lin, g_rot = compute_global_kinematics(
                        t, kinematics_gen, body_gen, wing_controls, body_controls, r_val, r_rate
                    )
                    teacher.step(g_pos, g_ang, g_lin, g_rot)
                    t += DT_LBM

                # --- SNAPSHOT (Every 10 LBM steps) ---
                # 1. Get Simulation Truth (Global Frame)
                # pts_global: Absolute pos (includes body jitter) -> UNUSED for Training Input
                # vels_global: Absolute flow velocity -> USED (Airspeed)
                # forces: Aerodynamic forces -> TARGET
                pts_global, vels_global, forces, _, step_max = teacher.get_observation()

                if step_max > max_f_in_cycle:
                    max_f_in_cycle = step_max
                
                # We need the current Body Angle (b_theta) from the kinematic generato
                # Re-calculate body pitch for this instant t
                b_pos_now, _ = body_gen.get_body_state(t, body_controls)
                b_theta_now = b_pos_now[2] * r_val # Apply ramp scaling if necessary

                # Rotation Matrix (Global to Body = Rotate by -theta)
                c_b, s_b = np.cos(-b_theta_now), np.sin(-b_theta_now)
                
                # Apply rotation to every node's force vector
                # forces shape is (N_PTS, 2) where col 0 is Fx, col 1 is Fz
                fx_global = forces[:, 0]
                fz_global = forces[:, 1]

                fx_body = fx_global * c_b - fz_global * s_b
                fz_body = fx_global * s_b + fz_global * c_b
                
                # Stack them back together
                forces_body_frame = np.stack([fx_body, fz_body], axis=1)

                # 2. Compute Body-Frame Position (The "Correct" Training Input)
                r_val, r_rate = get_ramp_state(t)
                
                # [FIX START] Correct Unpacking & Orientation -----------------
                # get_local_pose returns: (p_final, v_final) where p_final is [stroke, dev, pitch]
                w_pos, _ = kinematics_gen.get_local_pose_and_velocity(
                    t, wing_controls, r_val, r_rate
                )
                
                # Unpack the JAX array
                l_str, l_dev, l_pitch_raw = w_pos
                
                # Apply the Pi/2 offset to match the LBM Engine's global convention
                # (Without this, the reconstructed wing shape is rotated 90 deg relative to reality)
                l_pitch_body_frame = l_pitch_raw + (np.pi / 2.0)
                
                # Reconstruct points: Wing is a rigid line rotated by pitch + translated by str/dev
                c_w, s_w = np.cos(l_pitch_body_frame), np.sin(l_pitch_body_frame)
                
                # X_body = Stroke_Translation + Span * cos(Pitch)
                xs_body = l_str + span_offsets * c_w
                # Y_body = Dev_Translation + Span * sin(Pitch)
                ys_body = l_dev + span_offsets * s_w
                
                pts_body_frame = np.stack([xs_body, ys_body], axis=1)
                # [FIX END] ---------------------------------------------------
                
                # 3. Compute Global Acceleration (Backward Finite Difference)
                accels = (vels_global - prev_model_vels) / DT_MODEL
                
                # Transform Vel & Acc -> Body Frame
                # We reuse c_b, s_b (cos(-theta), sin(-theta)) calculated earlier for forces
                
                # 1. Rotate Velocity
                vx_glob = vels_global[:, 0]
                vz_glob = vels_global[:, 1]
                
                vx_body = vx_glob * c_b - vz_glob * s_b
                vz_body = vx_glob * s_b + vz_glob * c_b
                vels_body_frame = np.stack([vx_body, vz_body], axis=1)

                # 2. Rotate Acceleration
                ax_glob = accels[:, 0]
                az_glob = accels[:, 1]

                ax_body = ax_glob * c_b - az_glob * s_b
                az_body = ax_glob * s_b + az_glob * c_b
                accels_body_frame = np.stack([ax_body, az_body], axis=1)

                # Update prev velocity
                prev_model_vels = vels_global.copy()
                
                # 4. Check for NaNs
                if np.isnan(pts_global).any() or np.isnan(forces).any():
                    raise ValueError("Simulation Exploded (NaN detected)")
                
                # 5. Pack Data
                flat_pts = pts_body_frame.flatten() / NORM_POS
                flat_vels = vels_body_frame.flatten() / NORM_VEL
                flat_accs = accels_body_frame.flatten() / NORM_ACC
                flat_forces = forces_body_frame.flatten() * NORM_FORCE
                
                traj_x[k] = np.concatenate([flat_pts, flat_vels, flat_accs])
                traj_y[k] = flat_forces
            
            # D. BUFFER UPDATE
            end_idx = buffer_ptr + collect_steps_model
            if end_idx <= BUFFER_SIZE:
                buffer_x[buffer_ptr:end_idx] = traj_x
                buffer_y[buffer_ptr:end_idx] = traj_y
            else:
                overflow = end_idx - BUFFER_SIZE
                split = collect_steps_model - overflow
                buffer_x[buffer_ptr:] = traj_x[:split]
                buffer_y[buffer_ptr:] = traj_y[:split]
                buffer_x[:overflow] = traj_x[split:]
                buffer_y[:overflow] = traj_y[split:]
            
            buffer_ptr = (buffer_ptr + collect_steps_model) % BUFFER_SIZE
            buffer_len = min(buffer_len + collect_steps_model, BUFFER_SIZE)
            
        except ValueError as e:
            print(f"Skipping Cycle {cycle}: {e}")
            continue

        # 5. Training
        loss_val = 0.0
        if buffer_len > 5000: 
            for _ in range(50):
                indices = np.random.randint(0, buffer_len, size=256)
                batch_x = jnp.array(buffer_x[indices])
                batch_y = jnp.array(buffer_y[indices])
                
                if jnp.isnan(batch_x).any(): continue
                
                params, opt_state, loss_val = train_step(params, opt_state, batch_x, batch_y)
        
        # 6. Logging & Saving
        if cycle % 1 == 0:
            elapsed = time.time() - start_time
            saturation = (max_f_in_cycle / 0.05) * 100
            
            # Convert to readable units for logging
            str_mm = stroke_amp * 1000.0
            dev_mm = dev_amp * 1000.0
            body_deg = np.degrees(b_Ap) # Body pitch amplitude

            # Create a concise log string
            # Freq: Hz | Str: mm | Dev: mm | Body: deg | MaxF: LB Units (%) | Buff: count | Loss: val | Time: s
            log_msg = (
                f"Cycle {cycle:04d} | "
                f"Freq: {freq:.1f}Hz | "
                f"Str: {str_mm:.1f}mm | "
                f"Dev: {dev_mm:.1f}mm | "
                f"Body: {body_deg:.1f}° | "
                f"MaxF_LB: {max_f_in_cycle:.3f} ({saturation:.1f}%) | "
                f"Buff: {buffer_len} | "
                f"Loss: {loss_val:.2e} | "
                f"Time: {elapsed:.1f}s"
            )
            print(log_msg)
            
        if cycle % 20 == 0:
             save_data = {
                 'params': params,
                 'opt_state': opt_state,
                 'cycle': cycle
             }
             with open(ckpt_path, "wb") as f:
                 pickle.dump(save_data, f)
             print(f"--> Checkpoint saved: {ckpt_path}")

if __name__ == "__main__":
    train_online()