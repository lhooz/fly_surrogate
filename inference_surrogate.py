import os
# Force JAX to CPU (Simpler for inference/viz)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pickle
import time  # <--- Global import for timing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- IMPORT ENGINE ---
try:
    import taichi as ti
    try:
        ti.init(arch=ti.gpu, device_memory_fraction=0.5)
    except:
        ti.init(arch=ti.cpu)
    from environment_engine import TaichiFluidEngine
except ImportError:
    print("Error: 'environment_engine.py' not found.")
    exit()

# ==========================================
# 1. KINEMATICS & MODEL (MATCHING TRAIN)
# ==========================================
class WingKinematics:
    def get_pose_and_velocity(self, t, controls, ramp_val=1.0, ramp_rate=0.0):
        f_flap, raw_str_amp, raw_aoa_d, raw_aoa_u, phase_pitch, raw_dev_amp, f_dev, phase_dev = controls
        
        stroke_amp = raw_str_amp * ramp_val
        dev_amp    = raw_dev_amp * ramp_val
        aoa_down   = raw_aoa_d   * ramp_val
        aoa_up     = raw_aoa_u   * ramp_val
        
        TWO_PI = 2.0 * np.pi
        phi_flap = TWO_PI * f_flap * t
        phi_dev  = TWO_PI * f_dev * t + phase_dev
        
        # Stroke
        sin_f, cos_f = np.sin(phi_flap), np.cos(phi_flap)
        local_stroke = stroke_amp * sin_f
        v_stroke_local = (raw_str_amp * ramp_rate * sin_f) + (stroke_amp * (TWO_PI * f_flap) * cos_f)
        
        # Deviation
        sin_d, cos_d = np.sin(phi_dev), np.cos(phi_dev)
        local_dev = -dev_amp * sin_d
        v_dev_local = (-raw_dev_amp * ramp_rate * sin_d) + (-dev_amp * (TWO_PI * f_dev) * cos_d)
        
        # Pitch
        k_sharpness = 10.0
        tanh_val = np.tanh(k_sharpness * cos_f)
        switch = 0.5 * (1.0 + tanh_val)
        sech_sq = 1.0 - tanh_val**2
        phi_dot = TWO_PI * f_flap
        d_switch_dt = 0.5 * sech_sq * (k_sharpness * -sin_f * phi_dot)
        
        current_mid_aoa = aoa_up + (aoa_down - aoa_up) * switch
        pitch = -current_mid_aoa * np.cos(phi_flap + phase_pitch)
        wing_ang = pitch + (np.pi / 2.0)
        
        # Pitch Velocity
        raw_mid_aoa = raw_aoa_u + (raw_aoa_d - raw_aoa_u) * switch
        d_raw_mid_aoa_dt = (raw_aoa_d - raw_aoa_u) * d_switch_dt
        
        M_t = current_mid_aoa
        dM_dt = (raw_mid_aoa * ramp_rate) + (d_raw_mid_aoa_dt * ramp_val)
        theta = phi_flap + phase_pitch
        w_total = -(dM_dt * np.cos(theta) - M_t * np.sin(theta) * phi_dot)
        
        return (local_stroke, local_dev), wing_ang, (v_stroke_local, v_dev_local), w_total

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
        n2 = self.n_points * 2
        pos_flat = x[:, :n2]
        vel_flat = x[:, n2 : n2*2]
        acc_flat = x[:, n2*2 :]
        pos = jnp.reshape(pos_flat, (batch_size, self.n_points, 2))
        vel = jnp.reshape(vel_flat, (batch_size, self.n_points, 2))
        acc = jnp.reshape(acc_flat, (batch_size, self.n_points, 2))
        h = jnp.concatenate([pos, vel, acc], axis=-1)
        h = hk.Conv1D(output_channels=self.hidden_dim, kernel_shape=3, padding='SAME')(h)
        h = jax.nn.gelu(h)
        for _ in range(3):
            h = ResNetBlock(channels=self.hidden_dim, kernel_size=3)(h)
        pred_forces = hk.Conv1D(output_channels=2, kernel_shape=3, padding='SAME')(h)
        return jnp.reshape(pred_forces, (batch_size, -1))

# ==========================================
# 2. VALIDATION LOGIC
# ==========================================
def run_validation():
    print("=== STARTING DETAILED VALIDATION (INDIVIDUAL FORCES) ===")
    
    ckpt_path = "checkpoints/checkpoint.pkl"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
        params = data['params']
        print(f"--> Loaded checkpoint from Cycle {data['cycle']}")

    engine = TaichiFluidEngine()
    kinematics = WingKinematics()
    
    # Constants
    DT_LBM = engine.DT
    SUBSAMPLE_RATE = 10
    DT_MODEL = DT_LBM * SUBSAMPLE_RATE
    NORM_POS = engine.PHYS_SIZE 
    NORM_VEL = 10.0
    NORM_ACC = 1000.0 
    NORM_FORCE = 100.0
    N_PTS = engine.N_PTS

    def forward_fn(x):
        model = FluidSurrogateResNet(n_points=N_PTS, hidden_dim=64)
        return model(x)
    model = hk.without_apply_rng(hk.transform(forward_fn))

    def run_case(case_idx):
        raw_freq      = np.random.uniform(-1.2, 1.2)
        raw_str       = np.random.uniform(-2.5, 0.8)
        raw_aoa_d     = np.random.uniform(-0.6, 0.6)
        raw_aoa_u     = np.random.uniform(-0.6, 0.6)
        raw_pitch_ph = np.random.uniform(-1.2, 0.6)
        raw_dev_amp  = np.random.uniform(-0.1, 1.0)
        raw_f_dev    = np.random.uniform(-1.0, 1.0)
        raw_dev_ph   = np.random.uniform(-1.0, 1.0)
        
        freq = np.clip(25.0 + raw_freq * 10.0, 15.0, 35.0)
        stroke_amp = np.clip(0.022 + raw_str * 0.005, 0.012, 0.026)
        aoa_down = np.clip(0.8 + raw_aoa_d, 0.3, 1.8)
        aoa_up   = np.clip(0.8 + raw_aoa_u, 0.3, 1.8)
        pitch_phase = np.clip(0.3 + raw_pitch_ph * 0.5, -0.4, 0.6)
        dev_amp = np.clip(raw_dev_amp * 0.008, 0.0, 0.008)
        f_dev_ratio = np.clip(2.0 + raw_f_dev * 0.2, 1.8, 2.2)
        f_dev = freq * f_dev_ratio
        dev_phase = np.clip(raw_dev_ph * 1.57, -1.57, 1.57)
        
        controls = np.array([freq, stroke_amp, aoa_down, aoa_up, pitch_phase, dev_amp, f_dev, dev_phase])

        print("-" * 60)
        print(f"CASE {case_idx} KINEMATICS:")
        print(f"  Freq: {freq:.2f} Hz | Stroke: {stroke_amp*1000:.1f} mm")
        print("-" * 60)

        # Setup
        period = 1.0 / freq
        warmup_lbm = int(period / DT_LBM)
        collect_model = int((period * 2.5) / DT_MODEL) 
        ramp_dur = period

        def get_ramp(t_val):
            if t_val >= ramp_dur: return 1.0, 0.0
            p = (np.pi/2)*(t_val/ramp_dur)
            return np.sin(p)**2, 2*np.sin(p)*np.cos(p)*((np.pi/2)/ramp_dur)

        # Reset
        p0, a0, _, _ = kinematics.get_pose_and_velocity(0.0, controls, 0.0, 0.0)
        engine.reset(p0[0], p0[1], a0)
        t = 0.0
        
        history = {
            'curl': [], 'pts': [], 'time': [],
            'gt_forces_all': [],    
            'pred_forces_all': [], 
            'gt_sum': [], 'pred_sum': []
        }

        # --- 1. WARMUP (Silent) ---
        print(f"--> Warming up ({warmup_lbm} steps)...")
        prev_model_vels = np.zeros((N_PTS, 2))
        
        for _ in range(warmup_lbm):
            r_val, r_rate = get_ramp(t)
            pos, ang, vl, va = kinematics.get_pose_and_velocity(t, controls, r_val, r_rate)
            engine.step(pos, ang, vl, va)
            t += DT_LBM
            # [FIX] No print here

        _, prev_model_vels, _, _, _ = engine.get_observation()

        # --- 2. COLLECTION (With Progress Bar) ---
        print(f"--> Collecting Data ({collect_model} steps)...")
        
        dummy_in = jnp.zeros((1, N_PTS*6))
        _ = model.apply(params, dummy_in)
        
        for step in range(collect_model):
            for _ in range(SUBSAMPLE_RATE):
                r_val, r_rate = get_ramp(t)
                pos, ang, vl, va = kinematics.get_pose_and_velocity(t, controls, r_val, r_rate)
                engine.step(pos, ang, vl, va)
                t += DT_LBM
            
            # [FIX] Progress Bar Here
            if step % 50 == 0:
                print(f"\r    Progress: {step}/{collect_model}", end="", flush=True)

            pts, vels, forces_gt, _, _ = engine.get_observation()
            
            # Input Prep
            accels = (vels - prev_model_vels) / DT_MODEL
            prev_model_vels = vels.copy()

            in_pos = pts.flatten() / NORM_POS
            in_vel = vels.flatten() / NORM_VEL
            in_acc = accels.flatten() / NORM_ACC
            x_in = jnp.concatenate([in_pos, in_vel, in_acc])[None, :]

            # Inference
            y_pred = model.apply(params, x_in)
            y_pred.block_until_ready() 
            forces_pred = np.array(y_pred[0]).reshape(N_PTS, 2) / NORM_FORCE
            
            # Store
            history['pts'].append(pts.copy())
            history['gt_forces_all'].append(forces_gt.copy())
            history['pred_forces_all'].append(forces_pred.copy())
            history['gt_sum'].append(np.sum(forces_gt, axis=0))
            history['pred_sum'].append(np.sum(forces_pred, axis=0))
            history['time'].append(t)
            
            u_field = engine.u.to_numpy()
            ux, uy = u_field[:, :, 0], u_field[:, :, 1]
            curl = np.gradient(uy, axis=0) - np.gradient(ux, axis=1)
            history['curl'].append(curl.T)
        
        print("\n--> Case Completed.")
        return history

    case_1 = run_case(1)
    case_2 = run_case(2)
    
    # ==========================================
    # VISUALIZATION (ENHANCED DARK MODE)
    # ==========================================
    print("--> Generating Professional Plot (This may take a minute)...")
    
    # Dark Theme Colors
    bg_color = '#1e1e1e'
    text_color = '#dcdcdc'
    grid_color = '#444444'
    gt_color = '#00ffaa'   # Neon Green
    pred_color = '#ff00aa' # Neon Magenta
    
    plt.rcParams.update({
        'axes.facecolor': bg_color,
        'figure.facecolor': bg_color,
        'axes.edgecolor': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'text.color': text_color,
        'axes.titlecolor': text_color,
        'grid.color': grid_color,
        'legend.facecolor': '#333333',
        'legend.edgecolor': grid_color
    })

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0.15, hspace=0.25)
    cases = [case_1, case_2]
    
    plots = []
    
    dummy_pos = np.zeros((N_PTS, 2))
    dummy_uv  = np.zeros((N_PTS, 2))

    for i in range(2):
        # --- LEFT: SPATIAL (Fluid + Vectors) ---
        ax_l = axes[i, 0]
        ext = [-engine.PHYS_SIZE/2, engine.PHYS_SIZE/2, -engine.PHYS_SIZE/2, engine.PHYS_SIZE/2]
        
        # [TWEAK] Vorticity Range (Tighter range = Brighter Vortices)
        im = ax_l.imshow(cases[i]['curl'][0], cmap='RdBu', vmin=-0.005, vmax=0.005, 
                         origin='lower', extent=ext, alpha=0.9)
        
        # Wing Drawing
        wing_shadow, = ax_l.plot([], [], 'k-', lw=5.0, alpha=0.7)
        wing_ln, = ax_l.plot([], [], 'w.-', lw=2.0, markersize=4, alpha=1.0)
        le_dot, = ax_l.plot([], [], 'o', color='#ff3333', markersize=6, label='Leading Edge')
        
        qv_gt = ax_l.quiver(dummy_pos[:,0], dummy_pos[:,1], dummy_uv[:,0], dummy_uv[:,1], 
                            color=gt_color, scale=0.5, scale_units='xy', width=0.004, 
                            headwidth=4, headlength=5, label='Ground Truth')
        
        qv_pr = ax_l.quiver(dummy_pos[:,0], dummy_pos[:,1], dummy_uv[:,0], dummy_uv[:,1], 
                            color=pred_color, scale=0.5, scale_units='xy', width=0.003, alpha=0.9, 
                            headwidth=4, headlength=5, label='Prediction')
        
        ax_l.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax_l.set_title(f"Case {i+1}: Flow Field & Wing Pose", fontsize=11, fontweight='bold')
        ax_l.set_aspect('equal')
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        
        # --- RIGHT: TEMPORAL (Curves) ---
        ax_r = axes[i, 1]
        gt_sum = np.array(cases[i]['gt_sum'])
        pr_sum = np.array(cases[i]['pred_sum'])
        time_arr = np.array(cases[i]['time']) 
        
        ln_gt_y, = ax_r.plot([], [], color=gt_color, lw=2.0, label='Lift (GT)')
        ln_pr_y, = ax_r.plot([], [], color=pred_color, lw=1.5, ls='--', label='Lift (Pred)')
        ln_gt_x, = ax_r.plot([], [], color=gt_color, lw=1.0, alpha=0.5, label='Drag (GT)')
        ln_pr_x, = ax_r.plot([], [], color=pred_color, lw=1.0, ls='--', alpha=0.5, label='Drag (Pred)')
        
        ax_r.set_xlim(time_arr[0], time_arr[-1])
        all_vals = np.concatenate([gt_sum.flatten(), pr_sum.flatten()])
        y_max = np.max(all_vals) * 1.3
        y_min = np.min(all_vals) * 1.3
        ax_r.set_ylim(y_min, y_max)
        
        ax_r.grid(True, linestyle=':', alpha=0.6)
        ax_r.legend(loc='upper right', ncol=2, fontsize=9)
        ax_r.set_title(f"Case {i+1}: Integrated Force History", fontsize=11, fontweight='bold')
        ax_r.set_ylabel("Force (Newtons)")
        if i == 1: ax_r.set_xlabel("Time (seconds)")
        
        plots.append({
            'data': cases[i],
            'im': im, 
            'wing_grp': (wing_shadow, wing_ln, le_dot),
            'qv_gt': qv_gt, 'qv_pr': qv_pr,
            'ln_gt_y': ln_gt_y, 'ln_pr_y': ln_pr_y,
            'ln_gt_x': ln_gt_x, 'ln_pr_x': ln_pr_x,
            'gt_sum': gt_sum, 'pr_sum': pr_sum, 'time_arr': time_arr
        })

    def update(frame):
        artists = []
        for p in plots:
            d = p['data']
            if frame >= len(d['time']): continue
            
            # 1. Update Spatial
            p['im'].set_array(d['curl'][frame])
            
            # Wing Updates
            pts = d['pts'][frame]
            w_shad, w_main, w_le = p['wing_grp']
            
            w_shad.set_data(pts[:, 0], pts[:, 1])
            w_main.set_data(pts[:, 0], pts[:, 1])
            w_le.set_data([pts[0, 0]], [pts[0, 1]])
            
            # Vectors
            f_gt = d['gt_forces_all'][frame]
            f_pr = d['pred_forces_all'][frame]
            
            p['qv_gt'].set_offsets(pts)
            p['qv_gt'].set_UVC(f_gt[:, 0], f_gt[:, 1])
            p['qv_pr'].set_offsets(pts)
            p['qv_pr'].set_UVC(f_pr[:, 0], f_pr[:, 1])
            
            # 2. Update Temporal
            curr = slice(0, frame+1)
            t = p['time_arr'][curr]
            
            # [KEY FIX: Used 'pr_sum' instead of 'pred_sum']
            p['ln_gt_y'].set_data(t, p['gt_sum'][curr, 1])
            p['ln_pr_y'].set_data(t, p['pr_sum'][curr, 1])
            p['ln_gt_x'].set_data(t, p['gt_sum'][curr, 0])
            p['ln_pr_x'].set_data(t, p['pr_sum'][curr, 0])
            
            artists.extend([p['im'], w_shad, w_main, w_le, p['qv_gt'], p['qv_pr'], 
                            p['ln_gt_y'], p['ln_pr_y'], p['ln_gt_x'], p['ln_pr_x']])
        return artists

    total_frames = min(len(case_1['time']), len(case_2['time']))
    skip_step = 10 
    render_frames = range(0, total_frames, skip_step)
    
    print(f"--> Rendering Animation ({len(render_frames)} frames, skipping {skip_step})...")
    
    # Pass 'render_frames' instead of 'total_frames'
    ani = animation.FuncAnimation(fig, update, frames=render_frames, interval=40, blit=False)
    ani.save("inference.gif", writer='pillow', fps=30, dpi=100)
    print("--> Saved validation.gif")

if __name__ == "__main__":
    run_validation()