"""
F1 — Real Sobolev surrogate training with FINITE-DIFFERENCE teacher Jacobians.

The manuscript claims the Bio-Fluid Surrogate is trained via "Sobolev state-gradient
supervision" against the LBM solver. The shipped trainer (train_surrogate.py:427-429)
is pure MSE. This module implements the real thing:

  1. Run the Taichi-LBM teacher, collecting (x, F) snapshots exactly as train_surrogate does.
  2. At sampled snapshots, compute a finite-difference TEACHER Jacobian
        J_gt = d F_body / d q,  q = [stroke, dev, pitch]   (the manuscript's w_p pose subset)
     by saving the full fluid+structure state, re-running the model-step with the
     commanded pose perturbed +/- eps (mapped to the global ghost via the exact
     compute_global_kinematics transform), reading the resulting body-frame force,
     and restoring state. This is a real, physically meaningful state-gradient given
     the current wake (finite-difference, since the Taichi-LBM is not jax-traceable).
  3. Train two surrogates on identical data:
        - MSE-only           (lambda_g = 0)
        - Sobolev            (MSE + lambda_g * || J_pred - J_gt ||_F^2,  lambda_g = 0.1)
     where J_pred = jacrev over the SAME pose -> input reconstruction.
  4. Report the MEASURED held-out Jacobian Frobenius error for both models.
     These measured numbers replace the fabricated "26.18 -> 12.90" in main.tex:59.

Usage:
  python sobolev_train.py --validate        # tiny run to prove the mechanism
  python sobolev_train.py --cycles 30 --jac-every 4 --epochs 4000   # real reduced run
"""
import os, sys, time, json, pickle, argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse the EXACT teacher + kinematics + model from the shipped trainer so the
# Sobolev-trained surrogate is consistent with how hornetRL consumes fluid.pkl.
from environment_engine import TaichiFluidEngine
from train_surrogate import (
    WingKinematics, BodyKinematics, compute_global_kinematics,
    FluidSurrogateResNet,
)

PI = np.pi


def ramp_state(t_val, ramp_duration):
    """sin^2 warm-up ramp — identical to train_surrogate.get_ramp_state."""
    if t_val >= ramp_duration:
        return 1.0, 0.0
    phase = (PI / 2.0) * (t_val / ramp_duration)
    val = np.sin(phase) ** 2
    rate = 2.0 * np.sin(phase) * np.cos(phase) * ((PI / 2.0) / ramp_duration)
    return val, rate


def random_controls(rng):
    freq = float(np.clip(20.0 + rng.uniform(-1, 1) * 10.0, 10.0, 30.0))
    stroke_amp = float(np.clip(0.02 + rng.uniform(-2, 2) * 0.005, 0.01, 0.03))
    aoa_down = float(np.clip(0.8 + rng.uniform(-1.3, 1.2), -0.5, 2.0))
    aoa_up = float(np.clip(0.8 + rng.uniform(-1.3, 1.2), -0.5, 2.0))
    pitch_phase = float(np.clip(rng.uniform(-1, 1) * 0.8, -0.8, 0.8))
    dev_amp = float(np.clip(rng.uniform(-1, 1) * 0.008, -0.008, 0.008))
    f_dev = freq * float(np.clip(2.0 + rng.uniform(-1, 1) * 0.3, 1.7, 2.3))
    dev_phase = float(np.clip(rng.uniform(-1, 1) * 1.57, -1.57, 1.57))
    wing = np.array([freq, stroke_amp, pitch_phase, dev_amp, f_dev, dev_phase, aoa_down, aoa_up])
    body = np.array([
        rng.uniform(-0.01, 0.01), rng.uniform(0, 40), rng.uniform(-PI, PI),
        rng.uniform(-0.01, 0.01), rng.uniform(0, 40), rng.uniform(-PI, PI),
        rng.uniform(-1, 1), rng.uniform(0, 40), rng.uniform(-PI, PI),
    ])
    return wing, body, freq


# ---- teacher fluid+structure state save/restore (for FD Jacobian) ----
def save_state(teacher):
    return dict(
        f=teacher.f.to_numpy(), f_new=teacher.f_new.to_numpy(),
        rho=teacher.rho.to_numpy(), u=teacher.u.to_numpy(),
        s_pos=teacher.s_pos.copy(), s_vel=teacher.s_vel.copy(),
    )


def restore_state(teacher, st):
    teacher.f.from_numpy(st['f']); teacher.f_new.from_numpy(st['f_new'])
    teacher.rho.from_numpy(st['rho']); teacher.u.from_numpy(st['u'])
    teacher.s_pos[:] = st['s_pos']; teacher.s_vel[:] = st['s_vel']


def run_model_step(teacher, wing_gen, body_gen, wing_ctrl, body_ctrl,
                   t_start, dt_lbm, subsample, ramp_dur, dpose):
    """Run one model-step (subsample LBM steps) with commanded pose perturbed by
    dpose=[dstroke,ddev,dpitch] mapped to the global ghost. Returns (t_end, F_body_total)."""
    t = t_start
    for _ in range(subsample):
        rv, rr = ramp_state(t, ramp_dur)
        (gx, gz), gang, (gvx, gvz), gom = compute_global_kinematics(
            t, wing_gen, body_gen, wing_ctrl, body_ctrl, rv, rr)
        if dpose is not None:
            b_pos, _ = body_gen.get_body_state(t, body_ctrl)
            b_theta = b_pos[2] * rv
            c, s = np.cos(b_theta), np.sin(b_theta)
            ds, dd, dp = dpose
            gx = gx + (ds * c - dd * s)
            gz = gz + (ds * s + dd * c)
            gang = gang + dp
        teacher.step((gx, gz), gang, (gvx, gvz), gom)
        t += dt_lbm
    # body-frame total aero force at end of model-step
    b_pos_now, _ = body_gen.get_body_state(t, body_ctrl)
    rv_now, _ = ramp_state(t, ramp_dur)
    b_theta_now = b_pos_now[2] * rv_now
    cb, sb = np.cos(-b_theta_now), np.sin(-b_theta_now)
    _, _, forces, _, _ = teacher.get_observation()
    F = forces.sum(axis=0)
    F_body = np.array([F[0] * cb - F[1] * sb, F[0] * sb + F[1] * cb])
    return t, F_body


def collect(cycles, jac_every, eps, seed, norm):
    teacher = TaichiFluidEngine()
    wing_gen, body_gen = WingKinematics(), BodyKinematics()
    DT_LBM = teacher.DT
    SUB = 10
    DT_MODEL = DT_LBM * SUB
    N_PTS = teacher.N_PTS
    NORM_POS, NORM_VEL, NORM_ACC, NORM_FORCE = norm
    span = np.linspace(teacher.WING_LEN / 2.0, -teacher.WING_LEN / 2.0, N_PTS)
    rng = np.random.RandomState(seed)

    X, Y = [], []                 # all (x, force) snapshots for MSE training
    JX, JP, JG = [], [], []       # jac samples: x_nom, local_pose, J_gt

    for cyc in range(cycles):
        wing_ctrl, body_ctrl, freq = random_controls(rng)
        period = 1.0 / freq
        ramp_dur = period
        warmup = int(period / DT_LBM)
        collect_steps = int(period / DT_MODEL)

        (gx0, gz0), gang0, _, _ = compute_global_kinematics(
            0.0, wing_gen, body_gen, wing_ctrl, body_ctrl, 0.0, 0.0)
        teacher.reset(gx0, gz0, gang0)
        t = 0.0
        for _ in range(warmup):
            rv, rr = ramp_state(t, ramp_dur)
            g = compute_global_kinematics(t, wing_gen, body_gen, wing_ctrl, body_ctrl, rv, rr)
            teacher.step(*g); t += DT_LBM
        _, prev_vels, _, _, _ = teacher.get_observation()

        for k in range(collect_steps):
            do_jac = (k % jac_every == 0)
            st = save_state(teacher) if do_jac else None
            t_step = t

            # nominal model-step (advances main sim)
            t, _ = run_model_step(teacher, wing_gen, body_gen, wing_ctrl, body_ctrl,
                                  t_step, DT_LBM, SUB, ramp_dur, None)

            # build training input x exactly as train_surrogate does
            pts_g, vels_g, forces, _, _ = teacher.get_observation()
            b_pos_now, _ = body_gen.get_body_state(t, body_ctrl)
            rv_now, _ = ramp_state(t, ramp_dur)
            b_theta = b_pos_now[2] * rv_now
            cb, sb = np.cos(-b_theta), np.sin(-b_theta)
            w_pos, _ = wing_gen.get_local_pose_and_velocity(t, wing_ctrl, rv_now, 0.0)
            l_str, l_dev, l_pitch = float(w_pos[0]), float(w_pos[1]), float(w_pos[2])
            pg = l_pitch + PI / 2.0
            xs = l_str + span * np.cos(pg); ys = l_dev + span * np.sin(pg)
            pts_body = np.stack([xs, ys], 1)
            accels = (vels_g - prev_vels) / DT_MODEL
            vx, vz = vels_g[:, 0], vels_g[:, 1]
            vels_body = np.stack([vx * cb - vz * sb, vx * sb + vz * cb], 1)
            ax, az = accels[:, 0], accels[:, 1]
            accs_body = np.stack([ax * cb - az * sb, ax * sb + az * cb], 1)
            fx, fz = forces[:, 0], forces[:, 1]
            forces_body = np.stack([fx * cb - fz * sb, fx * sb + fz * cb], 1)
            prev_vels = vels_g.copy()
            if np.isnan(pts_g).any() or np.isnan(forces).any():
                if do_jac:
                    restore_state(teacher, st)
                break

            x = np.concatenate([pts_body.flatten() / NORM_POS,
                                vels_body.flatten() / NORM_VEL,
                                accs_body.flatten() / NORM_ACC]).astype(np.float32)
            y = (forces_body.flatten() * NORM_FORCE).astype(np.float32)
            X.append(x); Y.append(y)

            # finite-difference teacher Jacobian d F_body_total / d[stroke,dev,pitch]
            if do_jac:
                Jg = np.zeros((2, 3), dtype=np.float32)
                ok = True
                for j in range(3):
                    dp = np.zeros(3); dp[j] = eps
                    restore_state(teacher, st)
                    _, Fp = run_model_step(teacher, wing_gen, body_gen, wing_ctrl, body_ctrl,
                                           t_step, DT_LBM, SUB, ramp_dur, dp)
                    dp[j] = -eps
                    restore_state(teacher, st)
                    _, Fm = run_model_step(teacher, wing_gen, body_gen, wing_ctrl, body_ctrl,
                                           t_step, DT_LBM, SUB, ramp_dur, dp)
                    if not (np.isfinite(Fp).all() and np.isfinite(Fm).all()):
                        ok = False; break
                    Jg[:, j] = (Fp - Fm) / (2 * eps)
                restore_state(teacher, st)
                # re-advance nominal so main sim continues from the SAME state it had
                t = t_step
                t, _ = run_model_step(teacher, wing_gen, body_gen, wing_ctrl, body_ctrl,
                                      t_step, DT_LBM, SUB, ramp_dur, None)
                if ok:
                    JX.append(x); JP.append([l_str, l_dev, l_pitch]); JG.append(Jg)
        print(f"  cycle {cyc+1}/{cycles}: snapshots={len(X)} jac_samples={len(JX)}", flush=True)

    return (np.array(X), np.array(Y),
            np.array(JX), np.array(JP, dtype=np.float32), np.array(JG),
            dict(N_PTS=N_PTS, span=span, norm=norm))


def make_model(N_PTS):
    def fwd(x):
        return FluidSurrogateResNet(n_points=N_PTS, hidden_dim=64)(x)
    return hk.without_apply_rng(hk.transform(fwd))


def build_jpred_fn(model, N_PTS, span, norm):
    NORM_POS, NORM_VEL, NORM_ACC, NORM_FORCE = norm
    span_j = jnp.asarray(span)
    n2 = N_PTS * 2

    def F_body(params, q, x_nom, pose):
        l_str = pose[0] + q[0]; l_dev = pose[1] + q[1]; l_pitch = pose[2] + q[2]
        pg = l_pitch + PI / 2.0
        xs = l_str + span_j * jnp.cos(pg)
        ys = l_dev + span_j * jnp.sin(pg)
        pos_flat = jnp.stack([xs, ys], -1).reshape(-1) / NORM_POS
        x = jnp.concatenate([pos_flat, x_nom[n2:2 * n2], x_nom[2 * n2:]])[None, :]
        out = model.apply(params, x)[0].reshape(N_PTS, 2)
        return jnp.sum(out, axis=0) / NORM_FORCE   # physical body force

    def jpred(params, x_nom, pose):
        return jax.jacrev(F_body, argnums=1)(params, jnp.zeros(3), x_nom, pose)  # (2,3)

    return jax.jit(jax.vmap(jpred, in_axes=(None, 0, 0)))


def train(data, lambda_g, epochs, batch, lr, seed, label):
    X, Y, JX, JP, JG, meta = data
    N_PTS, span, norm = meta['N_PTS'], meta['span'], meta['norm']
    model = make_model(N_PTS)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros([1, X.shape[1]]))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = opt.init(params)
    jpred_batched = build_jpred_fn(model, N_PTS, span, norm)

    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    JXj, JPj, JGj = jnp.asarray(JX), jnp.asarray(JP), jnp.asarray(JG)
    rng = np.random.RandomState(seed)

    @jax.jit
    def step(params, opt_state, bx, by, jx, jp, jg):
        def loss_fn(p):
            mse = jnp.mean(jnp.square(model.apply(p, bx) - by))
            if lambda_g > 0:
                jpred = jpred_batched(p, jx, jp)
                sob = jnp.mean(jnp.sum((jpred - jg) ** 2, axis=(1, 2)))
            else:
                sob = 0.0
            return mse + lambda_g * sob, (mse, sob)
        (loss, (mse, sob)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, mse, sob

    nj = JX.shape[0]
    jbatch = min(32, nj)
    for e in range(epochs):
        idx = rng.randint(0, X.shape[0], size=batch)
        ji = rng.randint(0, nj, size=jbatch)
        params, opt_state, mse, sob = step(
            params, opt_state, Xj[idx], Yj[idx], JXj[ji], JPj[ji], JGj[ji])
        if e % max(1, epochs // 5) == 0 or e == epochs - 1:
            print(f"    [{label}] epoch {e}: mse={float(mse):.4f} sob={float(sob):.4f}", flush=True)
    return params, model, jpred_batched


def jac_frob_error(params, jpred_batched, JX, JP, JG):
    """Mean Frobenius-norm error between predicted and teacher Jacobians (held-out)."""
    jpred = np.asarray(jpred_batched(params, jnp.asarray(JX), jnp.asarray(JP)))
    diff = jpred - JG
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=(1, 2)))))


def make_figure2(p_mse, p_sob, jpred_fn, JX, JP, JG, err_mse, err_sob, out_pngs):
    """Figure 2: gradient fidelity, generated ENTIRELY from real held-out data."""
    Jp_mse = np.asarray(jpred_fn(p_mse, jnp.asarray(JX), jnp.asarray(JP))).reshape(-1)
    Jp_sob = np.asarray(jpred_fn(p_sob, jnp.asarray(JX), jnp.asarray(JP))).reshape(-1)
    Jg = np.asarray(JG).reshape(-1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3.4))
    lim = np.percentile(np.abs(Jg), 99) * 1.1
    a.plot([-lim, lim], [-lim, lim], 'k--', lw=1, zorder=1, label='ideal ($J_{pred}=J_{gt}$)')
    a.scatter(Jg, Jp_mse, s=6, alpha=0.25, c='#d1495b', label='MSE-only', zorder=2)
    a.scatter(Jg, Jp_sob, s=6, alpha=0.25, c='#1f77b4', label='Sobolev', zorder=3)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
    a.set_xlabel('True LBM Jacobian $\\partial F/\\partial w_p$ (finite difference)')
    a.set_ylabel('Surrogate Jacobian (autodiff)')
    a.set_title('Gradient fidelity (held-out poses)'); a.legend(fontsize=7, loc='upper left')
    b.bar(['MSE-only', 'Sobolev'], [err_mse, err_sob], color=['#d1495b', '#1f77b4'])
    b.set_ylabel('Mean Jacobian Frobenius error')
    b.set_title(f'{100*(err_mse-err_sob)/err_mse:.0f}% error reduction')
    for i, v in enumerate([err_mse, err_sob]):
        b.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    for p in out_pngs:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"  figure written: {out_pngs}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--from-dataset', type=str, default=None,
                    help='skip LBM collection; load X,Y,JX,JP,JG from this .npz (fast, reproducible figure)')
    ap.add_argument('--cycles', type=int, default=30)
    ap.add_argument('--jac-every', type=int, default=4)
    ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lambda-g', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sobolev_outputs'))
    args = ap.parse_args()
    if args.validate:
        args.cycles, args.jac_every, args.epochs = 2, 3, 300

    os.makedirs(args.out, exist_ok=True)
    norm = (0.20, 10.0, 1000.0, 100.0)  # NORM_POS(=PHYS_SIZE), NORM_VEL, NORM_ACC, NORM_FORCE
    t0 = time.time()
    if args.from_dataset:
        print(f"=== F1 Sobolev: loading archived dataset {args.from_dataset} (no LBM) ===", flush=True)
        d = np.load(args.from_dataset)
        X, Y, JX, JP, JG = d['X'], d['Y'], d['JX'], d['JP'], d['JG']
        N_PTS = JX.shape[1] // 6           # INPUT_DIM = N_PTS*6
        WING_LEN = 0.01                    # teacher.WING_LEN
        span = np.linspace(WING_LEN / 2.0, -WING_LEN / 2.0, N_PTS)
        meta = dict(N_PTS=N_PTS, span=span, norm=norm)
        data = (X, Y, JX, JP, JG, meta)
    else:
        print(f"=== F1 Sobolev: collecting (cycles={args.cycles}, jac_every={args.jac_every}) ===", flush=True)
        data = collect(args.cycles, args.jac_every, args.eps, args.seed, norm)
        X, Y, JX, JP, JG, meta = data
    print(f"collected {X.shape[0]} snapshots, {JX.shape[0]} jacobian samples in {time.time()-t0:.1f}s", flush=True)
    print(f"J_gt magnitude: mean Frob = {np.mean(np.sqrt(np.sum(JG**2,axis=(1,2)))):.4g}", flush=True)

    # held-out split for the Jacobian metric
    nj = JX.shape[0]
    ho = max(1, nj // 5)
    tr = slice(0, nj - ho); te = slice(nj - ho, nj)
    data_tr = (X, Y, JX[tr], JP[tr], JG[tr], meta)

    print("=== training MSE-only ===", flush=True)
    p_mse, model, jpred_fn = train(data_tr, 0.0, args.epochs, args.batch, args.lr, args.seed, "MSE")
    print("=== training Sobolev ===", flush=True)
    p_sob, _, _ = train(data_tr, args.lambda_g, args.epochs, args.batch, args.lr, args.seed, "SOB")

    err_mse = jac_frob_error(p_mse, jpred_fn, JX[te], JP[te], JG[te])
    err_sob = jac_frob_error(p_sob, jpred_fn, JX[te], JP[te], JG[te])
    result = dict(
        n_snapshots=int(X.shape[0]), n_jac=int(nj), n_heldout=int(ho),
        lambda_g=args.lambda_g, eps=args.eps, cycles=args.cycles,
        jac_frob_error_mse=err_mse, jac_frob_error_sobolev=err_sob,
        improvement_pct=100.0 * (err_mse - err_sob) / err_mse if err_mse > 0 else 0.0,
        seed=args.seed,
    )
    print("\n=== MEASURED held-out Jacobian Frobenius error (replaces fabricated 26.18->12.90) ===")
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out, 'sobolev_jacobian_result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, 'fluid_sobolev.pkl'), 'wb') as f:
        pickle.dump({'params': p_sob, 'meta': {'N_PTS': meta['N_PTS'], 'norm': norm}}, f)
    with open(os.path.join(args.out, 'fluid_mse.pkl'), 'wb') as f:
        pickle.dump({'params': p_mse, 'meta': {'N_PTS': meta['N_PTS'], 'norm': norm}}, f)
    if not args.from_dataset:
        np.savez(os.path.join(args.out, 'sobolev_dataset.npz'), X=X, Y=Y, JX=JX, JP=JP, JG=JG)

    # Figure 2 — generated entirely from real held-out data
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'figures_and_scripts', 'figure2_sobolev_gradients')
    manuscript_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                  'manuscript-flight-optimisation', 'figures',
                                  'flight_control_sobolev_gradients.png')
    out_pngs = [os.path.join(args.out, 'flight_control_sobolev_gradients.png')]
    if os.path.isdir(os.path.dirname(manuscript_fig)):
        out_pngs.append(manuscript_fig)
    make_figure2(p_mse, p_sob, jpred_fn, JX[te], JP[te], JG[te], err_mse, err_sob, out_pngs)
    print(f"saved results to {args.out}")


if __name__ == '__main__':
    main()
