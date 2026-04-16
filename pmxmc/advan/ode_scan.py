import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve
from pytensor import wrap_jax
from pmxmc.utils import rate_at


@wrap_jax
def ode_advan(meas_time, infu_time, infu_rate, pk_ode, params, y0):
    _meas   = np.asarray(meas_time)
    _itimes = np.asarray(infu_time)
    _irates = np.asarray(infu_rate)

    t_start = min(float(_meas[0]), float(_itimes[0]))
    t_end   = float(_meas[-1])

    # Split only at infusion discontinuities (~4 segments vs ~24).
    _relevant = _itimes[(_itimes >= t_start) & (_itimes <= t_end)]
    all_times = np.unique(np.concatenate([_relevant, [t_end]]))
    n_segs    = len(all_times) - 1

    seg_t0s = all_times[:-1]
    seg_t1s = all_times[1:]

    _itime_set = set(float(t) for t in _itimes)
    _rates     = np.array([rate_at(t, _itimes, _irates) for t in seg_t0s])
    _is_jump   = np.array(
        [(i > 0) and (float(seg_t0s[i]) in _itime_set) for i in range(n_segs)]
    )

    # Measurements per segment (strictly inside (t0, t1]).
    seg_meas = [_meas[(_meas > float(seg_t0s[i])) & (_meas <= float(seg_t1s[i]))]
                for i in range(n_segs)]

    # max_meas: fixed output width for scan.
    # Measurements at exactly t1 are excluded from save_ts and captured via
    # SaveAt(t1=True), so we track them separately.
    seg_meas_before_t1 = [m[m < float(seg_t1s[i])] for i, m in enumerate(seg_meas)]
    seg_has_t1_meas    = np.array(
        [len(m) > 0 and float(m[-1]) == float(seg_t1s[i])
         for i, m in enumerate(seg_meas)]
    )
    max_meas = max(len(m) for m in seg_meas_before_t1)

    # Build padded_save_ts: shape (n_segs, max_meas).
    # Rows contain real measurements (strictly < t1) padded with dummy times
    # to reach max_meas.  Dummy times are spaced in (last_real, t1) so that
    # save_ts remains strictly increasing and never touches t1 itself
    # (t1 is handled separately via SaveAt(t1=True)).
    padded_save_ts = np.zeros((n_segs, max_meas))
    for i, (t0, t1, m_before) in enumerate(
        zip(seg_t0s, seg_t1s, seg_meas_before_t1)
    ):
        n = len(m_before)
        padded_save_ts[i, :n] = m_before
        pad_count = max_meas - n
        if pad_count > 0:
            t_last = float(m_before[-1]) if n > 0 else float(t0)
            # pad_count values strictly in (t_last, t1)
            padded_save_ts[i, n:] = np.linspace(t_last, float(t1), pad_count + 2)[1:-1]

    # Static index array: maps positions in the flattened (n_segs, max_meas+1)
    # scan output back to the ordered measurement values.
    # Each scan step outputs sol.ys[:, 0] of shape (max_meas + 1,):
    #   positions 0..max_meas-1  → values at padded_save_ts[i]
    #   position  max_meas       → value at t1 (from SaveAt t1=True)
    valid_idx = []
    for i in range(n_segs):
        offset = i * (max_meas + 1)
        n = len(seg_meas_before_t1[i])
        for j in range(n):
            valid_idx.append(offset + j)
        if seg_has_t1_meas[i]:
            valid_idx.append(offset + max_meas)
    valid_idx = np.array(valid_idx)

    # ── scan ────────────────────────────────────────────────────────────────
    def step_fn(state, inputs):
        t0, t1, rate, jump, save_ts_i = inputs
        sol = diffeqsolve(
            terms=ODETerm(pk_ode),
            # scan_kind='lax' compiles RK stages via lax.scan instead of unrolled loop.
            # Benchmarked: no benefit for Tsit5 (5 stages). Revisit for Dopri8 (13 stages).
            solver=diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            y0=state,
            dt0=None,
            stepsize_controller=PIDController(rtol=1e-6, atol=1e-8),
            max_steps=1_000_000,
            # save_ts_i contains values strictly < t1; t1 itself via t1=True.
            # sol.ys shape: (max_meas + 1, 3)
            saveat=SaveAt(ts=save_ts_i, t1=True),
            args={**params, "rate": rate},
            made_jump=jump,
        )
        return sol.ys[-1], sol.ys[:, 0]   # carry=state@t1, out=(max_meas+1,) A1

    _, all_outputs = jax.lax.scan(
        step_fn,
        jnp.asarray(y0, dtype=jnp.float64),
        (
            jnp.array(seg_t0s),
            jnp.array(seg_t1s),
            jnp.array(_rates),
            jnp.array(_is_jump),
            jnp.array(padded_save_ts),
        ),
    )

    # all_outputs: (n_segs, max_meas + 1) → extract real measurements
    return all_outputs.reshape(-1)[valid_idx]
