# LEGS_FORMAT v1: real stand legs

> **Shared contract.** Three repositories are involved:
>
> | role | repository | what it holds |
> |---|---|---|
> | producer code | `crazyflow_experiments` (`downwash/`) | decoding, sync, conversion, export, validation |
> | producer data | `crazyflow_experiments_data` | raw recordings and the exported dataset under `legs/<version>/` |
> | consumer | `afterwash` | replay, fit |
>
> This file lives in `crazyflow_experiments/docs/LEGS_FORMAT.md`. Identical copies go in `crazyflow_experiments_data/docs/` and `afterwash/docs/`. Change all copies together and bump `format_version`.
>
> **afterwash never imports code from either producer repository.** The exported dataset is the only interface.
>
> **Scope of v1:** stand legs in S0 (victim rotors off, props free) and S1 (victim rotors spun at a fixed command for each leg). Free flight (F) is reserved (§7).
>
> **Layout:** it mirrors afterwash's sim run folders: `meta.json` plus `states.npz`, arrays `(T, 2, …)`, index 0 = victim (`lower`), index 1 = source (`upper`).
>
> **Revision of 26 Nov 2026,** after the Step 0 map of the real pipeline. Changes from the first draft:
> - repository roles;
> - mount sign $+1$;
> - static victim pose;
> - 80 Hz load cell with a sample mask;
> - no filter in the stored labels;
> - the stand delay from the existing `cell_delay_s`;
> - a constant baseline from the existing pre-window;
> - NaN in card gaps;
> - unique leg IDs;
> - a thrust curve per drone;
> - a code commit instead of a self-referencing commit.

## 1. Folder layout

```
crazyflow_experiments_data/legs/<version>/      e.g. legs/legs_v1/   (immutable once tagged)
  dataset.json
  manifest.csv
  chain_test.npz                                reference input/output of the stand chain (§6.3)
  <leg_id>/
    meta.json
    states.npz
    quicklook.png                               optional
```

- `leg_id = <block>_<condition>_<run_id>_leg<NN>`. Here `run_id` is the unique run-folder name (with its timestamp), so reflown legs, reruns and flight numbers reused across days all get distinct IDs.
- Never change a tagged version. Any change in processing creates a new version (`legs_v2`).

## 2. `dataset.json`

| key | content |
|---|---|
| `format`, `format_version` | `"LEGS_FORMAT"`, `1` |
| `version` | `"legs_v1"` |
| `created` | ISO time |
| `producer` | `{"code_repo": "crazyflow_experiments", "code_commit": "<hash>", "code_dirty": false}`. The data repository's own commit is **not** stored here (a file can't hold the commit that contains it); the consumer records it when it reads the dataset |
| `synthetic` | `false` for real data. `true` for afterwash's self-test sets, which also carry `truth` |
| `truth` | `null`, or `{"c_w", "b_v", "kappa", "t0", "stand", "noise"}` for synthetic sets |
| `runs` | list of `run_id` with date, design version and raw-data path |
| `calibrations` | `rpm2thrust` per drone (coefficients, source, sha256); stand (`cell_delay_s` source, gain fit) |
| `counts` | legs per `(date, design_version, block, condition, dz_nominal, v_nominal)`, total and usable |

## 3. `manifest.csv`

One row per leg, with these columns:

`leg_id, run_id, date, design_version, block, condition, flight, leg_index, repeat, direction, dz_nominal, v_nominal, y_nominal, dz_meas, v_meas, y_meas, noise_fz_N, hard_fail, flags, rerun_of, path`

- `rerun_of` holds the `leg_id` of the earlier attempt this leg repeats (empty otherwise).
- `flags` is a `;`-separated list (§5.6).

## 4. `states.npz`

All arrays share **one 500 Hz grid**, and **$t = 0$ is the crossing** (§6.1).

**Span:** from `window_pre[0] − 0.1 s` (at most −2.5 s) to $+4.0$ s. In S1 the end is $\min(4.0,\ t_\text{spin\_down} - 0.2)$. Use SI units unless stated otherwise.

| array | shape | content |
|---|---|---|
| `t` | (T,) | s, uniform step 0.002 |
| `pos` | (T,2,3) | m, world. **Victim:** the stand pose, constant (a single mocap mean at run start). **Source:** mocap time series |
| `quat` | (T,2,4) | order **x, y, z, w**. Victim: constant stand attitude (mocap mean if recorded, else $(0,0,0,1)$ with `victim_attitude_assumed`). Source: mocap |
| `vel` | (T,2,3) | m/s, world. Victim: zeros. Source: mocap central difference + 10 Hz zero-phase low-pass |
| `ang_vel` | (T,2,3) | rad/s, body, from `gyro.*`. NaN where not logged or not decoded |
| `rotor_vel` | (T,2,4) | rpm, measured (`motor.m1_rpm` … `m4_rpm`). Victim in S0: 0 (windmill rpm not logged). NaN in gaps > 5 samples |
| `motor_thrust` | (T,2,4) | N per rotor, `rpm2thrust` **of that drone** (§5.4). Victim in S0: 0. Source without a decoded SD card: total thrust from mocap split equally, flag `source_thrust_from_mocap` |
| `thrust_sum` | (T,) | N, victim $\sum_i f_i$ from `motor_thrust`, unfiltered |
| `fz_new` | (T,) bool | true where a **new** load-cell sample arrived. The cell samples at about 80 Hz, and the 500 Hz grid holds each value (zero-order hold) |
| `fz_raw` | (T,) | N, load cell: scale and sign (§6.2) with the `window_pre` mean removed. No shift, no filter. **NaN inside card gaps** |
| `fz_total` | (T,) | N. Here, equal to `fz_raw` (no processing is applied, and the baseline is the same constant) |
| `fz_wake` | (T,) | N. S1: `fz_total` minus the chain-processed thrust change (§6.5). S0: equal to `fz_total` |
| `sim/freq` | () | `500` |

**Label used by the fit:** `fz_total` in S0 and `fz_wake` in S1.

## 5. `meta.json`

### 5.1 `run`
`{"id": leg_id, "run_id", "date", "plugin": "real", "created", "producer_code_commit"}`

### 5.2 `design`
The nominal values from the design file:
`{"design_version", "block", "condition", "dz", "v", "y", "repeat", "direction", "flight", "leg_index"}`

### 5.3 `derived` (measured)

| key | definition |
|---|---|
| `cross_time_abs` | crossing time on the session clock (s); $t = 0$ in `states.npz` |
| `v_meas` | mean horizontal source speed over $t \in [-0.3, 0.3]$ s (mocap) |
| `dz_meas` | mean $z_\text{src} - z_\text{stand}$ over the same interval |
| `y_meas` | mean signed offset of the source from the stand, perpendicular to the path |
| `t_pass_start`, `t_spin_on`, `t_spin_down` | relative to the crossing (spin times: S1 only) |
| `window_pre` | `[t_a, t_b]`: the existing converter's baseline window for this condition (S1: $[t_\text{spin\_on} + 1.5,\ t_\text{pass\_start}]$) |
| `window_tail` | the last 0.5 s of the leg (diagnostic only, never subtracted) |
| `victim_pose_source` | `"stand_mean"` |

### 5.4 `sim_params`
- `drone/rpm2thrust`: shape (2,3), one row per drone (victim, source).
- `drone/rpm2thrust_source`: for example `["A1 fit cf10", "crazyflow cf21B_500 × 0.92 (cf3)"]`.
- `drone/mixing_matrix`.
- `drone/mass`: `[0.05425, m_src]`. The victim value is with the systemID deck (A1); the stand fit doesn't use it.

### 5.5 `rig.stand`: the measurement chain (§6.3)

```json
"rig": {"stand": {
  "mount_sign": 1,
  "sample_rate_hz": 80,
  "dynamics": {"kind": "delay", "tau_eff_s": 0.027, "gain_ratio": 1.00,
               "source": "cell_delay_s (a3_stand_convert, 9 runs, std < 1 ms); gain from export Step 1b"},
  "processing": {"shift_s": 0.0, "filter": "none"}
}}
```

`dynamics` is what the stand does physically. `processing` is what the code does afterwards. Both must be linear and time-invariant. `dynamics.kind` takes one of three forms:

| `kind` | fields | from |
|---|---|---|
| `"second_order"` | `f_n_hz`, `zeta`, `tau_s` | a tap test (not done) |
| `"delay"` | `tau_eff_s`, `gain_ratio`, `source` | the motor edges. $G = e^{-s\tau_\text{eff}}$. **This is the current case** |
| `"unknown"` | none | $G = 1$; every leg is flagged `stand_dynamics_unknown` |

$\tau_\text{eff}$ is measured on the held 500 Hz signal, so it already includes the hold's mean delay (about half a sample, 6 ms).

### 5.6 `quality`

| key | content |
|---|---|
| `noise_fz_N`, `noise_fz_wake_N` | std of the label in `window_pre`, **over the `fz_new` samples only** |
| `n_fz_pre` | number of `fz_new` samples in `window_pre` |
| `peak_fz_N`, `peak_time_s` | minimum of the label **after the existing peak finder's 10 Hz zero-phase low-pass**, over $[-0.5/v_\text{meas},\ 0.5/v_\text{meas} + 1.0]$ s, and its time |
| `tail_level_N` | mean label in `window_tail` (wake tail plus drift; the two can't be separated) |
| `card_gaps` | list of `[t_a, t_b]` where the SD card stalled (load cell NaN) |
| `sync` | residuals of the sync methods used (ms) |
| `rpm_gaps` | per drone and motor: interpolated samples |
| `hard_fail` | bool, with `hard_fail_reason` |
| `flags` | list: `noise_floor`, `card_gap`, `sync_resid_high` (> 5 ms), `rpm_gap`, `speed_err` (> 5 %), `offset_err` (> 3 cm), `overload`, `not_spun`, `superseded`, `aborted_run`, `source_thrust_from_mocap`, `victim_attitude_assumed`, `windmill_unmeasured`, `stand_dynamics_unknown`, `nonlinear_processing` |

**Thresholds:**
- `noise_floor`: $\lvert$peak$\rvert < 5\,\sigma$.
- `superseded`: an earlier attempt of a leg that was flown again on the same date.
- `hard_fail` if any of these holds:
  - a card gap longer than 25 ms overlaps $[-0.5/v_\text{meas} - 0.2,\ 0.5/v_\text{meas} + 1.0]$ s;
  - the sync failed;
  - there is no source mocap;
  - an S1 leg was `not_spun`.

### 5.7 `signals`
`{"quat_order": "xyzw", "rotor_order": ["FR","RR","RL","FL"], "fz_rate_hz": 80, "units": {…}, "arrays": {name: shape}}`

## 6. Conventions that silently break fits

### 6.1 Time
$t = 0$ is when the source passes the stand along the path: the zero crossing of $(\mathbf p_\text{src} - \mathbf p_\text{stand})\cdot\hat{\mathbf d}$, interpolated between samples, from mocap.

### 6.2 Frames, signs, order
- **World frame:** $z$ up.
- **Force sign:** $F_z < 0$ is downward. Wake pulses are **negative**; thrust is positive.
- **Load-cell conversion:** $F_z = s\,\tfrac{g}{1000}\,(w - \bar w_\text{pre})$ with $w$ in grams and $s =$ `mount_sign`. On this stand $s = +1$: the cell was calibrated with the drone on top and negative weights, so the calibrated reading already points along the force. **Check:** a 10 g weight placed on the drone reads $-10.03$ g, i.e. $F_z = -0.098$ N.
- **Quaternion order:** $(x, y, z, w)$.
- **Rotor order:** index $0..3$ ↔ `motor.m1..m4` ↔ front-right, rear-right, rear-left, front-left (crazyflow's mixing matrix). The physical motor numbering on the real drones is still [VERIFY].

### 6.3 Stand chain
The load cell reports

$$
F_\text{rep} = \mathcal P\big[\,G * F\,\big]
$$

where $G$ is given by `dynamics` (§5.5) and $\mathcal P$ by `processing` (currently none). The **chain** is $\mathcal C[x] = \mathcal P[G * x]$. Implement it on the 500 Hz grid like this:
- `second_order`: `scipy.signal.cont2discrete` (bilinear) + `lfilter` from rest, then the delay;
- delays and shifts: linear interpolation, with values before the start held at the first sample;
- a filter, if any: `scipy.signal.butter(order, fc_hz, fs=500)`, applied with `filtfilt` if zero-phase.

With the current values, $\mathcal C[x](t) = x(t - 0.027\ \text{s})$.

`chain_test.npz` holds the producer's output for a unit impulse and for a 0.3 s rectangular pulse at $t = 0$, for every distinct `rig.stand` in the dataset, keyed by its hash. **The consumer's implementation must reproduce them to $10^{-9}$.**

**What the chain affects.** The chain keeps areas unchanged, so the fitted magnitudes don't depend on it. A chain delay error $\Delta d$ shifts every centroid by $\Delta d$, which ends up in the consumer's $t_0$.

In S1 it also leaves a zero-area residual in `fz_wake` (from the rotor correction, §6.5). That residual moves each leg's centroid by a further $\approx \Delta d \cdot A_{\Delta T}/J$, which differs between legs; $A_{\Delta T}$ is the area of the thrust change and $J$ the label area. The current $\tau_\text{eff}$ was measured against the rpm, so for the rotor correction $\Delta d \approx 0$.

### 6.4 Baseline
- **Constant:** the mean of the label over the `fz_new` samples in `window_pre`, which is the existing converter's window.
- **No post-pass window is subtracted.** The wake has a long tail: at $\Delta z = 0.5$ m and $v \ge 1.4$ m/s the force is still about −30 mN with the source 1 m away. A line through a post window would remove real wake.
- Drift and tail are reported in `tail_level_N`.

### 6.5 Rotor correction (S1)
$$\texttt{fz\_wake} = \texttt{fz\_total} - \mathcal C\big[\texttt{thrust\_sum} - \overline{\texttt{thrust\_sum}}_\text{pre}\big]$$

The thrust change must pass through **the same chain** as the load cell. The existing `dforce_wake` subtracts the raw rpm thrust without the 27 ms delay. That leaves a derivative-shaped artifact with zero area that shifts centroids.

### 6.6 Gaps
- Load-cell values inside `card_gaps` are NaN. **Never interpolate across a gap of more than 2 load-cell samples (25 ms).**
- rpm gaps of ≤ 5 samples (10 ms) may be interpolated; longer ones are NaN.
- NaN propagates through the chain. The consumer excludes a leg from any integral whose window contains NaN, and reports it.

## 7. Reserved: free flight (F)

v2 adds `condition = "F"`, the victim labels `force_imu` and `force_fd` (T,2,3) in the body frame, `acc` (T,2,3), and a victim mocap time series. Do not use these names for anything else.

## 8. Validation (both sides implement these checks)

1. Every listed file exists; the shapes match §4; `t` is uniform with step $0.002 \pm 10^{-6}$.
2. NaN appears only in: `fz_*` inside `card_gaps`; `rotor_vel`, `motor_thrust` and `thrust_sum` inside rpm gaps; `ang_vel`.
3. $t = 0$ lies within one sample of the sign change of $(\mathbf p_\text{src}-\mathbf p_\text{stand})\cdot\hat{\mathbf d}$.
4. The victim's `pos` and `quat` are constant over the leg; $\lvert\mathbf q\rvert = 1 \pm 10^{-6}$; $w > 0.99$.
5. `motor_thrust` equals each drone's own `rpm2thrust(rotor_vel)` to $10^{-9}$ N (except legs flagged `source_thrust_from_mocap`). The victim's `motor_thrust` is 0 in S0.
6. The label's mean over the `fz_new` samples in `window_pre` is 0 to $10^{-12}$ (by construction). Separately, `noise_fz_N` is computed from the `fz_new` samples only: $N$ = `n_fz_pre`, not the 500 Hz count.
7. Every S0 leg without `noise_floor` and `hard_fail` has `peak_fz_N` < 0.
8. The consumer reproduces `chain_test.npz` to $10^{-9}$.
9. The manifest and the leg folders match one to one; `counts` matches the manifest; `leg_id`s are unique.
