# Pipeline map: A3 stand legs → LEGS_FORMAT v1 (Step 0)

Written 2026-11-26 from the code as it is, no changes made. **The code is not in this repository.**
It lives in the sibling repo `../crazyflow_experiments` (package `crazyflow_experiments/downwash/`,
tests `tests/downwash/`, env `pixi run -e sim2real`, checked-out branch `log-rpm-via-radio`, the
README's branch `a1_rpm2thrust`). This repo holds the data the code writes through the
`results/` and `plots/` symlinks, plus `notebooks/` and `tools/a3_to_states.py`.
Code paths below are relative to `../crazyflow_experiments/crazyflow_experiments/downwash/`.

## 1. Items

### Raw data
| what | where |
|---|---|
| run folder | `results/downwash/a3/stand_flight<k>_<still\|spin>_<YYYY-MM-DD_HH-MM-SS>/` (33 runs, 21–22 Nov) |
| run metadata | `<run>/meta.json`: config, `design_rows`, `stand_pos`, `load_cell` {a, b, force_sign, sample_rate}, `victim_schedule` (per-leg schedule + PC-time events), `usd` (card variables, files) |
| flight records (PC side) | `<run>/records.pkl` → `["flight"]`: ~100 Hz list of {`t_wall`, `t_perf`, `source`: {pos, quat, vel, ang_vel, pos_raw, quat_raw, cmd_*, phase, leg}, `victim`: **None**} |
| radio log | `<run>/radio.npz`: `victim/loadcell/{t_pc, t_tick_ms, loadcell.rawWeight, loadcell.weight}`, `victim/status/…` (100 Hz) |
| SD cards | `<run>/sd/victim/vic_<ts>_SD`, `<run>/sd/source/src_<ts>_SD`; backups in `raw_cards/a3_stand/` |
| card configs | victim `sd/config_lc_victim.txt` (500 Hz: stateEstimate x,y,z,vx,vy,vz, `stateEstimateZ.quat`, gyro.xyz, motor.m1–m4_rpm, acc.xyz, ctrltarget.z, pm.vbatMV, loadcell.rawWeight); source `sd/config_a3_source.txt` (same minus load cell, plus ctrltarget.x) |
| mocap bag | only `results/downwash/a3/bags/stand_flight1_still/` (one run); no bag for the others |
| copies | `results/downwash/a3_evening22/` = 19 of the 22-Nov-evening runs (identical leg files); `a3_evening22_states/` = viewer files made from them |

### Decoding, sync
| stream | function | method | residuals stored |
|---|---|---|---|
| victim card | `usd.decode(usd.sd_file(folder, "victim", meta))` | — | — |
| victim card → PC | `sync.sync_drone(own, "victim", card, 1/500, LOAD_CELL_SYNC)` (`sync.py:248`), called in `a3_stand_convert.load_run` (`:205`) | radio tick → PC through the arrival-time floor (`fit_radio_clock`); card timer → tick by matching `loadcell.rawWeight` logged on both (`match_offset`, rate 1 or 85/84) | `<run>/sync.json`: `latency_median`, `latency_p95`, `drift_percent`, `sd_rate`, `sd_offset_ms`, `sd_offset_halves_ms`, `match_fraction`, `sd_gaps_over_10ms`, `sd_longest_gap_ms` |
| source pose | `records.pkl` `source.pos` (ROS connector) / `pos_raw` (`/tf`), stamped `t_wall` on arrival | already PC time | — |
| source card | **not decoded or synced by the stand pipeline** (NOTES G4). The free-flight converter can: `a3_convert.read_onboard`, `setpoint_echo` + `sync.fit_clock` (ctrltarget echo) | — | — |
| rpm (option) | `--rpm-source radio`: 100 Hz rpm from the radio load-cell block (runs after `d3e8cde` only) | | `sync.json` `rpm_source` |

### Leg cutting and time zero (`a3_stand_convert.cut_leg`, `:257`)
- Legs come from `meta.victim_schedule`: per leg `tare_start → arm_at (+2 s) → spin_on (+1.5) → baseline_start (+1.5) → pass_start (+2) → cross → pass_end → spin_off (+1.5 after pass_end) → leg_end`. Still runs run the same schedule without spinning.
- **t = 0 is from mocap**: `measured_crossing` (`:141`): zero crossing of `source.pos[:,0] − stand_pos[0]` (records, 100 Hz), linearly interpolated, within [pass_start, pass_end]; with several crossings the one at the highest speed. Projection on +x only (all passes are −x→+x, `direction` = +1, NOTES D2). `stand_pos` = mean mocap position of the victim over 1 s at run start (`lc_test_deploy.still_position`), constant per run. No crossing → `cross_nominal`, status `no_crossing`.
- Grid: `t_grid = arange(baseline_start, stop, 1/500)`, `t = t_grid − cross` → **t = 0 is not a grid sample**. `stop = min(spin_off event, schedule spin_off)`. Span e.g. −5.4…+4.2 s (0.5 m/s), −4.0…+3.3 s (2 m/s).
- Resampling: `a3_convert.resample` (`:204`), linear, NaN across gaps > 10 ms (card), > 50 ms (mocap); no gap filling.

### `force_z` (the "force_total" of the viewer files)
`load_run` `:244`, `cut_leg` `:288-289, :297`:
- `weight_g = a · rawWeight + b` (a, b from `meta.load_cell` or `--calibration`; runs ≤ flight 4 old cell, run 7: a = 7.2113e-5; flight 6 and the two late flight-4 reflies new cell, run 9: a = 7.6393e-5).
- **sign +1**: `force_sign = 1` in meta; the calibration was taken drone-on-top with negative weights, so `weight_g` already reads upward force (A1 NOTES dev. 2, A3 NOTES D5). Probe: a 10 g weight reads −10.03 g → −0.098 N.
- `force_z = (weight_g − tare_g) · 1e-3 · 9.81`, `tare_g` = mean over the per-leg **motors-off tare window** [tare_start, tare_end] (2 s, ~3.5 s before spin_on). So in S1 `force_z` contains the rotor thrust (~0.5 N).
- **No shift, no filter.** Load cell at 80 SPS (`sampleRate 3`), the card samples the held value at 500 Hz (zero-order hold).
- `dforce_z = force_z − mean(force over baseline)`, baseline = [spin_on + 1.5 s, pass_start] (2 s, motors spinning in S1, source ≥ 1.9 m away).

### `thrust_rpm`, `dthrust_rpm`
`load_run` `:226-248`, `cut_leg` `:298, :304, :307`:
- rpm: card `motor.m1..m4_rpm` (500 Hz), `65535` (`usd.RPM_INVALID`) → NaN, no interpolation.
- `rpm2thrust`: `a3_stand.a1_curve()` (`:501`) reads `results/downwash/a1/rpm2thrust.toml`, fit `zero_offset`: **[0.0, −6.064710107326551e-7, 4.376601568313169e-10]**, f = c0 + c1·rpm + c2·rpm² per rotor (N), `thrust.rotor_thrust`. cf10, A1 sweep 2026-11-21, 8016–19557 rpm, R² 0.99997.
- `thrust_rpm` = Σ of the 4 rotors. `dthrust_rpm` = `thrust_rpm − mean(thrust_rpm over the same baseline)`.

### `dforce_z`, `dforce_wake` (the "force", "force_wake" of the viewer files)
`cut_leg` `:306-308`: `dforce_wake = dforce_z − dthrust_rpm`. **The rpm thrust is subtracted raw: no filter, no delay**, although the cell lags it by 0.027 s. In S0 the rpm read 0 → `dforce_wake = dforce_z`.

### Summary table (the one in CLAUDE_export_legs.md Step 9)
- Code: `notebooks/a3_cell_runs.ipynb`, cell 14 (`peak_rows`). Data: `results/downwash/a3_evening22_states/dz*/v*/*/states.npz`, made by `tools/a3_to_states.py` from the leg files of `results/downwash/a3_evening22/` (22-Nov-evening runs only).
- Peak: signal low-passed by 4th-order Butterworth 10 Hz, zero phase (`sosfiltfilt`, NaN bridged by index interpolation); **minimum** in t ∈ [−0.25, 1.0] s. Delay = time of that minimum after the crossing. Mean ± sample std (ddof 1) over runs.
- "spin" column = spin `force_wake` (= `dforce_wake`); "still" column = still `force_wake` (= `dforce_z`).
- Equals `summary.csv` `peak_dforce_wake_N`, `t_peak_wake_s` to 1e-12 (cell 15). `summary.csv`'s own peak (`a3_stand_convert.peak`, `:117`) is the signed max |·| of the 10 Hz signal over [pass_start, stop] — same result on these legs.
- The Step 9 numbers match cell 14's output to rounding.

### Calibrations
| | path | values |
|---|---|---|
| rpm2thrust (A1) | `results/downwash/a1/rpm2thrust.toml` (+ `NOTES.md`) | above; crazyflow's own cf21B_500: [0, −3.1334e-7, 4.4074e-10]; `thrust_scale` 0.9503 |
| load cell | `calibration/calibration_run7.toml` (old cell), `calibration_run9.toml` (new cell); the values actually used are copied into each run's `meta.json` `load_cell` | a, b as above |
| stand dynamics | **none** (no tap test, NOTES G1). Measured but not applied: `cell_delay_s` | 0.027 s (0.0266–0.0271 s, std < 1 ms, both cells) |
| source thrust | no A1 curve for cf3; config `thrust_scale` 0.92 × crazyflow's curve | |
| masses | victim `config.drones.victim.mass_kg` = **0.0** (not weighed without deck); A1 toml `mass_kg` 0.05425 (cf10 on the stand); source cf3 0.04525 | |

### S1 motor edges
- One spin-up and one spin-down **per spun S1 leg** (the victim spins per leg on its battery, NOTES D4), not per flight. PC times in `meta.victim_schedule.legs[<leg>].events`: `tare_start`, `tare_end`, `spin_on`, `spin_off`, `motors_stopped`; nominal times in `…schedule`.
- The victim card records the whole run (`meta.usd.victim.t_wall_start` precedes the legs), so both edges are on the card. Legs are cut before the spin-down.
- **No 0.3 s sync thrust pulses exist** (NOTES G9).
- Existing delay fit: `a3_stand_convert.spin_on_delay` (`:172`) → `a3_convert.lag` (`:243`): mean-free RMS difference over shifts ±0.3 s, parabola refine, window [spin_on − 0.3, spin_on + 1.0] s, edge must rise ≥ 0.2 N. No gain K is fitted. Result per leg in `summary.csv` `cell_delay_s`, per run in `sync.json` `cell_delay`.

### Conventions
| | existing |
|---|---|
| quaternion | `records.pkl` `quat` (estimator) and `quat_raw` (`/tf`) are **x, y, z, w** (scipy `from_quat`, `deploy/crazyflie.py:110`); bag reader also xyzw (`a3_convert.read_mocap`); card `stateEstimateZ.quat` decompressed to xyzw |
| motor order | `motor.m1..m4` = FR, RR, RL, FL (Crazyflie numbering); crazyflow `mixing_matrix` rows [−1,−1,1,1], [−1,1,1,−1] match that geometry. Real numbering vs matrix still **[VERIFY]** (A2 NOTES) |
| mount sign | **+1** (see `force_z`) |
| forces | N, up positive, wake negative |
| g | 9.81 |
| gyro | card `gyro.*` in deg/s |
| windmill (S0) | not measured: `motor.mX_rpm` = 0, all valid, in every still run (NOTES G7) |

## 2. Contract field → existing source → gap

| contract field | existing source | gap |
|---|---|---|
| `t` (500 Hz, t = 0 a sample, [−2.5, 3.0]) | `cut_leg` grid from baseline_start, offset by cross | regrid at `cross + k·0.002` |
| `pos[:,1]` source | `records.pkl` `source.pos_raw` (100 Hz, PC arrival time) | world frame not in leg files (only `src_pos_rel`); resample |
| `pos[:,0]` victim | `meta.stand_pos` (constant, mocap at run start) | **no time series** (records `victim` = None) |
| `quat[:,1]` source | `records.pkl` `source.quat_raw` | slerp; not in leg files |
| `quat[:,0]` victim | none from mocap; card `stateEstimateZ.quat` (onboard estimate) | **contract conflict** |
| `vel` | source: records `vel` (estimator); victim: 0 | contract wants CD + 10 Hz zero-phase of mocap (`a3_convert` has the filter, `VEL_CUTOFF`) |
| `ang_vel` | victim card `gyro.*` (deg/s); source card `gyro.*` | not in leg files; source card unsynced; deg → rad |
| `rotor_vel` | victim card rpm; source card rpm | source card unsynced; ≤ 5-sample gap fill missing (`a3_convert.fill_dropouts` exists) |
| `motor_thrust` | victim: A1 curve; source: none | per-drone curve needed; S0 victim = 0 |
| `thrust_sum` | `thrust_rpm` | resample only |
| `fz_raw` | `force_z` (tare = motors-off window) | define "pre-mean" = existing tare |
| `fz_total` | `dforce_z` (constant baseline) | linear baseline (§6.4) new |
| `fz_wake` | `dforce_wake` (raw thrust, no chain) | chain-processed thrust, new (deliberate change) |
| `derived.cross_time_abs` | `cross` (PC wall time) | — |
| `derived.v_meas` | `cross_speed` (estimator speed at crossing) | mean over ±0.3 s from mocap |
| `derived.dz_meas`, `y_meas` | `height_sep_meas`, `lateral_offset_meas` over ±0.6/v s | window ±0.3 s |
| `design` | `meta.design_rows` (`repeat` 0-based, `direction` always +1) | — |
| `sim_params` | A1 curve, crazyflow mixing matrix, masses | victim mass unknown (0.0) |
| `rig.stand` | `cell_delay_s` 0.027 s, sign +1, processing none | K, `kind`, record |
| `quality.sync` | `sync.json` | map to ms residuals |
| `quality` peaks, noise, flags | `status` ∈ ok / card_gap / not_spun / no_crossing; `baseline_std_N`; `card_gaps_near` (any NaN in [−1, 1.5] s) | all contract flags new |
| `manifest.csv`, `dataset.json`, `chain_test.npz` | `stand_summary.csv` (93 `ok` legs) | new |

Leg inventory: 33 runs, 118 design rows: 93 ok, 12 card_gap, 1 not_spun, 12 not_flown. 25
(mode, flight, leg) triples occur in more than one run.
