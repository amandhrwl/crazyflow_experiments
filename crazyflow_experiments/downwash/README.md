# Downwash: flyover passes over a hovering victim (block A3)

One Crazyflie, the *victim*, hovers at a fixed point. A second, the *source*, flies straight passes
over it at 0.4, 0.8 and 1.2 m above, at 0.5, 1.0 and `v_max` m/s. Each of the nine cells is flown
once per flight in three flights, 27 legs in all. Every pass flies from -x to +x; after it the
source flies back to the next start point through a lane `return_lane` (1.0 m) to the side of the
victim, clear of its wake, so it only ever crosses over the victim during a pass. Both drones log
onboard to their micro-SD card decks at about 505 Hz, and the PC records the motion capture and
every setpoint it sends.

| File | Role |
|---|---|
| `config.toml` | Drones, profile, sync hop, takeoff and safety limits; `[FILL]` values come from the lab |
| `design.py` | Leg table `results/downwash/a3/design_a3.csv` and its plot |
| `timeline.py` | Setpoints of both drones over a flight: holds, moves, hops and passes in closed form |
| `pair.py` | Both drones from one loop: joint takeoff, legs with the safety checks, joint landing |
| `usd.py` | SD cards: `config.txt`, logging start and stop, the log decoder |
| `deploy_a3.py` | Flies one flight and records it |
| `convert_a3.py` | Turns a flight into per-leg arrays in the thesis format |
| `sd/config_victim.txt`, `sd/config_source.txt` | The two card configs, 20 variables each at 500 Hz |

## Before the first flight

1. Fill in `config.toml`: the drone ids, the weighed masses (deck, card, markers and battery
   included), each drone's `thrust_scale` and the real flight volume in `[safety]`. The
   `thrust_scale` calibrates crazyflow's rpm2thrust for that drone: hover it, and divide
   `mass_kg * g` by the mean thrust its logged rpm give on the uncalibrated curve. `design.py` derives `v_max` from the arena
   and refuses one below 1.2 m/s.
2. Write the card configs and put each card in its drone before power-up, the firmware reads
   `config.txt` only at boot:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --role victim
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --role source
   ```

3. Write and check the design, then plan every flight without connecting anything:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/downwash/design.py --config a3
   pixi run -e sim2real python crazyflow_experiments/downwash/design.py plot
   pixi run -e sim2real python crazyflow_experiments/downwash/deploy_a3.py --flight 1 --dry-run
   ```

   `--single V DZ` writes a one-leg design for a quick test instead, e.g. `--single 1.0 0.5`, flown
   with `--design results/downwash/a3/design_single_v1.00_dz0.50.csv`.

## Flying

Terminals, all in `pixi run -e sim2real`:

```bash
ros2 launch motion_capture_tracking launch.py
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<victim id> --legacy
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<source id> --legacy
ros2 bag record /tf -o results/downwash/a3/bags/flight<k>
python crazyflow_experiments/downwash/deploy_a3.py --flight <k>
```

Fly in this order, and look at each flight before the next:

| Flight | Command | Look at |
|---|---|---|
| F0 | `--flight 1 --only-source`, victim on the ground and unconnected | cruise speed within 2 % in every settled window, off-path error below 5 cm |
| F1 | `--flight 1`, shakedown order from Δz 1.2 to 0.4 m | convert it first: sync, label bias, one Fz peak per leg at the predicted lag |
| F2, F3 | `--flight 2`, `--flight 3` | |
| Make-up | `design.py --makeup <legs>` writes `design_makeup_f4_legs….csv`, flown with `--flight 4 --design <that csv>` | the legs of F1–F3 whose `summary.csv` status is not `ok`: `hard_fail` (e.g. `sd_gap_*`, a card stall inside the label window; stalls elsewhere only flag `sd_gap_outside_label_*`), `skipped` or `not_flown`; they keep their leg numbers |

Before each flight: fresh batteries in both drones (at least 3.8 V at rest), cards empty apart from
`config.txt` or the next file numbers noted, doors closed and ventilation off. The drones may stand
anywhere inside the arena with `x_margin` to its walls, at least 1 m apart; the script refuses to
take off otherwise.

The script connects the victim, then the source, both on radio 0, checks where they stand, the card
variables against each drone's log TOC and the batteries, and starts SD logging. Both drones
wait disarmed until then and are armed together right before the takeoff. 0.6 s into the climb
both report their state: the firmware silently disarms a drone whose motors do not idle in the
arming rpm range (1000 to 3500 rpm), so a drone that did not arm ends the takeoff and both land,
and the run prints the firmware's console line naming the motor. The takeoff flies both to their
positions: both climb straight up, the victim to its hover height and the source
to 2.2 m, the height of the highest pass, turning to face +x; then both fly level to above their
targets, so the source is at least 1.2 m above the victim even where their paths cross; then the
source sinks to the start point of its first leg. One loop streams the setpoints of both drones at
100 Hz throughout and checks every tick: fresh motion capture, the arena, the vertical separation
while the two are horizontally close, each drone's distance from its commanded path of the last
0.3 s, and the victim's drop. The onboard controller follows the setpoints about 0.17 s late (F0),
so a drone is measured against its path rather than its current setpoint: late but on its path
passes. Any violation, or Ctrl+C, lands both in place; the upper one climbs first if they are close.
Before every pass at Δz = 0.4 m the victim's battery is read under load; below `vbat_min_close_leg` (3.45 V) the leg becomes
a hold of the same length, and the next leg flies back from where the source holds. After each leg
the script prints the cruise speed, how far the source strayed from its path and fell behind its
setpoint, the victim's excursion and its position spread in the pre-pass hover, which flags the leg
`air_not_still` above 1 cm.

## After a flight

Copy the SD files into the flight folder, the console lines in `meta.json` name them:

```
results/downwash/a3/flight<k>_<timestamp>/sd/victim/vicNN
results/downwash/a3/flight<k>_<timestamp>/sd/source/srcNN
```

and convert:

```bash
pixi run -e sim2real python crazyflow_experiments/downwash/convert_a3.py results/downwash/a3/flight<k>_<timestamp> \
    --bag results/downwash/a3/bags/flight<k>
```

The conversion maps each drone's clock onto PC time from the setpoint echo, checks it against the
IMU over the sync hops and stops if the two disagree by more than 5 ms (`--sync-lag-tol`), then
writes per leg `leg<NN>/states.npz`, `meta.json` and `quicklook.png`, and per flight `sync.json`,
`summary.csv`, `summary.png` and `checks.json`. Write the numbers into `results/downwash/a3/NOTES.md`.

## Outputs

| Path | Content |
|---|---|
| `results/downwash/a3/design_a3.csv` | The 27 legs, times from each flight's timeline zero |
| `results/downwash/a3/NOTES.md` | Bench, flight and conversion results |
| `results/downwash/a3/flight<k>_<timestamp>/` | `meta.json`, `records.pkl`, `sd/`, the conversion outputs |
| `plots/downwash/` | Design and dry-run timelines |

# Block A1: crazyflow's rpm2thrust on the load cell

The victim (cf10) sits upright on the load cell, which its own firmware reads (the sysid build),
and its motors run through fixed `motorPowerSet` commands while its card logs the load cell and
the four rpm at 500 Hz. The fit of the steady windows says whether crazyflow's cf21B_500 curve
holds, or which curve or `thrust_scale` replaces it. Spec: `CLAUDE_A1_rpm2thrust.md`; where it
and the load cell tests disagree, `results/downwash/a1/NOTES.md` lists what the code does instead.

| File | Role |
|---|---|
| `a1_config.toml` | Victim, mount, load cell calibration and sign, sweep levels, limits, references |
| `sd/config_a1.txt` | Card config, 15 variables at 500 Hz, 8192 B buffer |
| `a1_stand.py` | Config, crazyflow's curve, thrust cap, thrust predictor, the plan of each block |
| `a1_probe.py` | Bench checks B1-B4: case, calibration weights, sample rate, card |
| `a1_sweep.py` | Runs block `all`, `single` or `bench` (B5) and records it |
| `a1_convert.py` | Card onto PC time, thrust, tares, steady windows, sync pulses, timeline |
| `a1_fit.py` | Curve fits, rho, the decision, `rpm2thrust.toml` and the plots |

**Safety.** `motorPowerSet` overrides the motors even when the firmware stops them, so neither a
disarm nor the supervisor's watchdogs stop them; only the script's zero commands do. A lost link
or a killed script leaves the motors running: power the victim from a bench supply with its
switch in reach, props spinning with nobody near the stand and safety glasses on. The victim is
clamped on the stand, so the thrust may reach twice its weight, bounded by 0.8 of the cell's
rating (`capacity_g`, which has to be filled in); with `clamped = false` it would stay below 0.8
of the weight so it never lifts off.

Steps, all in `pixi run -e sim2real`, the props off until B5 has passed:

```bash
cp crazyflow_experiments/downwash/sd/config_a1.txt /media/<card>/config.txt   # card in the PC
python crazyflow_experiments/downwash/a1_probe.py                      # B1: case S expected
python crazyflow_experiments/downwash/a1_probe.py --calibration-check  # B2: 10, 20, 50 g
python crazyflow_experiments/downwash/a1_probe.py --rate-check         # B3: 80 and 320 SPS
python crazyflow_experiments/downwash/a1_probe.py --sd-check 60        # B4, then --decode <file>
python crazyflow_experiments/downwash/a1_sweep.py --block bench        # B5, props off
python crazyflow_experiments/downwash/a1_sweep.py --block all --dry-run
python crazyflow_experiments/downwash/a1_sweep.py --block all          # props on
python crazyflow_experiments/downwash/a1_convert.py results/downwash/a1/all_<timestamp> --collect /media/<card>
python crazyflow_experiments/downwash/a1_fit.py results/downwash/a1/all_<timestamp>
python crazyflow_experiments/downwash/a1_sweep.py --block single --from results/downwash/a1/all_<timestamp>
```

The card reads `config.txt` only at boot, and the scripts reboot the victim. Keep the flight
config (`sd/config_victim.txt`) aside and put it back on the card afterwards. Fill in `mass_kg`
and `capacity_g` first, the sweep refuses to run without them. Block `all` takes
about 3 min: a tare, a sync pulse, an ascending ramp through the levels that stops at the first
level predicted over the cap, then two seeded random orders of the allowed levels with a center
level after every six and a tare between them, and tare, pulse, tare. The live line shows the
four rpm, the load cell thrust and the thrust crazyflow's curve gives for those rpm. A block
stops on Ctrl+C, a lost link, the battery under 3.4 V, a motor's rpm invalid for 0.5 s, the thrust
over the cap, or a disarmed, locked or crashed supervisor. Repeat block `all` on another day or
battery and fit both: their scales should agree within their errors.
