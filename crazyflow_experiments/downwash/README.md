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
