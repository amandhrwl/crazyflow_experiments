# Downwash experiments

The blocks of the downwash experiment, all in this package and the `sim2real` environment:

- **A3**, flyover passes of the source over the hovering victim, both in flight.
- **A1**, crazyflow's rpm2thrust checked with the victim's motors running on the load cell.
- The **load cell setup test**, the source hovering beside and over the victim on the load cell.

They share one layer:

| File | Role |
|---|---|
| `usd.py` | Card configs onto a card (`write-config --block <a1, a3, lc> --role <role>`), logging start and stop, card files into their recordings (`collect --block <a1, a3, lc, hover>`), the card bench test (`bench`), the log decoder |
| `sd/config_<block>_<role>.txt` | The card configs of every block, file stem `vic` on the victim and `src` on the source |
| `sync.py` | Onboard clocks onto PC time: through the radio (A1, load cell test) or the setpoint echo (A3) |
| `radio.py` | Radio logs stamped with their PC arrival time, the victim's link on the load cell |
| `thrust.py` | Rotor thrust from the measured rpm: crazyflow's rpm2thrust times the drone's `thrust_scale` |

Every card file is named after its recording: the scripts set `usd.fileDate` and `usd.fileTime`
to the timestamp of the recording folder before they start the card, and the card writes
`<stem>_<YYYY-MM-DD>_<hh-mm-ss>_SD`, e.g. `vic_2026-11-21_16-31-44_SD` for
`results/load_cell/test_2026-11-21_16-31-44`. The console line naming it goes into the
recording's `meta.json`, and after the run

```bash
pixi run -e sim2real python crazyflow_experiments/downwash/usd.py collect /media/<card> --block <a1|a3|lc|hover>
```

copies every file of the card into `<recording>/sd/<role>/` of the recording that names it. This
needs the patched `usddeck.c` of `../crazyflie-firmware`; older firmware numbers the files
(`vicNN`, `srcNN`) with a warning, and the conversions still find those by the console's name.
The hover experiment (`hover/deploy.py`) names and collects its card the same way.

## Block A3: flyover passes over a hovering victim

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
| `a3_design.py` | Leg table `results/downwash/a3/design_a3.csv` and its plot |
| `timeline.py` | Setpoints of both drones over a flight: holds, moves, hops and passes in closed form |
| `pair.py` | Both drones from one loop: joint takeoff, legs with the safety checks, joint landing |
| `a3_deploy.py` | Flies one flight and records it |
| `a3_convert.py` | Turns a flight into per-leg arrays in the thesis format |
| `sd/config_a3_victim.txt`, `sd/config_a3_source.txt` | The two card configs, 20 variables each at 500 Hz |

### Before the first flight

1. Fill in `config.toml`: the drone ids, the weighed masses (deck, card, markers and battery
   included), each drone's `thrust_scale` and the real flight volume in `[safety]`. The
   `thrust_scale` calibrates crazyflow's rpm2thrust for that drone: hover it, and divide
   `mass_kg * g` by the mean thrust its logged rpm give on the uncalibrated curve. `a3_design.py` derives `v_max` from the arena
   and refuses one below 1.2 m/s.
2. Write the card configs and put each card in its drone before power-up, the firmware reads
   `config.txt` only at boot:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --block a3 --role victim
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --block a3 --role source
   ```

3. Write and check the design, then plan every flight without connecting anything:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py --config a3
   pixi run -e sim2real python crazyflow_experiments/downwash/a3_design.py plot
   pixi run -e sim2real python crazyflow_experiments/downwash/a3_deploy.py --flight 1 --dry-run
   ```

   `--single V DZ` writes a one-leg design for a quick test instead, e.g. `--single 1.0 0.5`, flown
   with `--design results/downwash/a3/design_single_v1.00_dz0.50.csv`.

### Flying

Terminals, all in `pixi run -e sim2real`:

```bash
ros2 launch motion_capture_tracking launch.py
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<victim id> --legacy
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<source id> --legacy
ros2 bag record /tf -o results/downwash/a3/bags/flight<k>
python crazyflow_experiments/downwash/a3_deploy.py --flight <k>
```

Fly in this order, and look at each flight before the next:

| Flight | Command | Look at |
|---|---|---|
| F0 | `--flight 1 --only-source`, victim on the ground and unconnected | cruise speed within 2 % in every settled window, off-path error below 5 cm |
| F1 | `--flight 1`, shakedown order from Δz 1.2 to 0.4 m | convert it first: sync, label bias, one Fz peak per leg at the predicted lag |
| F2, F3 | `--flight 2`, `--flight 3` | |
| Make-up | `a3_design.py --makeup <legs>` writes `design_makeup_f4_legs….csv`, flown with `--flight 4 --design <that csv>` | the legs of F1–F3 whose `summary.csv` status is not `ok`: `hard_fail` (e.g. `sd_gap_*`, a card stall inside the label window; stalls elsewhere only flag `sd_gap_outside_label_*`), `skipped` or `not_flown`; they keep their leg numbers |

Before each flight: fresh batteries in both drones (at least 3.8 V at rest), doors closed and
ventilation off. The cards name their files after the flight; on older firmware they number them,
so note the next file numbers. The drones may stand
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

### After a flight

Copy the SD files into the flight folder, the console lines in `meta.json` name them, with
`usd.py collect /media/<card> --block a3` for each card:

```
results/downwash/a3/flight<k>_<timestamp>/sd/victim/vic_<timestamp>_SD   (vicNN before)
results/downwash/a3/flight<k>_<timestamp>/sd/source/src_<timestamp>_SD   (srcNN before)
```

and convert:

```bash
pixi run -e sim2real python crazyflow_experiments/downwash/a3_convert.py results/downwash/a3/flight<k>_<timestamp> \
    --bag results/downwash/a3/bags/flight<k>
```

The conversion maps each drone's clock onto PC time from the setpoint echo, checks it against the
IMU over the sync hops and stops if the two disagree by more than 5 ms (`--sync-lag-tol`), then
writes per leg `leg<NN>/states.npz`, `meta.json` and `quicklook.png`, and per flight `sync.json`,
`summary.csv`, `summary.png` and `checks.json`. Write the numbers into `results/downwash/a3/NOTES.md`.

### Outputs

| Path | Content |
|---|---|
| `results/downwash/a3/design_a3.csv` | The 27 legs, times from each flight's timeline zero |
| `results/downwash/a3/NOTES.md` | Bench, flight and conversion results |
| `results/downwash/a3/flight<k>_<timestamp>/` | `meta.json`, `records.pkl`, `sd/`, the conversion outputs |
| `plots/downwash/` | Design and dry-run timelines |

## Block A1: crazyflow's rpm2thrust on the load cell

The victim (cf10) sits upright on the load cell, which its own firmware reads (the sysid build),
and its motors run through fixed `motorPowerSet` commands while its card logs the load cell and
the four rpm at 500 Hz. The fit of the steady windows says whether crazyflow's cf21B_500 curve
holds, or which curve or `thrust_scale` replaces it. Spec: `CLAUDE_A1_rpm2thrust.md`; where it
and the load cell tests disagree, `results/downwash/a1/NOTES.md` lists what the code does instead.

| File | Role |
|---|---|
| `a1_config.toml` | Victim, mount, load cell calibration and sign, sweep levels, limits, references |
| `sd/config_a1_victim.txt` | Card config, 15 variables at 500 Hz, 8192 B buffer |
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
python crazyflow_experiments/downwash/usd.py write-config /media/<card> --block a1 --role victim
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

The card reads `config.txt` only at boot, and the scripts reboot the victim. Put the flight
config back on the card afterwards (`usd.py write-config <card> --block a3 --role victim`). Fill in `mass_kg`
and `capacity_g` first, the sweep refuses to run without them. Block `all` takes
about 3 min: a tare, a sync pulse, an ascending ramp through the levels that stops at the first
level predicted over the cap, then two seeded random orders of the allowed levels with a center
level after every six and a tare between them, and tare, pulse, tare. The live line shows the
four rpm, the load cell thrust and the thrust crazyflow's curve gives for those rpm. A block
stops on Ctrl+C, a lost link, the battery under 3.4 V, a motor's rpm invalid for 0.5 s, the thrust
over the cap, or a disarmed, locked or crashed supervisor. Repeat block `all` on another day or
battery and fit both: their scales should agree within their errors.

## Load cell setup test

The victim sits upright on the load cell with its motors off. Its firmware reads the load cell
itself (the `loadcell` deck driver of the sysid build), so the card can log the load cell like any
other variable, and the radio streams it at 100 Hz as well. The source flies one simple pattern,
about 70 s in all:

| Phase | Duration (`[flight]`) | What happens |
|---|---|---|
| `still_before` | 5 s | nothing moves, this sets the zero of the load cell |
| `takeoff` | ~8 s | the source climbs from where it stands to its hold beside the victim and calibrates |
| `settle` | 1 to 15 s | holds there until it stays within `settle_tolerance` (10 cm) of the hold for 1 s |
| `aside_1` | 5 s | holds `aside` (1 m) beside the victim, `height_sep` (0.6 m) above it |
| `move_in` | 4 s | flies level to straight over the victim |
| `over` | 10 s | holds over the victim |
| `move_out`, `aside_2` | 4 s, 5 s | back beside it and holds |
| `landing` | ~7 s | flies back and lands where it took off |
| `still_after` | 5 s | nothing moves |

| File | Role |
|---|---|
| `lc_test_config.toml` | Drone ids, load cell calibration and sample rate, flight timings, safety limits |
| `sd/config_lc_victim.txt`, `sd/config_lc_source.txt` | Card configs, 500 Hz, as flown on 21 Nov: the A3 variables, with `loadcell.rawWeight` in place of the victim's `ctrlMel.cmd_thrust` |
| `lc_test_deploy.py` | Flies the test and records it |
| `lc_test_convert.py` | Puts the cards, the radio logs and the motion capture onto PC time |

### Before the test

1. The victim runs the sysid build of the firmware (`CONFIG_DECK_LOADCELL`, `bcLoadcell` forced),
   which `lc_test_deploy.py` checks in its log TOC, the source a plain `cf21bl_defconfig` build. Firmware
   with the `usd.fileDate` parameter of our `usddeck.c` names each card file after its recording,
   `vic_2026-11-21_16-31-44_SD` for the recording `test_2026-11-21_16-31-44`; older firmware
   numbers it `vicNN` and `lc_test_deploy.py` warns.
2. Write both card configs, with the cards in the PC, and put each card back in its drone before
   power-up:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --block lc --role victim
   pixi run -e sim2real python crazyflow_experiments/downwash/usd.py write-config /media/<card> --block lc --role source
   ```

3. Check `lc_test_config.toml`: the ids, `[load_cell]` `a` and `b` from `calibration.toml` of the cell that
   is mounted (a new cell needs its own run, the first test still had the old cell's), and
   `aside`, so that the source stands on the side of the victim where it holds. The script refuses
   a takeoff path that passes closer than `clearance` (0.5 m) to the victim. Place the source
   close to below its hold: the takeoff ramps there in 4 s however far it is.

### Flying

Terminals, all in `pixi run -e sim2real`. The victim needs no estimator, only its rigid body in the
motion capture:

```bash
ros2 launch motion_capture_tracking launch.py
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<source id> --legacy
python crazyflow_experiments/downwash/lc_test_deploy.py --dry-run   # plans from the motion capture only
python crazyflow_experiments/downwash/lc_test_deploy.py
```

The script reboots and connects the victim, sets the load cell calibration and sample rate
(neither survives a reboot), and starts its card. Then it records the still air and prints the
load cell rate and the victim's battery. Next it connects the source, starts its card and flies.
After the takeoff the source settles at its hold, within the clearance of it, and then flies the
holds. While it flies them, it is stopped when it strays 0.25 m from its setpoint, when it sinks to
`min_height_sep` above the victim, or with Ctrl+C. It then flies back at its height and lands where
it took off rather than dropping onto the load cell. At the end the script prints the mean load cell
reading of every phase and the card files to copy.

#### Spinning victim

`lc_test_deploy.py --spin` runs the victim's motors on the load cell while the source flies, at the command
that makes `[spin] thrust_fraction` (0.5, at most 0.8) of its weight in thrust. With its motors off
the load cell only feels the wake on the still airframe; a flying victim mostly loses rotor thrust
in it, which the spinning victim shows. The script arms the victim after the still air, overrides
its motors with `motorPowerSet` and keeps its supervisor's watchdog fed with zero setpoints, so the
supervisor does not lock it. The override holds even when the firmware stops the motors
(`motorsStop` goes through it in `motors.c`), so neither a disarm nor the watchdog stops them: the
script stops them by disarming, then writing zero commands, then `motorPowerSet.enable = 0`. A lost
link leaves them at their last command, so keep a power cut within reach. New phases: `spin_up`, `spin_alone` (the load cell measures
the thrust at the command, against the thrust from the logged rpm), then the source's flight,
`spin_alone_after` and `spin_down` before `still_after`. Its card logs the four motor rpm and its
tilt for the force balance F = m a - T(rpm).

### After the test

Copy the card files into their recordings and convert. `collect` takes a mounted card, or a folder
the card files were copied to, and puts every file into `<recording>/sd/<role>/` of the recording
whose `meta.json` names it:

```bash
pixi run -e sim2real python crazyflow_experiments/downwash/usd.py collect /media/<card> --block lc
pixi run -e sim2real python crazyflow_experiments/downwash/lc_test_convert.py results/load_cell/test_<timestamp>
```

It writes `synced.npz` (every record on one time axis, s from the start of `still_before`, the
force on the victim in N with up positive) and `sync.json` into the recording and a quicklook to
`plots/load_cell/`. The force uses the calibration the test flew with; a test flown with an
outdated one converts with `--calibration <calib_run.toml>` of `calibscale.py`, and `sync.json`
notes which one it used. What the setup should show:

- `load_cell.zero_g` near 0 g: the calibration offset still holds. `noise_std_g` is the noise floor.
- `over` clearly negative, the source's wake pushing the victim down, `aside_1` and `aside_2`
  near zero.
- `match_fraction` near 0.95 for the victim and 0.5 for the source (its card catches every other
  accelerometer value, and it starts at the takeoff, after the radio), and both halves of each
  drone agree on `sd_offset_ms` within 1 ms.
- `sd_rate` 1 for both. A drone whose firmware predates the fix of the card timer (bitcraze
  crazyflie-firmware#1684, September 2026) gets 85/84 and a warning; the sync corrects it, but
  reflash it when convenient. cf3 had it in the first test.
- `sd_gaps_over_10ms` zero, see the card stalls of the A3 flights otherwise.
- In the bottom panel every radio sample sits on a step of the card.

### Time sync

Each drone has two clocks: the millisecond tick that stamps its radio log packets and the
microsecond timer of its card records. The PC wall clock stamps the arrival of every radio packet
and the motion capture. The line under the arrival times maps the tick onto PC time, including the
drift of the drone's crystal, about 0.1 %. The tick and the card timer count the same crystal, but
the timer starts a little later during the boot, and on older firmware runs 1.19 % slow. A
variable logged both ways measures the offset and picks the rate: the load cell reading on the
victim, the accelerometer on the source. Every record then lands within about 0.5 ms of its PC
time, late by the shortest radio latency (about 1 ms) for both drones alike.
