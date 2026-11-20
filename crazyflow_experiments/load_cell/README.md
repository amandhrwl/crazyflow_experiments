# Load cell setup test

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
| `config.toml` | Drone ids, load cell calibration and sample rate, flight timings, safety limits |
| `sd/config_victim.txt`, `sd/config_source.txt` | Card configs, 500 Hz; the victim's logs `loadcell.rawWeight` |
| `usd.py` | Card configs onto a card, logging start and stop, the log decoder |
| `radio.py` | Radio logs stamped with their PC arrival time, the victim's link |
| `deploy.py` | Flies the test and records it |
| `sync.py`, `convert.py` | Put the cards, the radio logs and the motion capture onto PC time |

## Before the test

1. The victim runs the sysid build of the firmware (`CONFIG_DECK_LOADCELL`, `bcLoadcell` forced),
   which `deploy.py` checks in its log TOC.
2. Write both card configs, with the cards in the PC, and put each card back in its drone before
   power-up:

   ```bash
   pixi run -e sim2real python crazyflow_experiments/load_cell/usd.py write-config /media/<card> --role victim
   pixi run -e sim2real python crazyflow_experiments/load_cell/usd.py write-config /media/<card> --role source
   ```

3. Check `config.toml`: the ids, `[load_cell]` `a` and `b` from `calibration.toml` of the cell that
   is mounted (a new cell needs its own run, the first test still had the old cell's), and
   `aside`, so that the source stands on the side of the victim where it holds. The script refuses
   a takeoff path that passes closer than `clearance` (0.5 m) to the victim. Place the source
   close to below its hold: the takeoff ramps there in 4 s however far it is.

## Flying

Terminals, all in `pixi run -e sim2real`. The victim needs no estimator, only its rigid body in the
motion capture:

```bash
ros2 launch motion_capture_tracking launch.py
python -m drone_estimators.ros_nodes.ros2_node --drone_name cf<source id> --legacy
python crazyflow_experiments/load_cell/deploy.py --dry-run   # plans from the motion capture only
python crazyflow_experiments/load_cell/deploy.py
```

The script reboots and connects the victim, sets the load cell calibration and sample rate
(neither survives a reboot), and starts its card. Then it records the still air and prints the
load cell rate and the victim's battery. Next it connects the source, starts its card and flies.
After the takeoff the source settles at its hold, within the clearance of it, and then flies the
holds. While it flies them, it is stopped when it strays 0.25 m from its setpoint, when it sinks to
`min_height_sep` above the victim, or with Ctrl+C. It then flies back at its height and lands where
it took off rather than dropping onto the load cell. At the end the script prints the mean load cell
reading of every phase and the card files to copy.

## After the test

Copy the card files into the recording, as the script names them, and convert:

```bash
results/load_cell/test_<timestamp>/sd/victim/lcvNN
results/load_cell/test_<timestamp>/sd/source/lcsNN

pixi run -e sim2real python crazyflow_experiments/load_cell/convert.py results/load_cell/test_<timestamp>
```

It writes `synced.npz` (every record on one time axis, s from the start of `still_before`, the
force on the victim in N with up positive) and `sync.json` into the recording and a quicklook to
`plots/load_cell/`. What the setup should show:

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

## Time sync

Each drone has two clocks: the millisecond tick that stamps its radio log packets and the
microsecond timer of its card records. The PC wall clock stamps the arrival of every radio packet
and the motion capture. The line under the arrival times maps the tick onto PC time, including the
drift of the drone's crystal, about 0.1 %. The tick and the card timer count the same crystal, but
the timer starts a little later during the boot, and on older firmware runs 1.19 % slow. A
variable logged both ways measures the offset and picks the rate: the load cell reading on the
victim, the accelerometer on the source. Every record then lands within about 0.5 ms of its PC
time, late by the shortest radio latency (about 1 ms) for both drones alike.
