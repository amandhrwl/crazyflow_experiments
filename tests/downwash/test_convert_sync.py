"""Clock synchronization and conversion of the downwash flights, on synthetic data.

The end-to-end test flies a single leg with the fake drones, writes both SD logs as the firmware
would, with their own clocks, the radio latency and rpm dropouts, and a rosbag of /tf, then
converts the flight and checks what comes out.
"""

from __future__ import annotations

from tests.conftest import module_requires_env

module_requires_env("sim2real")

import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import threading  # noqa: E402
import zlib  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from rosbags.rosbag2 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

from crazyflow_experiments.downwash import a3_convert, pair, usd  # noqa: E402
from crazyflow_experiments.downwash.a3_convert import (  # noqa: E402
    body_forces,
    derive,
    fill_dropouts,
    lag,
    resample,
)
from crazyflow_experiments.downwash.a3_deploy import write_outputs  # noqa: E402
from crazyflow_experiments.downwash.a3_design import (  # noqa: E402
    build_design,
    load_config,
    model_params,
)
from crazyflow_experiments.downwash.sync import ClockFit, change_events, fit_clock  # noqa: E402
from crazyflow_experiments.downwash.timeline import Pass, victim_hover_point  # noqa: E402
from tests.downwash.fakes import FakeClock, FakeDrone, fake_read_once, patch_pair  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

LOG_RATE = 505.5  # Hz, the stabilizer runs on the IMU clock
G = 9.81
LATENCY = (0.0005, 0.001)  # s, radio latency of the synthetic setpoints
# The echo fit maps onboard time to the send time: early by the mean latency and half a log period
MAPPING_LEAD = sum(LATENCY) / 2 + 0.5 / LOG_RATE


def onboard_echo(
    t_send: np.ndarray, mm: np.ndarray, t_samples: np.ndarray, latency: np.ndarray
) -> np.ndarray:
    """The setpoint a drone holds at each sample time: the last one received, 0 before any."""
    order = np.argsort(t_send + latency)
    arrival, values = (t_send + latency)[order], mm[order]
    k = np.searchsorted(arrival, t_samples, side="right") - 1
    return np.where((k >= 0)[:, None], values[np.maximum(k, 0)], 0)


def test_fit_clock_recovers_offset_and_drift():
    rng = np.random.default_rng(0)
    t_send = 1.7e9 + 50.0 + np.arange(0.0, 200.0, 0.01)  # 100 Hz for 200 s
    t = t_send - t_send[0]
    z = np.clip(t / 4.0, 0.0, 1.0) + 0.025 * (1 - np.cos(2 * np.pi * np.clip(t - 10.0, 0.0, 1.0)))
    x = -1.2 + np.clip(t - 20.0, 0.0, 170.0) * 0.01
    mm = np.trunc(np.stack([x, z], axis=1) * 1000).astype(int)
    keep = rng.random(len(t)) > 0.02  # 2 % of the setpoints are lost on the radio
    latency = rng.uniform(0.002, 0.004, keep.sum())
    offset, drift = 1.7e9 + 12.345, 37e-6
    t_onboard = np.arange(
        (t_send[0] - offset) / (1 + drift), (t_send[-1] - offset) / (1 + drift), 1 / LOG_RATE
    )
    echo = onboard_echo(t_send[keep], mm[keep], offset + (1 + drift) * t_onboard, latency)
    fit = fit_clock(t_send, mm, t_onboard, echo, prior_offset=offset + 0.4)
    assert fit.drift == pytest.approx(drift, abs=1e-6)
    # Mapped to the send time: early by the mean latency and half a log period
    expected = offset - 0.003 - 0.5 / LOG_RATE
    assert fit.offset == pytest.approx(expected, abs=1e-3)
    assert fit.residual_std < 1e-3
    n_changes = len(change_events(t_send, mm)[0])  # Only a change of the mm value is echoed
    assert fit.n_matches > 0.95 * n_changes
    mapped = fit.to_pc(t_onboard[[0, -1]])
    np.testing.assert_allclose(mapped, offset + (1 + drift) * t_onboard[[0, -1]] - 0.004, atol=2e-3)


@pytest.mark.parametrize("drift", [13.4e-3, -13.4e-3])
def test_fit_clock_follows_the_cf21b_drift_across_the_flight(drift: float):
    """The cf21B clocks run 1.3 % off the PC (F1, 18 Nov); one offset held for 2 s only.

    Like the victim, the setpoint changes only at the takeoff (the first 20 s) and at the sync hops
    at the end, so the end is matched by extrapolating the fit over the takeoff.
    """
    t_send = 1.7e9 + 50.0 + np.arange(0.0, 290.0, 0.01)
    t = t_send - t_send[0]
    climb = np.clip(t / 15.0, 0.0, 1.0)
    hop = 0.025 * (1 - np.cos(2 * np.pi * np.clip(t - 280.0, 0.0, 1.0)))
    mm = np.trunc((climb + hop)[:, None] * 1000).astype(int)
    offset = 1.7e9 + 12.345
    t_onboard = np.arange(
        (t_send[0] - offset) / (1 + drift), (t_send[-1] - offset) / (1 + drift), 1 / LOG_RATE
    )
    latency = np.full(len(t_send), 0.003)
    echo = onboard_echo(t_send, mm, offset + (1 + drift) * t_onboard, latency)
    fit = fit_clock(t_send, mm, t_onboard, echo, prior_offset=offset + 0.4)
    assert fit.drift == pytest.approx(drift, abs=1e-6)
    assert fit.residual_std < 1e-3
    hops = change_events(t_send[t > 270.0], mm[t > 270.0])[0]
    assert fit.n_matches > 0.95 * len(change_events(t_send, mm)[0]) > len(hops) > 50
    mapped = fit.to_pc(t_onboard[[0, -1]])
    np.testing.assert_allclose(mapped, offset + (1 + drift) * t_onboard[[0, -1]] - 0.004, atol=2e-3)


def test_fit_clock_needs_a_prior_near_the_truth():
    t_send = 1.7e9 + np.arange(0.0, 20.0, 0.01)
    mm = np.trunc(np.clip((t_send - t_send[0]) / 4.0, 0, 1)[:, None] * 1000).astype(int)
    t_onboard = np.arange(0.0, 20.0, 1 / LOG_RATE)
    echo = onboard_echo(t_send, mm, 1.7e9 + t_onboard, np.full(len(t_send), 0.003))
    assert fit_clock(t_send, mm, t_onboard, echo, 1.7e9 + 0.5).n_matches > 100
    with pytest.raises(RuntimeError, match="echoes"):
        fit_clock(t_send, mm, t_onboard, echo, 1.7e9 + 5.0)


def test_change_events():
    t = np.arange(6.0)
    values = np.array([[0, 1], [0, 1], [1, 1], [1, 1], [1, 2], [0, 2]])
    times, new = change_events(t, values)
    np.testing.assert_array_equal(times, [2.0, 4.0, 5.0])
    np.testing.assert_array_equal(new, [[1, 1], [1, 2], [0, 2]])


def test_find_sd_file_takes_new_and_old_card_names(tmp_path: Path):
    """Dated names of firmware that names the files, numbered ones of older flights, overrides."""
    folder = tmp_path / "flight1_2026-11-22_10-00-00"
    for role, name in (("victim", "vic_2026-11-22_10-00-00_SD"), ("source", "src07")):
        (folder / "sd" / role).mkdir(parents=True)
        (folder / "sd" / role / name).write_bytes(b"\xbc")
    meta = {"usd": {"victim": {"files": ["vic_2026-11-22_10-00-00_SD"]}, "source": {"files": []}}}
    victim = a3_convert.find_sd_file(folder, "victim", None, meta)
    assert victim == folder / "sd" / "victim" / "vic_2026-11-22_10-00-00_SD"
    # The console line was lost, and the only file there is an old numbered one
    assert a3_convert.find_sd_file(folder, "source", None, meta).name == "src07"
    meta["usd"]["source"]["files"] = ["src07"]  # Reported by the console of an older flight
    assert a3_convert.find_sd_file(folder, "source", None, meta).name == "src07"
    assert a3_convert.find_sd_file(folder, "victim", str(victim), meta) == victim
    (folder / "sd" / "victim" / "vic03").write_bytes(b"\xbc")
    assert a3_convert.find_sd_file(folder, "victim", "vic03", meta).name == "vic03"
    with pytest.raises(FileNotFoundError):
        a3_convert.find_sd_file(folder, "victim", "vic04", meta)


def test_lag_recovers_a_shift():
    t = np.arange(0.0, 2.0, 0.001)
    bump = np.exp(-(((t - 1.0) / 0.08) ** 2))
    late = np.exp(-(((t - 1.0 - 0.0073) / 0.08) ** 2))
    assert lag(t, late, t, bump, 0.05) == pytest.approx(0.0073, abs=2e-4)
    assert lag(t, bump, t, late, 0.05) == pytest.approx(-0.0073, abs=2e-4)


def test_resample_marks_gaps_and_nans():
    t_src = np.array([0.0, 0.01, 0.02, 0.5, 0.51])
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    out = resample(t_src, x, np.array([-0.1, 0.005, 0.015, 0.3, 0.505, 0.6]), max_gap=0.05)
    np.testing.assert_allclose(out[[1, 4]], [0.5, 3.5])
    assert np.isnan(out[[0, 2, 3, 5]]).all()


def test_fill_dropouts_bridges_short_runs_only():
    x = np.arange(20.0)
    x[[3, 4]] = np.nan  # Two samples: bridged
    x[8:15] = np.nan  # Seven samples: kept
    filled = fill_dropouts(x, max_run=5)
    np.testing.assert_allclose(filled[[3, 4]], [3.0, 4.0])
    assert np.isnan(filled[8:15]).all()


def test_body_forces_vanish_in_hover():
    params = model_params("cf21B_500")
    n, mass = 50, 0.045
    signals = {
        "pos": np.zeros((n, 3)),
        "quat": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        "vel": np.zeros((n, 3)),
        "vel_dot": np.zeros((n, 3)),
        "acc": np.tile([0.0, 0.0, G], (n, 1)),
        "motor_thrust": np.full((n, 4), mass * G / 4),
    }
    force_imu, force_fd = body_forces(signals, mass, params, G)
    np.testing.assert_allclose(force_imu, 0.0, atol=1e-12)
    np.testing.assert_allclose(force_fd, 0.0, atol=1e-12)
    signals["acc"] = signals["acc"] + [0.0, 0.0, -1.0]  # Pushed down at 1 m/s^2
    assert body_forces(signals, mass, params, G)[0][0, 2] == pytest.approx(-mass)


def test_derive_finds_the_crossing_and_the_phases():
    cfg = load_config(require_masses=False)
    a3 = cfg["a3"]
    row = build_design(cfg, single=(1.0, 0.8))[0][0]
    z = victim_hover_point(cfg)[2] + 0.8
    p_start, p_end = np.array([row["x_start"], 0.0, z]), np.array([row["x_end"], 0.0, z])
    segment = Pass(p_start, p_end, 1.0, a3["ramp"])
    t = np.arange(0.0, a3["hover_time"] + segment.duration + a3["pause_time"], 0.002)
    pos, vel, _ = segment.evaluate(np.clip(t - a3["hover_time"], 0.0, segment.duration))
    p_vic = np.tile(victim_hover_point(cfg), (len(t), 1))
    derived, phase = derive(t, pos, vel, p_vic, row, a3)
    cross = a3["hover_time"] + segment.duration / 2
    assert derived["cross_time"] == pytest.approx(cross, abs=1e-6)
    assert derived["cross_speed"] == pytest.approx(1.0, abs=1e-6)
    assert derived["height_sep"] == pytest.approx(0.8)
    assert derived["direction"] == 1
    assert derived["settled_start"] == pytest.approx(cross - 0.6, abs=1e-6)
    assert derived["settled_end_ext"] == pytest.approx(
        cross + 0.6 + row["arrival_s"] + 0.15, abs=1e-6
    )
    assert set(np.unique(phase)) == {0, 1, 2, 3, 4}
    ramp_end = a3["hover_time"] + a3["ramp"]
    assert derived["cruise_start"] == pytest.approx(ramp_end, abs=0.15)  # 95 % of the speed


def write_usd_log(path: Path, names: tuple[str, ...], t_us: np.ndarray, columns: dict):
    """An SD log as the firmware writes it, CRC included."""

    def kind(name: str) -> str:
        if name == "stateEstimateZ.quat":
            return "I"
        return "H" if name.endswith("_rpm") or name == "pm.vbatMV" else "f"

    kinds = "".join(kind(name) for name in names)
    data = b"\xbc" + struct.pack("<HH", 2, 1) + struct.pack("<H", 0xFFFF) + b"fixedFrequency\0"
    data += struct.pack("<H", len(names))
    data += b"".join(f"{name}({kind(name)})".encode() + b"\0" for name in names)
    record = struct.Struct("<HQ" + kinds)
    typed = [
        np.asarray(columns[name]).astype(np.int64 if kind(name) in "HI" else np.float64).tolist()
        for name in names
    ]
    rows = zip(t_us, *typed)
    data += b"".join(record.pack(0xFFFF, int(t), *values) for t, *values in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data + struct.pack("<I", zlib.crc32(data)))


def rpm_for(thrust_per_rotor: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Rotor speed that gives the thrust on crazyflow's curve."""
    c0, c1, c2 = coef
    return (-c1 + np.sqrt(c1**2 - 4 * c2 * (c0 - thrust_per_rotor))) / (2 * c2)


def synthetic_flight(
    folder: Path, only_source: bool = False, victim_gap: tuple[float, float] | None = None
) -> dict:
    """A one-leg flight flown on the fake drones, its SD logs and bag written, then converted.

    Args:
        folder: The flight folder to write.
        only_source: Fly the source alone.
        victim_gap: Seconds from the crossing over which the victim's card writes nothing, as when
            it stalls.
    """
    monkeypatch = pytest.MonkeyPatch()
    clock = FakeClock()
    patch_pair(monkeypatch, pair, clock)
    done = threading.Event()
    done.set()
    monkeypatch.setattr(pair, "calibrate_async", lambda drone_id, delay: done)
    monkeypatch.setattr(usd, "read_once", fake_read_once)

    cfg = load_config(require_masses=False)
    masses = {"victim": 0.045, "source": 0.046}
    # The rotors of both give a different thrust per rpm from crazyflow's curve
    thrust_scales = {"victim": 0.9, "source": 1.1}
    for role, mass in masses.items():
        cfg["drones"][role]["mass_kg"] = mass
        cfg["drones"][role]["thrust_scale"] = thrust_scales[role]
    rows = build_design(cfg, single=(1.0, 0.8))[0]
    victim = (
        None
        if only_source
        else FakeDrone(cfg["drones"]["victim"]["id"], np.array([0.0, 0.0, 0.02]))
    )
    source = FakeDrone(cfg["drones"]["source"]["id"], np.array([-2.0, 0.0, 0.02]))
    targets = {
        "victim": victim_hover_point(cfg),
        "source": np.array([rows[0]["x_start"], 0.0, 1.8]),
    }
    t_log_start = clock.time()
    takeoff = pair.takeoff_pair(victim, source, targets, cfg)
    flight = pair.run_flight(victim, source, rows, cfg)
    t_log_stop = clock.time() + 1.0
    monkeypatch.undo()

    records = takeoff.records + flight.records
    t_zero = flight.records[0]["t_wall"] - flight.records[0]["t_perf"]
    t_cross = t_zero + flight.source_timeline.leg_times(0)["cross"]
    timelines = {"victim": flight.victim_timeline, "source": flight.source_timeline}
    t_takeoff = np.array([r["t_wall"] for r in takeoff.records])

    def truth(role: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Motion of a drone in PC time: the takeoff ramp, then its timeline, analytic."""
        pos, vel, acc = timelines[role].state(np.maximum(t - t_zero, 0.0))
        ramp = np.array([r[role]["cmd_pos"] for r in takeoff.records])
        before = t < t_zero
        pos[before] = np.stack([np.interp(t[before], t_takeoff, ramp[:, i]) for i in range(3)], 1)
        vel[before], acc[before] = 0.0, 0.0
        return pos, vel, acc

    flying = {"source": source} if only_source else {"victim": victim, "source": source}
    clocks = {"victim": (1.7e9 + 50.0, 40e-6), "source": (1.7e9 + 37.0, -25e-6)}
    clocks = {role: clocks[role] for role in flying}
    coef = np.asarray(model_params("cf21B_500")["rpm2thrust"]).reshape(-1)
    rng = np.random.default_rng(1)
    for role, (offset, drift) in clocks.items():
        names = usd.role_config("a3", role).variables
        t_send = np.array([r["t_wall"] for r in records])
        mm = np.array([r[role]["cmd_mm"] for r in records])
        t_onboard = np.arange(
            (t_log_start + 0.02 - offset) / (1 + drift),
            (t_log_stop - offset) / (1 + drift),
            1 / LOG_RATE,
        )
        t_true = offset + (1 + drift) * t_onboard
        pos, vel, accel = truth(role, t_true)
        thrust = masses[role] * (G + accel[:, 2])
        if role == "victim":  # The wake: 30 mN down, 0.25 s after the crossing
            accel[:, 2] -= (
                0.03 / masses[role] * np.exp(-0.5 * ((t_true - t_cross - 0.25) / 0.05) ** 2)
            )
        rpm = np.round(np.tile(rpm_for(thrust / 4, thrust_scales[role] * coef)[:, None], (1, 4)))
        rpm[rng.choice(len(rpm), 20, replace=False), rng.integers(0, 4, 20)] = usd.RPM_INVALID
        echo = onboard_echo(t_send, mm, t_true, rng.uniform(LATENCY[0], LATENCY[1], len(t_send)))
        columns = {
            "stateEstimate.x": pos[:, 0],
            "stateEstimate.y": pos[:, 1],
            "stateEstimate.z": pos[:, 2],
            "stateEstimate.vx": vel[:, 0],
            "stateEstimate.vy": vel[:, 1],
            "stateEstimate.vz": vel[:, 2],
            "stateEstimateZ.quat": np.full(len(t_true), int(usd.quatcompress([0, 0, 0, 1])[0])),
            "acc.x": accel[:, 0] / G,
            "acc.y": accel[:, 1] / G,
            "acc.z": (accel[:, 2] + G) / G,
            "pm.vbatMV": np.full(len(t_true), 3900),
            "ctrlMel.cmd_thrust": np.full(len(t_true), 40000.0),
            "ctrltarget.x": echo[:, 0] / 1000,
            "ctrltarget.z": echo[:, 2] / 1000,
        }
        columns |= {f"gyro.{axis}": np.zeros(len(t_true)) for axis in "xyz"}
        columns |= {f"motor.m{i + 1}_rpm": rpm[:, i] for i in range(4)}
        written = np.ones(len(t_true), dtype=bool)
        if role == "victim" and victim_gap is not None:
            written = ~((t_true >= t_cross + victim_gap[0]) & (t_true <= t_cross + victim_gap[1]))
        columns = {key: np.asarray(value)[written] for key, value in columns.items()}
        name = usd.role_config("a3", role).name + "00"
        t_us = np.round(t_onboard[written] * 1e6)
        write_usd_log(folder / "sd" / role / name, names, t_us, columns)

    typestore = get_typestore(Stores.ROS2_FOXY)
    types = typestore.types
    t_mocap = np.arange(t_log_start - 1.0, t_log_stop, 1 / 200)
    bag = folder / "bag"
    poses = {role: truth(role, t_mocap)[0] for role in flying}
    with Writer(bag, version=8) as writer:
        connection = writer.add_connection("/tf", "tf2_msgs/msg/TFMessage", typestore=typestore)
        for k, t in enumerate(t_mocap):
            transforms = []
            for role, drone in flying.items():
                p = poses[role][k]
                stamp = types["builtin_interfaces/msg/Time"](
                    sec=int(t), nanosec=int(round((t % 1) * 1e9)) % 1_000_000_000
                )
                transforms.append(
                    types["geometry_msgs/msg/TransformStamped"](
                        header=types["std_msgs/msg/Header"](stamp=stamp, frame_id="world"),
                        child_frame_id=drone.name,
                        transform=types["geometry_msgs/msg/Transform"](
                            translation=types["geometry_msgs/msg/Vector3"](x=p[0], y=p[1], z=p[2]),
                            rotation=types["geometry_msgs/msg/Quaternion"](
                                x=0.0, y=0.0, z=0.0, w=1.0
                            ),
                        ),
                    )
                )
            message = types["tf2_msgs/msg/TFMessage"](transforms=transforms)
            data = typestore.serialize_cdr(message, "tf2_msgs/msg/TFMessage")
            writer.write(connection, int(t * 1e9), data)

    meta = {
        "block": "A3",
        "flight": 1,
        "only_source": only_source,
        "created": "2026-11-16T12:00:00+01:00",
        "argv": [],
        "hostname": "test",
        "git": {"hash": "0" * 40, "dirty": False},
        "config": cfg,
        "design_file": "synthetic",
        "design_rows": rows,
        "masses": masses,
        "usd": {
            role: {"t_wall_start": t_log_start, "files": [usd.role_config("a3", role).name + "00"]}
            for role in flying
        },
        "abort_reason": flight.abort_reason,
        "skipped_legs": flight.skipped,
        "leg_flags": flight.leg_flags,
        "timeline": {
            "victim": flight.victim_timeline.table(),
            "source": flight.source_timeline.table(),
        },
    }
    write_outputs(folder, json.loads(json.dumps(meta, default=float)), takeoff, flight)
    summary = a3_convert.convert_flight(folder, bag, {"victim": None, "source": None}, 0.005)
    return {"folder": folder, "summary": summary, "clocks": clocks, "rows": rows, "cfg": cfg}


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The synthetic flight with both drones, converted."""
    return synthetic_flight(tmp_path_factory.mktemp("flight1_synthetic"))


def test_conversion_synchronizes_both_clocks(converted: dict):
    sync = json.loads((converted["folder"] / "sync.json").read_text())
    for role, (offset, drift) in converted["clocks"].items():
        fit = ClockFit(
            **{k: sync[role]["fit"][k] for k in ("offset", "drift", "residual_std", "n_matches")}
        )
        # The victim echoes only its takeoff and hops, which pins the drift of a 35 s flight to a
        # few ppm; a real flight spans three minutes
        assert fit.drift == pytest.approx(drift, abs=6e-6)
        assert fit.offset == pytest.approx(offset - MAPPING_LEAD, abs=1e-3)
        assert fit.residual_std < 1e-3
        assert sync[role]["hop_lags_ok"]
        assert len(sync[role]["hop_lags_ms"]) == 2
        # The motion capture is exact, so the hops show the lead of the mapping
        for lag_ms in sync[role]["hop_lags_ms"]:
            assert lag_ms == pytest.approx(MAPPING_LEAD * 1000, abs=1.0)


def test_conversion_writes_the_leg(converted: dict):
    leg_dir = converted["folder"] / "leg00"
    states = np.load(leg_dir / "states.npz")
    n = len(states["t"])
    assert states["t"][0] == 0.0
    assert np.diff(states["t"]) == pytest.approx(np.full(n - 1, 1 / 500))
    for name, tail in {
        "pos": (3,),
        "vel": (3,),
        "quat": (4,),
        "pos_est": (3,),
        "ang_vel": (3,),
        "acc": (3,),
        "rotor_vel": (4,),
        "motor_thrust": (4,),
        "force_imu": (3,),
        "force_fd": (3,),
    }.items():
        assert states[name].shape == (n, 2, *tail), name
        assert np.isfinite(states[name]).all(), name
    assert states["vbat"].shape == (n, 2)
    assert float(states["sim/freq"]) == 500.0
    meta = json.loads(str(states["meta/json"]))
    assert meta == json.loads((leg_dir / "meta.json").read_text())
    assert (leg_dir / "quicklook.png").stat().st_size > 0
    np.testing.assert_allclose(states["vbat"], 3.9, atol=1e-6)
    coef = np.asarray(model_params("cf21B_500")["rpm2thrust"]).reshape(1, -1)
    np.testing.assert_allclose(states["drone/rpm2thrust"], 0.9 * coef)
    np.testing.assert_allclose(states["drone/rpm2thrust_upper"], 1.1 * coef)


def test_conversion_measures_the_pass_and_the_wake(converted: dict):
    meta = json.loads((converted["folder"] / "leg00" / "meta.json").read_text())
    derived, quality = meta["derived"], meta["quality"]
    a3, row = converted["cfg"]["a3"], converted["rows"][0]
    assert derived["cross_time"] == pytest.approx(a3["hover_time"] + row["D"] / 2, abs=0.005)
    assert derived["cross_speed"] == pytest.approx(1.0, abs=0.01)
    assert derived["height_sep"] == pytest.approx(0.8, abs=1e-3)
    assert derived["lateral_offset"] == pytest.approx(0.0, abs=1e-3)
    assert quality["peak_fz"] == pytest.approx(-0.03, abs=0.003)
    assert quality["lag"] == pytest.approx(0.25, abs=0.01)
    assert quality["in_wake_outside_settled"] == 0
    for name in ("lower", "upper"):  # Only on the calibrated curves
        assert quality["thrust_ratio"][name] == pytest.approx(1.0, abs=0.005)
    assert sum(quality["rpm_dropouts"]["lower"]) > 0  # Counted, and bridged
    assert not quality["hard_fail"], quality["hard_fail_reasons"]
    bias = np.array(quality["baseline_pre"]["force_imu_bias"])
    assert np.abs(bias).max() < 1e-3


def test_conversion_writes_the_flight_summary(converted: dict):
    folder = converted["folder"]
    assert [row["status"] for row in converted["summary"]] == ["ok"]
    assert (folder / "summary.csv").read_text().startswith("leg,")
    assert (folder / "summary.png").stat().st_size > 0
    checks = json.loads((folder / "checks.json").read_text())
    assert all(entry["drift_ok"] and entry["hop_lags_ok"] for entry in checks["sync"].values())
    assert checks["coverage"]["in_wake_inside_settled"]
    assert math.isclose(checks["thrust"]["victim"]["mean"], 1.0, abs_tol=0.005)


def test_a_card_stall_outside_the_label_window_keeps_the_leg(tmp_path: Path):
    """F1 (18 Nov) lost 5 legs to stalls in the pause or the hover, far from the wake."""
    result = synthetic_flight(tmp_path / "flight1_gap_in_pause", victim_gap=(3.0, 3.3))
    (row,) = result["summary"]
    assert row["status"] == "ok", row["hard_fail"]
    assert {"sd_gap_outside_label_lower", "rpm_gap_outside_label_lower"} <= set(
        row["soft_flags"].split(";")
    )
    assert row["peak_fz"] == pytest.approx(-0.03, abs=0.003)  # The label is untouched
    states = np.load(result["folder"] / "leg00" / "states.npz")
    t, cross = states["t"], row["cross_time"]
    stalled = (t > cross + 3.05) & (t < cross + 3.25)
    assert np.isnan(states["acc"][stalled, 0]).all()
    assert np.isfinite(states["acc"][~stalled & (t < cross + 2.9), 0]).all()


def test_a_card_stall_in_the_label_window_fails_the_leg(tmp_path: Path):
    """F1 (18 Nov) leg 1: the victim's card stalled 0.55 s after the crossing."""
    result = synthetic_flight(tmp_path / "flight1_gap_in_wake", victim_gap=(0.5, 0.7))
    (row,) = result["summary"]
    assert row["status"] == "hard_fail"
    assert {"sd_gap_lower", "rpm_gap_lower"} <= set(row["hard_fail"].split(";"))


def test_conversion_of_the_source_alone(tmp_path: Path):
    """F0: the source flies its timeline alone, the victim's arrays stay NaN."""
    result = synthetic_flight(tmp_path / "flight1_only_source", only_source=True)
    (row,) = result["summary"]
    assert row["status"] == "ok"
    assert row["cross_speed"] == pytest.approx(1.0, abs=0.01)
    assert math.isnan(row["peak_fz"])
    states = np.load(result["folder"] / "leg00" / "states.npz")
    assert np.isnan(states["pos"][:, 0]).all()
    assert np.isfinite(states["pos"][:, 1]).all()
    assert np.isfinite(states["force_imu"][:, 1]).all()
