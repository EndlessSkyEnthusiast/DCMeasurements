# -*- coding: utf-8 -*-
"""
Refactored DC measurement GUI for Kiutra + Keithley 2636A. Made By Benedikt Schoof


The old Main.py grew organically.  This file keeps the important hardware
semantics, but separates settings, cryostat control, saving, plotting and the
individual measurements.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from Connection_Codes import ConnectKeithley_ASRL5_TSP, ConnectKiutra
except Exception as exc:  # lets the GUI show a useful error instead of dying
    ConnectKeithley_ASRL5_TSP = None
    ConnectKiutra = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class EmergencyOutputGuard:
    """Best-effort shutdown hook for both SMU outputs."""

    _hardware: Optional["Hardware"] = None
    _lock = threading.RLock()
    _installed = False
    _old_excepthook: Optional[Callable[..., Any]] = None
    _old_threading_excepthook: Optional[Callable[..., Any]] = None
    _old_signal_handlers: Dict[int, Any] = {}

    @classmethod
    def register(cls, hardware: "Hardware") -> None:
        with cls._lock:
            cls._hardware = hardware
            if cls._installed:
                return
            cls._installed = True
            atexit.register(cls.output_off)
            cls._old_excepthook = sys.excepthook
            sys.excepthook = cls._excepthook
            if hasattr(threading, "excepthook"):
                cls._old_threading_excepthook = threading.excepthook
                threading.excepthook = cls._threading_excepthook
            for signame in ("SIGINT", "SIGTERM"):
                sig = getattr(signal, signame, None)
                if sig is None:
                    continue
                try:
                    cls._old_signal_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, cls._signal_handler)
                except Exception:
                    pass

    @classmethod
    def output_off(cls) -> None:
        with cls._lock:
            hardware = cls._hardware
        if hardware is None:
            return
        try:
            hardware.output_off_all()
        except Exception:
            pass

    @classmethod
    def _excepthook(cls, exc_type: type, exc: BaseException, tb: Any) -> None:
        cls.output_off()
        if cls._old_excepthook:
            cls._old_excepthook(exc_type, exc, tb)

    @classmethod
    def _threading_excepthook(cls, args: Any) -> None:
        cls.output_off()
        if cls._old_threading_excepthook:
            cls._old_threading_excepthook(args)

    @classmethod
    def _signal_handler(cls, signum: int, frame: Any) -> None:
        cls.output_off()
        old_handler = cls._old_signal_handlers.get(signum)
        if callable(old_handler):
            old_handler(signum, frame)
            return
        if signum == getattr(signal, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))


APP_TITLE = "Kiutra DC Measurements - Refactored"
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MainRefractored_settings.json")

DEFAULT_PATH = "//nas.ads.mwn.de/tuei/lab/ZEITlab-Equipment/00314_Kiutra_Cryostat_MOL_EG.058/Userdata"
DEFAULT_SAVE_PATH = (
    "//nas.ads.mwn.de/tuei/lab/ZEITlab-Equipment/00314_Kiutra_Cryostat_MOL_EG.058/Userdata/SweepSafes"
)
DEFAULT_BACKUP_PATH = "C:/Users/ge36kuc/Desktop/SafesBackup"

INFO_TEXT = """General settings

SMU On/Off:
  Selects whether SMU1 or SMU2 is used for the next measurement.

2 Point / 4 Point:
  Selects local sense or remote sense. 4 Point uses the sense terminals of the selected SMU.

Source Voltage mV / Source Current uA:
  Selects the sourced quantity. The routines measure the complementary quantity:
  current source measures voltage, voltage source measures current.

Parallel / Sequential:
  Parallel sets both active SMUs to the next setpoint, waits once, and then reads both channels.
  When both channels belong to the same Keithley 2636A, this is attempted as one TSP command.
  Sequential measures the full SMU1 sequence first and then the full SMU2 sequence.

Current Limit / Voltage Limit:
  Empty fields leave the current SMU settings unchanged. Entered values are applied when a measurement starts.

Put Header:
  Writes the most important settings into the first line of the TXT save files.

NPLC:
  Empty field keeps the current SMU value. Smaller values are faster, larger values are lower noise.

Autozero:
  Once performs one autozero before a measurement run. Automatic keeps autozero active.
  Now performs one autozero immediately.

Jitter:
  For cyclic measurements, the first cycle stays exactly nominal. Later cycles randomly move inner points
  by up to 50 percent of the local point spacing. Zero and the positive/negative extrema stay exact.

Fast Cooldown:
  If the current temperature is above 7 K and the target is below 5 K, TemperatureControl is stopped and
  the sample heat switch is closed best-effort. Below 5 K, the normal target control is started.

Measurements

Simple DC:
  Manual sweep segments with Min, Max, and Steps. SMU modes and limits come from General settings.

Cyclic DC:
  Measures 0 -> +Max -> 0 -> -Max -> 0. Empty Cycles or 0 runs until Save Now.
  In Parallel mode, both SMUs are set together for each sweep point and then read out.
  In continuous mode, shorter active SMU cycles repeat so they can keep running while the longer cycle finishes.

Tc Measurement:
  Can only be started when all active SMUs are current sources. Applies a constant current and measures voltage
  while the target temperatures are approached.

IVTemp continuous:
  Measures IV sweeps continuously until the respective temperature checkpoints are reached.

IVTemp at Temperatures:
  Moves to each target temperature, waits for the stabilization time, and then measures the IV curve.

Find JJ:
  Searches Ic with a soft current ramp and then performs a finer sweep around Ic.

TempJJ continuous:
  Performs Find JJ once at the beginning and then only fine sweeps. If the threshold is no longer crossed,
  a full Find-JJ run is performed again.

TempJJ at Temperatures:
  Moves to temperature points and performs Find JJ at each point.

Fraunhofer Pattern:
  SMU1 searches Ic, SMU2 applies the coil current. The conversion factor converts SMU2 current into expected field.

Plots

Live Plot:
  X, Y, and Color can be selected freely. Auto scales to the visible data range and chooses suitable units
  such as nA, uA, mV, or uV. Est shows a dynamic remaining-time estimate once enough data points are available.

Session Plots:
  Show all points measured since program start for each SMU. Reset hides older points without deleting data.
"""

SOURCE_CURRENT = "current"
SOURCE_VOLTAGE = "voltage"
SEQUENCE_PARALLEL = "parallel"
SEQUENCE_SEQUENTIAL = "sequential"
TEMP_GUARD_SAVE_STOP = "save_stop"
TEMP_GUARD_WAIT_CONTINUE = "wait_continue"

MIN_RESISTANCE_CURRENT_A = 1e-11
PLOT_FIELDS = ["Time", "Voltage", "Current", "Resistance", "Delta R", "Temperature", "Magnetic Field"]
PLOT_FIELD_KEYS = {
    "Time": "elapsed_s",
    "Voltage": "voltage",
    "Current": "current",
    "Temperature": "temperature",
    "Magnetic Field": "magnetic_field",
}
SMU_COLORS = {"SMU1": "#1f77b4", "SMU2": "#d62728"}


def fit_window_to_content(
    window: tk.Misc,
    min_w: int = 260,
    min_h: int = 140,
    max_w: int = 980,
    max_h: int = 760,
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> None:
    try:
        window.update_idletasks()
        width = max(min_w, min(max_w, window.winfo_reqwidth() + 12))
        height = max(min_h, min(max_h, window.winfo_reqheight() + 12))
        if x is None:
            x = max(0, window.winfo_x())
        if y is None:
            y = max(0, window.winfo_y())
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.minsize(min_w, min_h)
    except Exception:
        pass


def refit_parent_window(widget: tk.Widget) -> None:
    try:
        top = widget.winfo_toplevel()
        screen_h = max(360, top.winfo_screenheight() - 80)
        fit_window_to_content(top, min_w=top.winfo_width(), min_h=top.winfo_height(), max_w=1120, max_h=screen_h)
    except Exception:
        pass


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not is_number(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def choose_axis_scale(field: str, values: Sequence[float]) -> Tuple[float, str]:
    finite = [abs(float(v)) for v in values if is_number(v)]
    max_abs = max(finite) if finite else 0.0
    if field == "Current":
        units = [(1.0, "A"), (1e-3, "mA"), (1e-6, "uA"), (1e-9, "nA"), (1e-12, "pA")]
    elif field == "Voltage":
        units = [(1.0, "V"), (1e-3, "mV"), (1e-6, "uV"), (1e-9, "nV")]
    elif field in ("Resistance", "Delta R"):
        units = [(1e9, "GOhm"), (1e6, "MOhm"), (1e3, "kOhm"), (1.0, "Ohm"), (1e-3, "mOhm")]
    elif field == "Time":
        if max_abs >= 3600:
            return 3600.0, "h"
        if max_abs >= 120:
            return 60.0, "min"
        return 1.0, "s"
    elif field == "Temperature":
        return 1.0, "K"
    elif field == "Magnetic Field":
        return 1.0, "arb."
    else:
        return 1.0, ""
    for factor, unit in units:
        if max_abs >= factor:
            return factor, unit
    return units[-1]


def raw_values(rows: Sequence[Dict[str, Any]], field: str) -> List[float]:
    if field == "Resistance":
        out = []
        for row in rows:
            voltage = row.get("voltage", float("nan"))
            current = row.get("current", float("nan"))
            if not is_number(voltage) or not is_number(current) or abs(float(current)) < MIN_RESISTANCE_CURRENT_A:
                out.append(float("nan"))
            else:
                out.append(float(voltage) / float(current))
        return out
    if field == "Delta R":
        out: List[float] = []
        previous_by_smu: Dict[Any, Tuple[float, float]] = {}
        for row in rows:
            smu_name = row.get("smu", "_")
            voltage = row.get("voltage", float("nan"))
            current = row.get("current", float("nan"))
            value = float("nan")
            if is_number(voltage) and is_number(current) and abs(float(current)) >= MIN_RESISTANCE_CURRENT_A:
                previous = previous_by_smu.get(smu_name)
                if previous is not None:
                    prev_voltage, prev_current = previous
                    delta_current = float(current) - prev_current
                    if abs(delta_current) >= MIN_RESISTANCE_CURRENT_A:
                        value = (float(voltage) - prev_voltage) / delta_current
                previous_by_smu[smu_name] = (float(voltage), float(current))
            out.append(value)
        return out
    key = PLOT_FIELD_KEYS[field]
    out = []
    for row in rows:
        val = row.get(key, float("nan"))
        out.append(float(val) if is_number(val) else float("nan"))
    return out


def scaled_values(rows: Sequence[Dict[str, Any]], field: str) -> Tuple[List[float], str, float]:
    raw = raw_values(rows, field)
    factor, unit = choose_axis_scale(field, raw)
    factor = factor or 1.0
    return [(v / factor if is_number(v) else float("nan")) for v in raw], f"{field} [{unit}]", factor


def padded_limits(values: Sequence[float]) -> Optional[Tuple[float, float]]:
    finite = [float(v) for v in values if is_number(v)]
    if not finite:
        return None
    lo, hi = min(finite), max(finite)
    if abs(hi - lo) < 1e-15:
        pad = max(abs(lo) * 0.05, 1.0)
    else:
        pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def apply_auto_limits(ax: Any, xs: Sequence[float], ys: Sequence[float]) -> None:
    xlim = padded_limits(xs)
    ylim = padded_limits(ys)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sanitize_filename(text: str, fallback: str = "measurement") -> str:
    value = (text or "").strip() or fallback
    bad = '<>:"/\\|?*'
    out = "".join("_" if ch in bad else ch for ch in value)
    out = "_".join(out.split())
    return out or fallback


def parse_optional_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    return float(text)


def parse_optional_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    if not text:
        return None
    return int(float(text))


def unit_to_si(source_mode: str, value: float) -> float:
    if source_mode == SOURCE_CURRENT:
        return float(value) * 1e-6
    return float(value) * 1e-3


def si_to_user(source_mode: str, value: float) -> float:
    if source_mode == SOURCE_CURRENT:
        return float(value) * 1e6
    return float(value) * 1e3


def source_unit(source_mode: str) -> str:
    return "uA" if source_mode == SOURCE_CURRENT else "mV"


def ensure_dir(path: str) -> bool:
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False


def temp_range_tag(rows: Sequence[Dict[str, Any]]) -> str:
    temps = [float(r["temperature"]) for r in rows if is_number(r.get("temperature"))]
    if not temps:
        return ""
    lo, hi = min(temps), max(temps)
    if abs(hi - lo) < 1e-6:
        return f"{lo:.4g}K"
    return f"{lo:.4g}K-{hi:.4g}K"


def is_number(value: Any) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except Exception:
        return False


def linspace_inclusive(start: float, end: float, steps: int) -> List[float]:
    n = max(1, int(steps))
    if n == 1:
        return [float(start)]
    return [float(x) for x in np.linspace(float(start), float(end), n)]


def cyclic_levels(max_abs: float, steps: int) -> List[float]:
    n = max(1, int(steps))
    max_abs = abs(float(max_abs))
    parts = [
        np.linspace(0.0, max_abs, n + 1),
        np.linspace(max_abs, 0.0, n + 1)[1:],
        np.linspace(0.0, -max_abs, n + 1)[1:],
        np.linspace(-max_abs, 0.0, n + 1)[1:],
    ]
    return [float(x) for x in np.concatenate(parts)]


def jitter_nominal_levels(levels: Sequence[float], fraction: float = 0.5) -> List[float]:
    if len(levels) <= 2:
        return [float(v) for v in levels]
    values = [float(v) for v in levels]
    max_abs = max(abs(v) for v in values)
    out = values[:]
    eps = max(max_abs * 1e-12, 1e-18)
    for idx in range(1, len(values) - 1):
        val = values[idx]
        if abs(val) <= eps or abs(abs(val) - max_abs) <= eps:
            continue
        prev_delta = abs(val - values[idx - 1])
        next_delta = abs(values[idx + 1] - val)
        nominal_delta = min(d for d in (prev_delta, next_delta) if d > eps) if (prev_delta > eps or next_delta > eps) else 0.0
        if nominal_delta <= 0:
            continue
        out[idx] = val + float(np.random.uniform(-fraction * nominal_delta, fraction * nominal_delta))
    return out


def center_out(values: Sequence[float]) -> List[float]:
    return sorted([float(v) for v in values], key=lambda x: (abs(x), x < 0, x))


@dataclass
class SMUConfig:
    enabled: bool = True
    four_point: bool = True
    source_mode: str = SOURCE_CURRENT
    filename: str = "SMU"


@dataclass
class GeneralConfig:
    smu1: SMUConfig = field(default_factory=lambda: SMUConfig(True, True, SOURCE_CURRENT, "SMU1"))
    smu2: SMUConfig = field(default_factory=lambda: SMUConfig(False, True, SOURCE_CURRENT, "SMU2"))
    sequence: str = SEQUENCE_PARALLEL
    current_limit: Optional[float] = None
    voltage_limit: Optional[float] = None
    put_header: bool = True
    nplc: Optional[float] = None
    autozero: str = "Once"
    jitter: bool = False
    autorange_voltage: bool = True
    voltage_range_smu1_mV: Optional[float] = None
    voltage_range_smu2_mV: Optional[float] = None
    autorange_current: bool = True
    current_range_smu1_uA: Optional[float] = None
    current_range_smu2_uA: Optional[float] = None
    save_path: str = DEFAULT_SAVE_PATH
    backup_path: str = DEFAULT_BACKUP_PATH
    fast_cooldown: bool = True
    temperature_guard_enabled: bool = False
    temperature_guard_mode: str = TEMP_GUARD_SAVE_STOP
    temperature_guard_min_K: Optional[float] = None
    temperature_guard_max_K: Optional[float] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "GeneralConfig":
        cfg = GeneralConfig()
        if not isinstance(data, dict):
            return cfg
        for key in (
            "sequence",
            "put_header",
            "nplc",
            "autozero",
            "jitter",
            "autorange_voltage",
            "autorange_current",
            "save_path",
            "backup_path",
            "fast_cooldown",
            "temperature_guard_enabled",
            "temperature_guard_mode",
        ):
            if key in data:
                setattr(cfg, key, data[key])
        for key in (
            "current_limit",
            "voltage_limit",
            "voltage_range_smu1_mV",
            "voltage_range_smu2_mV",
            "current_range_smu1_uA",
            "current_range_smu2_uA",
            "temperature_guard_min_K",
            "temperature_guard_max_K",
        ):
            value = data.get(key)
            setattr(cfg, key, None if value in ("", None) else float(value))
        for name in ("smu1", "smu2"):
            raw = data.get(name)
            if isinstance(raw, dict):
                setattr(cfg, name, SMUConfig(**{**asdict(getattr(cfg, name)), **raw}))
        return cfg


class PersistentState:
    def __init__(self, path: str):
        self.path = path
        self.general = GeneralConfig()
        self.measurements: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.general = GeneralConfig.from_dict(raw.get("general", {}))
            self.measurements = raw.get("measurements", {}) if isinstance(raw.get("measurements"), dict) else {}
        except Exception:
            self.general = GeneralConfig()
            self.measurements = {}

    def save(self) -> None:
        raw = {"general": asdict(self.general), "measurements": self.measurements}
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, indent=2)
        except Exception:
            pass

    def reset_general(self) -> None:
        self.general = GeneralConfig()
        self.save()

    def get_measurement(self, key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
        saved = self.measurements.get(key, {})
        return {**defaults, **saved} if isinstance(saved, dict) else defaults.copy()

    def set_measurement(self, key: str, data: Dict[str, Any]) -> None:
        self.measurements[key] = data
        self.save()


class SegmentedControl(ttk.Frame):
    def __init__(
        self,
        master: tk.Widget,
        options: Sequence[Tuple[str, str]],
        value: str,
        command: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(master)
        self.options = list(options)
        self.command = command
        self.var = tk.StringVar(value=value)
        self.buttons: Dict[str, tk.Button] = {}
        for label, option_value in self.options:
            btn = tk.Button(self, text=label, width=max(7, len(label) + 1), command=lambda v=option_value: self.set(v))
            btn.pack(side=tk.LEFT, padx=1)
            self.buttons[option_value] = btn
        self.refresh()

    def set(self, value: str) -> None:
        self.var.set(value)
        self.refresh()
        if self.command:
            self.command(value)

    def get(self) -> str:
        return self.var.get()

    def refresh(self) -> None:
        active = self.var.get()
        for value, btn in self.buttons.items():
            if value == active:
                btn.configure(relief=tk.SUNKEN, bg="#2b7cff", fg="white")
            else:
                btn.configure(relief=tk.RAISED, bg="SystemButtonFace", fg="black")


class Hardware:
    def __init__(self):
        self.client = None
        self.host = None
        self.temperature_control = None
        self.sample_control = None
        self.adr_control = None
        self.keithley = None
        self.smu1: Optional[SMUDevice] = None
        self.smu2: Optional[SMUDevice] = None
        self.errors: List[str] = []
        self.connect()

    def connect(self) -> None:
        if _IMPORT_ERROR is not None:
            self.errors.append(f"Import failed: {_IMPORT_ERROR}")
            return
        try:
            self.client, self.host, self.temperature_control, self.sample_control, self.adr_control = ConnectKiutra()
        except Exception as exc:
            self.errors.append(f"ConnectKiutra failed: {exc}")
        try:
            self.keithley = ConnectKeithley_ASRL5_TSP()
            self.smu1 = SMUDevice("SMU1", self.keithley, self.keithley.smua, "smua")
            self.smu2 = SMUDevice("SMU2", self.keithley, self.keithley.smub, "smub")
            self.output_off_all()
            for smu in self.active_smus({"SMU1": True, "SMU2": True}):
                smu.set_autozero("Once")
        except Exception as exc:
            self.errors.append(f"ConnectKeithley_ASRL5_TSP failed: {exc}")

    def active_smus(self, enabled: Dict[str, bool]) -> List["SMUDevice"]:
        out: List[SMUDevice] = []
        if enabled.get("SMU1") and self.smu1:
            out.append(self.smu1)
        if enabled.get("SMU2") and self.smu2:
            out.append(self.smu2)
        return out

    def get_smu(self, name: str) -> Optional["SMUDevice"]:
        return self.smu1 if name == "SMU1" else self.smu2

    def output_off_all(self) -> None:
        for smu in (self.smu1, self.smu2):
            if smu:
                smu.output_off()

    def read_temperature(self) -> float:
        try:
            if self.temperature_control is not None and hasattr(self.temperature_control, "kelvin"):
                return float(self.temperature_control.kelvin)
        except Exception:
            pass
        try:
            if self.client is not None:
                return float(self.client.query("T_sample.kelvin") or 300.0)
        except Exception:
            pass
        return 300.0

    def _is_heat_switch_closed(self, switch_name: str) -> Optional[bool]:
        if self.client is None:
            return None
        query = getattr(self.client, "query", None)
        if not callable(query):
            return None
        try:
            value = query(f"{switch_name}.is_closed")
        except Exception:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "closed"):
                return True
            if normalized in ("false", "0", "no", "open"):
                return False
        return None

    def _close_heat_switch(self, switch_name: str) -> bool:
        # Kiutra's Heatswitch API exposes "closed" as the close command.
        calls: List[Callable[[], Any]] = []
        if switch_name == "hs_sample" and self.sample_control is not None:
            closed = getattr(self.sample_control, "closed", None)
            if callable(closed):
                calls.append(closed)
            call_method = getattr(self.sample_control, "call_method", None)
            if callable(call_method):
                calls.append(lambda: call_method("closed"))
            mover = getattr(self.sample_control, "move", None)
            if callable(mover):
                calls.append(lambda: mover("hs_sample", "closed"))
                calls.append(lambda: mover("sample", "closed"))
        if self.client is not None:
            caller = getattr(self.client, "call", None)
            if callable(caller):
                calls.append(lambda: caller(f"{switch_name}.closed"))
            for method_name in ("command", "execute", "write", "query"):
                method = getattr(self.client, method_name, None)
                if callable(method):
                    calls.append(lambda m=method: m(f"move({switch_name},'closed')"))
        for call in calls:
            try:
                call()
                return True
            except Exception:
                continue
        return False

    def close_sample_heat_switch(self) -> None:
        for switch_name in ("hs_sample", "hs1", "hs2"):
            is_closed = self._is_heat_switch_closed(switch_name)
            if is_closed is True:
                continue
            self._close_heat_switch(switch_name)


class SMUDevice:
    def __init__(self, name: str, instrument: Any, channel: Any, tsp_channel: Optional[str] = None):
        self.name = name
        self.instrument = instrument
        self.channel = channel
        self.tsp_channel = tsp_channel

    def configure(self, smu_cfg: SMUConfig, general: GeneralConfig) -> List[str]:
        warnings: List[str] = []
        try:
            self.channel.sense = self.channel.SENSE_REMOTE if smu_cfg.four_point else self.channel.SENSE_LOCAL
        except Exception:
            pass
        if general.nplc is not None:
            try:
                self.channel.measure.nplc = float(general.nplc)
            except Exception:
                pass
        if general.current_limit is not None:
            try:
                self.channel.source.limiti = float(general.current_limit)
            except Exception:
                pass
        if general.voltage_limit is not None:
            try:
                self.channel.source.limitv = float(general.voltage_limit)
            except Exception:
                pass
        self._configure_voltage_source_range(general, warnings)
        self._configure_current_measure_range(general, warnings)
        self.set_source_mode(smu_cfg.source_mode)
        return warnings

    def _warn_config(self, warnings: List[str], setting: str, exc: Exception) -> None:
        warnings.append(f"{self.name}: {setting} was not applied ({exc})")

    def _configure_voltage_source_range(self, general: GeneralConfig, warnings: List[str]) -> None:
        try:
            if general.autorange_voltage:
                self.channel.source.autorangev = self.channel.AUTORANGE_ON
                return
            range_mV = general.voltage_range_smu1_mV if self.name == "SMU1" else general.voltage_range_smu2_mV
            if range_mV is None:
                return
            self.channel.source.autorangev = self.channel.AUTORANGE_OFF
            self.channel.source.rangev = float(range_mV) * 1e-3
        except Exception as exc:
            self._warn_config(warnings, "voltage source autorange/range", exc)

    def _configure_current_measure_range(self, general: GeneralConfig, warnings: List[str]) -> None:
        try:
            if general.autorange_current:
                self.channel.measure.autorangei = self.channel.AUTORANGE_ON
                return
            range_uA = general.current_range_smu1_uA if self.name == "SMU1" else general.current_range_smu2_uA
            if range_uA is None:
                return
            self.channel.measure.autorangei = self.channel.AUTORANGE_OFF
            self.channel.measure.rangei = float(range_uA) * 1e-6
        except Exception as exc:
            self._warn_config(warnings, "current measure autorange/range", exc)

    def set_autozero(self, mode: str) -> None:
        try:
            if mode == "Automatic":
                self.channel.measure.autozero = self.channel.AUTOZERO_AUTO
            else:
                self.channel.measure.autozero = self.channel.AUTOZERO_ONCE
        except Exception:
            try:
                self.channel.autozero = self.channel.AUTOZERO_AUTO if mode == "Automatic" else self.channel.AUTOZERO_ONCE
            except Exception:
                pass

    def set_source_mode(self, source_mode: str) -> None:
        try:
            self.channel.source.output = 1
            self.channel.source.func = 2 if source_mode == SOURCE_CURRENT else 1
        except Exception:
            pass

    def apply_source(self, source_mode: str, value_si: float) -> None:
        self.set_source_mode(source_mode)
        if source_mode == SOURCE_CURRENT:
            try:
                self.instrument.apply_current(self.channel, float(value_si))
                return
            except Exception:
                pass
            try:
                self.channel.source.leveli = float(value_si)
            except Exception:
                pass
        else:
            try:
                self.instrument.apply_voltage(self.channel, float(value_si))
                return
            except Exception:
                pass
            try:
                self.channel.source.levelv = float(value_si)
            except Exception:
                pass

    def measure(self, source_mode: str, source_value_si: Optional[float] = None) -> Tuple[float, float]:
        current = float("nan")
        voltage = float("nan")
        try:
            i_val, v_val = self.channel.measure.iv()
            current, voltage = float(i_val), float(v_val)
        except Exception:
            try:
                current = float(self.channel.measure.i())
            except Exception:
                pass
            try:
                voltage = float(self.channel.measure.v())
            except Exception:
                pass
        if source_value_si is not None:
            if source_mode == SOURCE_CURRENT and not is_number(current):
                current = float(source_value_si)
            if source_mode == SOURCE_VOLTAGE and not is_number(voltage):
                voltage = float(source_value_si)
        return voltage, current

    def output_off(self) -> None:
        try:
            self.channel.source.output = 0
        except Exception:
            pass


class CryoController:
    def __init__(self, hardware: Hardware):
        self.hardware = hardware

    def start_target(
        self,
        target: float,
        ramp: float,
        pre_regenerate: bool,
        fast_cooldown: bool,
        stop_event: threading.Event,
        status_cb: Callable[[str], None],
        fast_cooldown_work: Optional[Callable[[float], bool]] = None,
    ) -> None:
        hw = self.hardware
        current = hw.read_temperature()
        if fast_cooldown and current > 7.0 and target < 5.0:
            status_cb(f"Fast cooldown: waiting for < 5 K")
            try:
                if hw.temperature_control is not None:
                    hw.temperature_control.stop()
            except Exception:
                pass
            hw.close_sample_heat_switch()
            while not stop_event.is_set():
                current = hw.read_temperature()
                if current < 5.0:
                    break
                if fast_cooldown_work is None:
                    time.sleep(0.5)
                else:
                    if not fast_cooldown_work(current):
                        return

        if stop_event.is_set():
            return
        if target > 3.0:
            try:
                if hw.temperature_control is not None and getattr(hw.temperature_control, "is_active", False):
                    hw.temperature_control.stop()
                    time.sleep(0.5)
            except Exception:
                pass
            status_cb(f"TemperatureControl -> {target:.3f} K at {ramp:.3f} K/min")
            if hw.temperature_control is None:
                raise RuntimeError("TemperatureControl is not connected")
            hw.temperature_control.start((float(target), float(ramp)))
        else:
            try:
                if hw.temperature_control is not None and getattr(hw.temperature_control, "is_active", False):
                    hw.temperature_control.stop()
                    time.sleep(0.5)
            except Exception:
                pass
            status_cb(f"ADR -> {target:.3f} K at {ramp:.3f} K/min")
            if hw.adr_control is None:
                raise RuntimeError("ADRControl is not connected")
            hw.adr_control.start_adr(
                setpoint=float(target),
                ramp=float(ramp),
                adr_mode=None,
                operation_mode="cadr",
                auto_regenerate=True,
                pre_regenerate=bool(pre_regenerate),
            )

    def wait_for_target(
        self,
        target: float,
        tolerance: float,
        stop_event: threading.Event,
        skip_event: threading.Event,
        status_cb: Callable[[str], None],
        while_waiting: Optional[Callable[[float], None]] = None,
        interval_s: float = 0.25,
    ) -> str:
        tolerance = max(float(tolerance), 1e-6)
        while not stop_event.is_set():
            temp = self.hardware.read_temperature()
            if while_waiting:
                while_waiting(temp)
            if stop_event.is_set():
                return "stopped"
            if skip_event.is_set():
                skip_event.clear()
                status_cb(f"Skipped target {target:.3f} K")
                return "skipped"
            if abs(temp - target) <= tolerance:
                status_cb(f"Reached {target:.3f} K")
                return "reached"
            status_cb(f"T={temp:.3f} K -> {target:.3f} K")
            time.sleep(interval_s)
        return "stopped"


class TextSaver:
    def __init__(self, app: "MeasurementApp"):
        self.app = app

    def write_standard(self, run: "MeasurementRun") -> List[str]:
        paths: List[str] = []
        snapshot = run.snapshot
        for smu_name in ("SMU1", "SMU2"):
            rows = [r for r in run.rows_snapshot() if r.get("smu") == smu_name]
            if not rows:
                continue
            smu_cfg = snapshot["smu"][smu_name]
            base = sanitize_filename(smu_cfg["filename"], f"{run.name}_{smu_name}")
            tag = temp_range_tag(rows)
            filename = f"{base}_{tag}_{now_stamp()}.txt" if tag else f"{base}_{now_stamp()}.txt"
            text = self._standard_text(run, smu_name, rows)
            paths.extend(self._write_to_primary_and_backup(filename, text))
        return paths

    def write_fraunhofer(self, run: "MeasurementRun") -> List[str]:
        rows = run.fraunhofer_rows_snapshot()
        if not rows:
            return self.write_standard(run)
        smu_cfg = run.snapshot["smu"]["SMU1"]
        base = sanitize_filename(smu_cfg["filename"], f"{run.name}_Fraunhofer")
        tag = temp_range_tag(rows)
        filename = f"{base}_{tag}_{now_stamp()}.txt" if tag else f"{base}_{now_stamp()}.txt"
        header_lines = []
        if run.snapshot["general"]["put_header"]:
            header_lines.append(self._settings_header(run, "Fraunhofer"))
        header_lines.append(
            "Time\tTemperature[K]\tSMU1 Voltage[V]\tSMU1 Current[A]\t"
            "SMU2 Voltage[V]\tSMU2 Current[A]\tExpected Magnetic Field"
        )
        body = []
        for row in rows:
            body.append(
                f"{row.get('time','')}\t{float(row.get('temperature', float('nan'))):.9e}\t"
                f"{float(row.get('smu1_voltage', float('nan'))):.9e}\t"
                f"{float(row.get('smu1_current', float('nan'))):.9e}\t"
                f"{float(row.get('smu2_voltage', float('nan'))):.9e}\t"
                f"{float(row.get('smu2_current', float('nan'))):.9e}\t"
                f"{float(row.get('magnetic_field', float('nan'))):.9e}"
            )
        return self._write_to_primary_and_backup(filename, "\n".join(header_lines + body) + "\n")

    def write_iv_round(
        self,
        run: "MeasurementRun",
        smu_name: str,
        rows: Sequence[Dict[str, Any]],
        round_index: int,
        label: str = "",
    ) -> List[str]:
        if not rows:
            return []
        folders = run.iv_round_folders(smu_name)
        tag = sanitize_filename(label, "") if label else ""
        filename = f"round_{round_index:04d}_{tag}.txt" if tag else f"round_{round_index:04d}.txt"
        text = self._standard_text(run, smu_name, rows)
        paths: List[str] = []
        for folder in folders:
            paths.extend(self._write_to_folder(folder, filename, text))
        return paths

    def _standard_text(self, run: "MeasurementRun", smu_name: str, rows: Sequence[Dict[str, Any]]) -> str:
        header = []
        if run.snapshot["general"]["put_header"]:
            header.append(self._settings_header(run, smu_name))
        header.append("Time\tVoltage[V]\tCurrent[A]\tTemperature[K]")
        body = []
        for row in rows:
            body.append(
                f"{row.get('time','')}\t{float(row.get('voltage', float('nan'))):.9e}\t"
                f"{float(row.get('current', float('nan'))):.9e}\t"
                f"{float(row.get('temperature', float('nan'))):.9e}"
            )
        return "\n".join(header + body) + "\n"

    def _settings_header(self, run: "MeasurementRun", smu_name: str) -> str:
        general = run.snapshot["general"]
        smu = run.snapshot["smu"].get(smu_name, {})
        measurement_settings = "; ".join(f"{k}={v}" for k, v in sorted(run.measurement_settings.items()))
        parts = [
            f"measurement={run.name}",
            f"smu={smu_name}",
            f"enabled={smu.get('enabled')}",
            f"sense={'4Point' if smu.get('four_point') else '2Point'}",
            f"source={smu.get('source_mode')}",
            f"sequence={general.get('sequence')}",
            f"nplc={general.get('nplc')}",
            f"autozero={general.get('autozero')}",
            f"jitter={general.get('jitter')}",
            f"autorange_voltage={general.get('autorange_voltage')}",
            f"voltage_range_smu1_mV={general.get('voltage_range_smu1_mV')}",
            f"voltage_range_smu2_mV={general.get('voltage_range_smu2_mV')}",
            f"autorange_current={general.get('autorange_current')}",
            f"current_range_smu1_uA={general.get('current_range_smu1_uA')}",
            f"current_range_smu2_uA={general.get('current_range_smu2_uA')}",
            f"current_limit={general.get('current_limit')}",
            f"voltage_limit={general.get('voltage_limit')}",
            f"settings={measurement_settings}",
        ]
        return "Settings: " + "; ".join(parts)

    def _write_to_primary_and_backup(self, filename: str, text: str) -> List[str]:
        cfg = self.app.state.general
        out: List[str] = []
        for path in (cfg.save_path, cfg.backup_path):
            if not path:
                continue
            if not ensure_dir(path):
                continue
            full = os.path.join(path, filename)
            try:
                with open(full, "w", encoding="utf-8") as fh:
                    fh.write(text)
                out.append(full)
            except Exception:
                continue
        return out

    def _write_to_folder(self, folder: str, filename: str, text: str) -> List[str]:
        if not ensure_dir(folder):
            return []
        full = os.path.join(folder, filename)
        try:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)
            return [full]
        except Exception:
            return []


class MeasurementRun:
    def __init__(
        self,
        app: "MeasurementApp",
        name: str,
        snapshot: Dict[str, Any],
        measurement_settings: Dict[str, Any],
        save_kind: str = "standard",
        create_live_plot: bool = True,
    ):
        self.app = app
        self.name = name
        self.snapshot = snapshot
        self.measurement_settings = measurement_settings
        self.save_kind = save_kind
        self.stop_event = threading.Event()
        self.skip_temperature_event = threading.Event()
        self.rows: List[Dict[str, Any]] = []
        self.fraunhofer_rows: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.saved_paths: List[str] = []
        self.saved = False
        self.save_now_requested = False
        self.error: Optional[str] = None
        self.temperature_guard_active = False
        self.temperature_guard_triggered = False
        self.temperature_guard_message = ""
        self.temperature_guard_pause_count = 0
        self.iv_round_enabled = False
        self.iv_round_stamp = now_stamp()
        self.iv_round_dirs: Dict[str, List[str]] = {}
        self.iv_round_saved_counts: Dict[str, int] = {"SMU1": 0, "SMU2": 0}
        self.iv_round_index = 0
        self.estimated_total_points: Optional[int] = None
        self.estimated_total_seconds: Optional[float] = None
        self.live_plot = LiveMeasurementPlot(app, self) if create_live_plot else None

    def add_point(
        self,
        smu: str,
        voltage: float,
        current: float,
        temperature: Optional[float] = None,
        magnetic_field: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = {
            "time": now_text(),
            "elapsed_s": time.time() - self.start_time,
            "smu": smu,
            "voltage": float(voltage) if is_number(voltage) else float("nan"),
            "current": float(current) if is_number(current) else float("nan"),
            "temperature": self.app.hardware.read_temperature() if temperature is None else float(temperature),
            "magnetic_field": float(magnetic_field) if is_number(magnetic_field) else float("nan"),
        }
        if extra:
            row.update(extra)
        with self.lock:
            self.rows.append(row)
        self.app.event_queue.put(("point", row))
        return row

    def add_fraunhofer_point(
        self,
        smu1_voltage: float,
        smu1_current: float,
        smu2_voltage: float,
        smu2_current: float,
        magnetic_field: float,
        temperature: Optional[float] = None,
    ) -> None:
        temp = self.app.hardware.read_temperature() if temperature is None else float(temperature)
        row = {
            "time": now_text(),
            "elapsed_s": time.time() - self.start_time,
            "temperature": temp,
            "smu1_voltage": float(smu1_voltage),
            "smu1_current": float(smu1_current),
            "smu2_voltage": float(smu2_voltage),
            "smu2_current": float(smu2_current),
            "magnetic_field": float(magnetic_field),
        }
        with self.lock:
            self.fraunhofer_rows.append(row)
        self.add_point("SMU1", smu1_voltage, smu1_current, temp, magnetic_field, {"fraunhofer": True})
        self.add_point("SMU2", smu2_voltage, smu2_current, temp, magnetic_field, {"fraunhofer": True})

    def rows_snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.rows)

    def fraunhofer_rows_snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.fraunhofer_rows)

    def enable_iv_round_saves(self) -> None:
        self.iv_round_enabled = True

    def iv_round_folders(self, smu_name: str) -> List[str]:
        cached = self.iv_round_dirs.get(smu_name)
        if cached:
            return cached
        base_name = self.snapshot["smu"].get(smu_name, {}).get("filename", smu_name)
        folder_name = f"{sanitize_filename(base_name, smu_name)}_{self.iv_round_stamp}"
        general = self.snapshot["general"]
        folders = []
        for root in (general.get("save_path") or DEFAULT_SAVE_PATH, general.get("backup_path") or DEFAULT_BACKUP_PATH):
            if root:
                folders.append(os.path.join(root, folder_name))
        self.iv_round_dirs[smu_name] = folders
        return folders

    def save_iv_round(self, label: str = "") -> None:
        if not self.iv_round_enabled:
            return
        self.iv_round_index += 1
        rows = self.rows_snapshot()
        for smu_name in ("SMU1", "SMU2"):
            smu_rows = [r for r in rows if r.get("smu") == smu_name]
            start = self.iv_round_saved_counts.get(smu_name, 0)
            new_rows = smu_rows[start:]
            if new_rows:
                self.app.saver.write_iv_round(self, smu_name, new_rows, self.iv_round_index, label)
                self.iv_round_saved_counts[smu_name] = len(smu_rows)

    def request_save_now(self) -> None:
        self.save_now_requested = True
        self.stop_event.set()
        self.skip_temperature_event.set()
        self.app.hardware.output_off_all()

    def enable_temperature_guard(self) -> None:
        general = self.snapshot.get("general", {})
        self.temperature_guard_active = bool(general.get("temperature_guard_enabled"))

    def check_temperature_guard(self, temperature: Optional[float] = None) -> bool:
        if not self.temperature_guard_active or self.temperature_guard_triggered:
            return not self.temperature_guard_triggered
        general = self.snapshot.get("general", {})
        lo = general.get("temperature_guard_min_K")
        hi = general.get("temperature_guard_max_K")
        if lo is None and hi is None:
            return True
        bounds = self._temperature_guard_bounds(lo, hi)
        if bounds is None:
            return True
        lo_f, hi_f = bounds
        try:
            temp = self.app.hardware.read_temperature() if temperature is None else float(temperature)
        except Exception:
            return True
        if self._temperature_in_guard_range(temp, lo_f, hi_f):
            return True
        mode = general.get("temperature_guard_mode", TEMP_GUARD_SAVE_STOP)
        if mode == TEMP_GUARD_WAIT_CONTINUE:
            return self._wait_for_temperature_guard_range(temp, lo_f, hi_f)
        self.temperature_guard_triggered = True
        self.temperature_guard_message = (
            f"Temperature guard: T={temp:.3f} K outside {self._temperature_guard_bounds_text(lo_f, hi_f)}"
        )
        self.request_save_now()
        return False

    def check_temperature_guard_setpoint(self, target: float) -> bool:
        general = self.snapshot.get("general", {})
        if not general.get("temperature_guard_enabled"):
            return True
        bounds = self._temperature_guard_bounds(
            general.get("temperature_guard_min_K"), general.get("temperature_guard_max_K")
        )
        if bounds is None:
            return True
        lo_f, hi_f = bounds
        target_f = float(target)
        if self._temperature_in_guard_range(target_f, lo_f, hi_f):
            return True
        self.temperature_guard_triggered = True
        self.temperature_guard_message = (
            f"Temperature guard: target {target_f:.3f} K outside {self._temperature_guard_bounds_text(lo_f, hi_f)}"
        )
        self.request_save_now()
        return False

    def _temperature_guard_bounds(self, lo: Any, hi: Any) -> Optional[Tuple[Optional[float], Optional[float]]]:
        try:
            lo_f = float(lo) if lo is not None else None
            hi_f = float(hi) if hi is not None else None
        except Exception:
            return None
        if lo_f is not None and hi_f is not None and lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        return lo_f, hi_f

    def _temperature_in_guard_range(self, temp: float, lo: Optional[float], hi: Optional[float]) -> bool:
        return not ((lo is not None and temp < lo) or (hi is not None and temp > hi))

    def _temperature_guard_bounds_text(self, lo: Optional[float], hi: Optional[float]) -> str:
        bounds = []
        if lo is not None:
            bounds.append(f"min {lo:.3f} K")
        if hi is not None:
            bounds.append(f"max {hi:.3f} K")
        return " / ".join(bounds)

    def _wait_for_temperature_guard_range(self, temp: float, lo: Optional[float], hi: Optional[float]) -> bool:
        self.temperature_guard_pause_count += 1
        self.app.hardware.output_off_all()
        bounds_text = self._temperature_guard_bounds_text(lo, hi)
        self.temperature_guard_message = f"Temperature guard: T={temp:.3f} K outside {bounds_text}; waiting"
        while not self.stop_event.is_set():
            time.sleep(0.5)
            try:
                temp = self.app.hardware.read_temperature()
            except Exception:
                continue
            if self._temperature_in_guard_range(temp, lo, hi):
                self.temperature_guard_message = f"Temperature guard: T={temp:.3f} K inside range; continuing"
                return True
            self.temperature_guard_message = f"Temperature guard: T={temp:.3f} K outside {bounds_text}; waiting"
        return False

    def set_estimate(self, total_points: Optional[int], total_seconds: Optional[float]) -> None:
        self.estimated_total_points = total_points
        self.estimated_total_seconds = total_seconds

    def eta_text(self) -> str:
        rows_done = len(self.rows_snapshot())
        if self.estimated_total_points and self.estimated_total_points > 0:
            remaining_points = max(0, self.estimated_total_points - rows_done)
            if rows_done > 0:
                seconds_per_point = (time.time() - self.start_time) / max(1, rows_done)
                return format_duration(remaining_points * seconds_per_point)
            return "--:--:--"
        if self.estimated_total_seconds is not None:
            return format_duration(max(0.0, self.estimated_total_seconds - (time.time() - self.start_time)))
        return "--:--:--"

    def save(self) -> List[str]:
        if self.saved:
            return self.saved_paths
        if self.save_kind == "fraunhofer":
            self.saved_paths = self.app.saver.write_fraunhofer(self)
        else:
            self.saved_paths = self.app.saver.write_standard(self)
        self.saved = True
        return self.saved_paths


class LiveMeasurementPlot:
    def __init__(self, app: "MeasurementApp", run: MeasurementRun):
        self.app = app
        self.run = run
        self.window = tk.Toplevel(app.root)
        self.window.title(f"Live Plot - {run.name}")
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)

        control = ttk.Frame(self.window)
        control.pack(fill=tk.X, padx=8, pady=6)
        row1 = ttk.Frame(control)
        row1.pack(fill=tk.X)
        row2 = ttk.Frame(control)
        row2.pack(fill=tk.X, pady=(4, 0))
        self.x_var = tk.StringVar(value="Current")
        self.y_var = tk.StringVar(value="Voltage")
        self.color_var = tk.StringVar(value="Off")
        self.style_var = tk.StringVar(value="scatter")
        self.scale_var = tk.StringVar(value="auto")
        self.axis_factors = {"x": 1.0, "y": 1.0}
        for label, var, values in (
            ("X", self.x_var, PLOT_FIELDS),
            ("Y", self.y_var, PLOT_FIELDS),
            ("Color", self.color_var, ["Off"] + PLOT_FIELDS),
            ("Style", self.style_var, ["scatter", "line", "line+scatter"]),
            ("Scale", self.scale_var, ["auto", "manual"]),
        ):
            ttk.Label(row1, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Combobox(row1, textvariable=var, values=values, width=13, state="readonly").pack(side=tk.LEFT)
        self.axis_entries: Dict[str, ttk.Entry] = {}
        for label in ("xmin", "xmax", "ymin", "ymax"):
            ttk.Label(row2, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ent = ttk.Entry(row2, width=9)
            ent.pack(side=tk.LEFT)
            self.axis_entries[label] = ent
        self.estimate_label = ttk.Label(row2, text="Est: --:--:--")
        self.estimate_label.pack(side=tk.LEFT, padx=(18, 2))

        self.fig = Figure(figsize=(6.6, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        for var in (self.x_var, self.y_var, self.color_var, self.style_var, self.scale_var):
            var.trace_add("write", lambda *_: self.redraw())
        fit_window_to_content(self.window, min_w=680, min_h=470, max_w=820, max_h=620, x=360, y=140)
        self.redraw()

    def redraw(self) -> None:
        if not self.window.winfo_exists():
            return
        rows = self.run.rows_snapshot()
        self._reset_axes()
        self.estimate_label.configure(text=f"Est: {self.run.eta_text()}")
        x_field = self.x_var.get()
        y_field = self.y_var.get()
        x_all, x_label, x_factor = scaled_values(rows, x_field)
        y_all, y_label, y_factor = scaled_values(rows, y_field)
        self.axis_factors = {"x": x_factor, "y": y_factor}
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        if not rows:
            self.canvas.draw_idle()
            return
        color_field = self.color_var.get()
        c_all: List[float] = []
        c_label = ""
        c_limits = None
        mappable = None
        if color_field != "Off":
            c_all, c_label, _ = scaled_values(rows, color_field)
            c_limits = padded_limits(c_all)
        for smu_name, marker in (("SMU1", "o"), ("SMU2", "s")):
            idxs = [idx for idx, row in enumerate(rows) if row.get("smu") == smu_name]
            if not idxs:
                continue
            x = [x_all[idx] for idx in idxs]
            y = [y_all[idx] for idx in idxs]
            style = self.style_var.get()
            if color_field != "Off":
                c = [c_all[idx] for idx in idxs]
                kwargs = {}
                if c_limits:
                    kwargs.update({"vmin": c_limits[0], "vmax": c_limits[1]})
                sc = self.ax.scatter(
                    x,
                    y,
                    c=c,
                    s=22,
                    marker=marker,
                    cmap="viridis",
                    edgecolors=SMU_COLORS.get(smu_name, "black"),
                    linewidths=0.7,
                    label=smu_name,
                    **kwargs,
                )
                if mappable is None:
                    mappable = sc
                if style in ("line", "line+scatter"):
                    self.ax.plot(x, y, linewidth=0.8, alpha=0.5, color=SMU_COLORS.get(smu_name))
            elif style == "line":
                self.ax.plot(x, y, marker="", label=smu_name, color=SMU_COLORS.get(smu_name))
            elif style == "line+scatter":
                self.ax.plot(x, y, marker=marker, markersize=3, label=smu_name, color=SMU_COLORS.get(smu_name))
            else:
                self.ax.scatter(x, y, s=22, marker=marker, label=smu_name, color=SMU_COLORS.get(smu_name))
        if mappable is not None:
            try:
                self.fig.colorbar(mappable, ax=self.ax, label=c_label)
            except Exception:
                pass
        if self.scale_var.get() == "manual":
            self._apply_manual_limits()
        else:
            apply_auto_limits(self.ax, x_all, y_all)
        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    def _reset_axes(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def _apply_manual_limits(self) -> None:
        try:
            xmin = parse_optional_float(self.axis_entries["xmin"].get())
            xmax = parse_optional_float(self.axis_entries["xmax"].get())
            ymin = parse_optional_float(self.axis_entries["ymin"].get())
            ymax = parse_optional_float(self.axis_entries["ymax"].get())
            if xmin is not None or xmax is not None:
                self.ax.set_xlim(left=xmin, right=xmax)
            if ymin is not None or ymax is not None:
                self.ax.set_ylim(bottom=ymin, top=ymax)
        except Exception:
            pass


class SessionPlotWindow:
    def __init__(self, app: "MeasurementApp", smu_name: str):
        self.app = app
        self.smu_name = smu_name
        self.reset_index = 0
        self.window: Optional[tk.Toplevel] = None
        self.fig: Optional[Figure] = None
        self.ax: Any = None
        self.canvas: Any = None
        self.closed_by_user = False
        self.x_var = tk.StringVar(value="Current")
        self.y_var = tk.StringVar(value="Voltage")
        self.color_var = tk.StringVar(value="Off")
        self.style_var = tk.StringVar(value="scatter")
        self.open()

    def open(self) -> None:
        self.closed_by_user = False
        self.window = tk.Toplevel(self.app.root)
        self.window.title(f"Session Plot - {self.smu_name}")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        ctl = ttk.Frame(self.window)
        ctl.pack(fill=tk.X, padx=8, pady=5)
        for label, var, values in (
            ("X", self.x_var, PLOT_FIELDS),
            ("Y", self.y_var, PLOT_FIELDS),
            ("Color", self.color_var, ["Off"] + PLOT_FIELDS),
            ("Style", self.style_var, ["scatter", "line", "line+scatter"]),
        ):
            ttk.Label(ctl, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Combobox(ctl, textvariable=var, values=values, width=13, state="readonly").pack(side=tk.LEFT)
        ttk.Button(ctl, text="Reset", command=self.reset).pack(side=tk.RIGHT)
        self.fig = Figure(figsize=(6.0, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        for var in (self.x_var, self.y_var, self.color_var, self.style_var):
            var.trace_add("write", lambda *_: self.redraw())
        fit_window_to_content(self.window, min_w=620, min_h=420, max_w=760, max_h=560)
        self.redraw()

    def on_close(self) -> None:
        self.closed_by_user = True
        self.reset()
        if self.window and self.window.winfo_exists():
            self.window.destroy()
        self.app.root.after(10000, self.open)

    def reset(self) -> None:
        self.reset_index = len(self.app.session_rows.get(self.smu_name, []))
        self.redraw()

    def redraw(self) -> None:
        if not self.window or not self.window.winfo_exists() or self.ax is None:
            return
        rows = self.app.session_rows.get(self.smu_name, [])[self.reset_index :]
        self._reset_axes()
        self.ax.set_title(self.smu_name)
        x, x_label, _ = scaled_values(rows, self.x_var.get())
        y, y_label, _ = scaled_values(rows, self.y_var.get())
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        if rows:
            style = self.style_var.get()
            color_field = self.color_var.get()
            if color_field != "Off":
                c, c_label, _ = scaled_values(rows, color_field)
                c_limits = padded_limits(c)
                kwargs = {}
                if c_limits:
                    kwargs.update({"vmin": c_limits[0], "vmax": c_limits[1]})
                sc = self.ax.scatter(x, y, c=c, s=18, cmap="viridis", **kwargs)
                try:
                    self.fig.colorbar(sc, ax=self.ax, label=c_label)
                except Exception:
                    pass
                if style in ("line", "line+scatter"):
                    self.ax.plot(x, y, linewidth=0.8, alpha=0.5, color=SMU_COLORS.get(self.smu_name))
            elif style == "line":
                self.ax.plot(x, y, color=SMU_COLORS.get(self.smu_name))
            elif style == "line+scatter":
                self.ax.plot(x, y, marker="o", markersize=3, color=SMU_COLORS.get(self.smu_name))
            else:
                self.ax.scatter(x, y, s=18, color=SMU_COLORS.get(self.smu_name))
            apply_auto_limits(self.ax, x, y)
        self.canvas.draw_idle()

    def _reset_axes(self) -> None:
        if self.fig is None:
            return
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)


class TemperaturePlotWindow:
    def __init__(self, app: "MeasurementApp"):
        self.app = app
        self.window: Optional[tk.Toplevel] = None
        self.fig: Optional[Figure] = None
        self.ax: Any = None
        self.canvas: Any = None
        self.current_temp_var = tk.StringVar(value="T avg 5 s: -- K")
        self.open()

    def open(self) -> None:
        self.app.temperature_history.clear()
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Temperature")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        ctl = ttk.Frame(self.window)
        ctl.pack(fill=tk.X, padx=8, pady=5)
        ttk.Label(ctl, textvariable=self.current_temp_var, font=("TkDefaultFont", 13, "bold")).pack(
            side=tk.LEFT, padx=(4, 16)
        )
        ttk.Button(ctl, text="Reset", command=self.reset).pack(side=tk.RIGHT)
        self.fig = Figure(figsize=(6.0, 3.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Temperature [K]")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        fit_window_to_content(self.window, min_w=620, min_h=390, max_w=760, max_h=520, x=40, y=520)

    def on_close(self) -> None:
        self.reset()
        if self.window and self.window.winfo_exists():
            self.window.destroy()
        self.app.root.after(10000, self.open)

    def reset(self) -> None:
        self.app.temperature_start = time.time()
        self.app.temperature_history.clear()
        self.redraw()

    def redraw(self) -> None:
        if not self.window or not self.window.winfo_exists() or self.ax is None:
            return
        self.ax.cla()
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Temperature [K]")
        if self.app.temperature_history:
            xs = [p[0] for p in self.app.temperature_history]
            ys = [p[1] for p in self.app.temperature_history]
            self.ax.plot(xs, ys, marker="o", markersize=2, linewidth=0.8)
            apply_auto_limits(self.ax, xs, ys)
        self._update_current_temperature_label()
        self.canvas.draw_idle()

    def _update_current_temperature_label(self) -> None:
        history = self.app.temperature_history
        if not history:
            self.current_temp_var.set("T avg 5 s: -- K")
            return
        latest_t = history[-1][0]
        values = [temp for elapsed, temp in history if latest_t - elapsed <= 5.0 and is_number(temp)]
        if not values:
            self.current_temp_var.set("T avg 5 s: -- K")
            return
        avg = sum(float(v) for v in values) / len(values)
        self.current_temp_var.set(f"T avg 5 s: {avg:.3f} K")


class MeasurementCoordinator:
    def __init__(self, app: "MeasurementApp"):
        self.app = app
        self.active_run: Optional[MeasurementRun] = None
        self.lock = threading.Lock()
        self.start_widgets: List[Tuple[tk.Button, Callable[[], Tuple[bool, str]]]] = []

    def register_start(self, button: tk.Button, validator: Callable[[], Tuple[bool, str]]) -> None:
        self.start_widgets.append((button, validator))
        self.refresh_buttons()

    def unregister_start(self, button: tk.Button) -> None:
        self.start_widgets = [(b, v) for b, v in self.start_widgets if b is not button]

    def refresh_buttons(self) -> None:
        running = self.active_run is not None
        for button, validator in list(self.start_widgets):
            try:
                if not button.winfo_exists():
                    continue
                if running:
                    button.configure(state=tk.DISABLED, text="Measurement Running")
                else:
                    ok, text = validator()
                    button.configure(state=(tk.NORMAL if ok else tk.DISABLED), text=text)
            except Exception:
                continue

    def start(self, run: MeasurementRun, worker: Callable[[MeasurementRun], None]) -> bool:
        with self.lock:
            if self.active_run is not None:
                return False
            self.active_run = run
        self.refresh_buttons()
        thread = threading.Thread(target=self._worker_wrapper, args=(run, worker), daemon=True)
        thread.start()
        return True

    def _worker_wrapper(self, run: MeasurementRun, worker: Callable[[MeasurementRun], None]) -> None:
        try:
            self.app.apply_snapshot_to_hardware(run.snapshot)
            worker(run)
        except Exception:
            run.error = traceback.format_exc()
        finally:
            self.app.hardware.output_off_all()
            try:
                run.save()
            except Exception:
                if run.error:
                    run.error += "\n\nSave failed:\n" + traceback.format_exc()
                else:
                    run.error = "Save failed:\n" + traceback.format_exc()
            with self.lock:
                self.active_run = None
            self.app.root.after(0, lambda r=run: self._finish_on_ui(r))

    def _finish_on_ui(self, run: MeasurementRun) -> None:
        self.refresh_buttons()
        if run.live_plot:
            run.live_plot.redraw()
        if run.error:
            messagebox.showerror(run.name, run.error)
        elif run.temperature_guard_triggered and run.temperature_guard_message:
            messagebox.showinfo(run.name, run.temperature_guard_message)
        else:
            messagebox.showinfo(run.name, "Measurement Finished")

    def save_now(self) -> None:
        if self.active_run is not None:
            self.active_run.request_save_now()
        else:
            self.app.save_session_snapshot()


class MeasurementWindowBase:
    title = "Measurement"
    state_key = "base"

    def __init__(self, app: "MeasurementApp"):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title(self.title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.body = ttk.Frame(self.window)
        self.body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.status_var = tk.StringVar(value="Ready")
        footer = ttk.Frame(self.window)
        footer.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.start_button = tk.Button(footer, text="Start Measurement")
        self.save_button = tk.Button(footer, text="Save Now", command=self.save_now)
        ttk.Label(footer, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_button.pack(side=tk.RIGHT, padx=4)
        self.start_button.pack(side=tk.RIGHT, padx=4)
        self.start_button.configure(command=self.on_start)
        self.app.coordinator.register_start(self.start_button, self.start_validator)
        self.build()
        fit_window_to_content(self.window, min_w=420, min_h=220, max_w=980, max_h=760, x=420, y=80)
        self.window.after(1000, self._refresh_loop)

    def build(self) -> None:
        raise NotImplementedError

    def collect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def worker(self, run: MeasurementRun) -> None:
        raise NotImplementedError

    def start_validator(self) -> Tuple[bool, str]:
        return True, "Start Measurement"

    def on_start(self) -> None:
        try:
            settings = self.collect()
            self.app.state.set_measurement(self.state_key, settings)
            snapshot = self.app.settings_snapshot()
            run = MeasurementRun(self.app, self.title, snapshot, settings, self.save_kind())
            if not self.app.coordinator.start(run, self.worker):
                messagebox.showinfo(self.title, "Another measurement is already running.")
        except Exception as exc:
            messagebox.showerror(self.title, str(exc))

    def save_kind(self) -> str:
        return "standard"

    def update_local_estimate(self) -> None:
        pass

    def save_now(self) -> None:
        self.app.coordinator.save_now()

    def set_status(self, text: str) -> None:
        self.app.root.after(0, lambda: self.status_var.set(text))

    def on_close(self) -> None:
        self.app.coordinator.unregister_start(self.start_button)
        self.window.destroy()

    def _refresh_loop(self) -> None:
        self.update_local_estimate()
        self.app.coordinator.refresh_buttons()
        run = self.app.coordinator.active_run
        if run and run.name == self.title and run.temperature_guard_triggered and run.temperature_guard_message:
            self.status_var.set(run.temperature_guard_message)
        if self.window.winfo_exists():
            self.window.after(1000, self._refresh_loop)


class SegmentEditor(ttk.LabelFrame):
    def __init__(self, master: tk.Widget, title: str, unit_getter: Callable[[], str], values: List[Dict[str, Any]]):
        super().__init__(master, text=title)
        self.unit_getter = unit_getter
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Button(btns, text="Add", command=lambda: self.add_row({"min": "0", "max": "100", "steps": "51"})).pack(
            side=tk.LEFT
        )
        ttk.Button(btns, text="Remove", command=self.remove_row).pack(side=tk.LEFT, padx=4)
        self.rows_frame = ttk.Frame(self)
        self.rows_frame.pack(fill=tk.X, padx=4, pady=4)
        self.rows: List[Tuple[ttk.Frame, ttk.Entry, ttk.Entry, ttk.Entry]] = []
        for row in values or [{"min": "0", "max": "100", "steps": "51"}]:
            self.add_row(row)

    def add_row(self, data: Dict[str, Any]) -> None:
        idx = len(self.rows)
        frame = ttk.Frame(self.rows_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=f"Segment {idx + 1}").grid(row=0, column=0, padx=3)
        ttk.Label(frame, text=f"Min [{self.unit_getter()}]").grid(row=0, column=1, padx=3)
        e_min = ttk.Entry(frame, width=10)
        e_min.insert(0, str(data.get("min", "0")))
        e_min.grid(row=0, column=2, padx=3)
        ttk.Label(frame, text=f"Max [{self.unit_getter()}]").grid(row=0, column=3, padx=3)
        e_max = ttk.Entry(frame, width=10)
        e_max.insert(0, str(data.get("max", "100")))
        e_max.grid(row=0, column=4, padx=3)
        ttk.Label(frame, text="Steps").grid(row=0, column=5, padx=3)
        e_steps = ttk.Entry(frame, width=8)
        e_steps.insert(0, str(data.get("steps", "51")))
        e_steps.grid(row=0, column=6, padx=3)
        self.rows.append((frame, e_min, e_max, e_steps))
        self.after_idle(lambda: refit_parent_window(self))

    def remove_row(self) -> None:
        if len(self.rows) <= 1:
            return
        frame, *_ = self.rows.pop()
        frame.destroy()
        self.after_idle(lambda: refit_parent_window(self))

    def get_segments(self) -> List[Dict[str, Any]]:
        out = []
        for _, e_min, e_max, e_steps in self.rows:
            out.append({"min": e_min.get(), "max": e_max.get(), "steps": e_steps.get()})
        return out

    def set_title_units(self) -> None:
        for frame, *_ in self.rows:
            labels = [w for w in frame.winfo_children() if isinstance(w, ttk.Label)]
            if len(labels) >= 3:
                labels[1].configure(text=f"Min [{self.unit_getter()}]")
                labels[2].configure(text=f"Max [{self.unit_getter()}]")


class SimpleDCWindow(MeasurementWindowBase):
    title = "Simple DC"
    state_key = "simple_dc"

    def build(self) -> None:
        defaults = self.app.state.get_measurement(
            self.state_key,
            {
                "settle_ms": "20",
                "segments_smu1": [{"min": "0", "max": "100", "steps": "51"}],
                "segments_smu2": [{"min": "0", "max": "100", "steps": "51"}],
            },
        )
        self.settle_entry = self._entry("Settle [ms]", defaults["settle_ms"], 0)
        self.seg1 = SegmentEditor(self.body, "SMU1 Sweep", lambda: self._unit("SMU1"), defaults["segments_smu1"])
        self.seg1.grid(row=1, column=0, sticky="ew", pady=8)
        self.seg2 = SegmentEditor(self.body, "SMU2 Sweep", lambda: self._unit("SMU2"), defaults["segments_smu2"])
        self.seg2.grid(row=2, column=0, sticky="ew", pady=8)

    def _entry(self, label: str, value: str, row: int) -> ttk.Entry:
        ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w")
        ent = ttk.Entry(self.body, width=12)
        ent.insert(0, str(value))
        ent.grid(row=row, column=1, sticky="w")
        return ent

    def _unit(self, smu_name: str) -> str:
        cfg = self.app.state.general.smu1 if smu_name == "SMU1" else self.app.state.general.smu2
        return source_unit(cfg.source_mode)

    def collect(self) -> Dict[str, Any]:
        return {
            "settle_ms": self.settle_entry.get(),
            "segments_smu1": self.seg1.get_segments(),
            "segments_smu2": self.seg2.get_segments(),
        }

    def worker(self, run: MeasurementRun) -> None:
        run.enable_temperature_guard()
        settings = run.measurement_settings
        settle = max(0.0, float(settings["settle_ms"]) * 1e-3)
        plans = build_segment_plans(run.snapshot, settings)
        points, _ = estimate_sweep_from_plans(run.snapshot, plans, settle)
        run.set_estimate(points, None)
        execute_sweep_plans(self.app, run, plans, settle)


class CyclicDCWindow(MeasurementWindowBase):
    title = "Cyclic DC"
    state_key = "cyclic_dc"

    def build(self) -> None:
        d = self.app.state.get_measurement(
            self.state_key,
            {"max_smu1": "100", "max_smu2": "100", "steps": "50", "cycles": "1", "settle_ms": "20"},
        )
        self.entries: Dict[str, ttk.Entry] = {}
        for row, (key, label) in enumerate(
            [
                ("max_smu1", "SMU1 Max [unit from settings]"),
                ("max_smu2", "SMU2 Max [unit from settings]"),
                ("steps", "Steps per branch"),
                ("cycles", "Cycles (empty or 0 = until Save Now)"),
                ("settle_ms", "Settle [ms]"),
            ]
        ):
            ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ent = ttk.Entry(self.body, width=14)
            ent.insert(0, d[key])
            ent.grid(row=row, column=1, sticky="w")
            self.entries[key] = ent

    def collect(self) -> Dict[str, Any]:
        return {key: ent.get() for key, ent in self.entries.items()}

    def worker(self, run: MeasurementRun) -> None:
        run.enable_temperature_guard()
        s = run.measurement_settings
        steps = int(float(s["steps"]))
        cycles = parse_optional_int(s["cycles"]) or 0
        if cycles <= 0 or cycles > 1:
            run.enable_iv_round_saves()
        settle = max(0.0, float(s["settle_ms"]) * 1e-3)
        one_cycle_plans: Dict[str, List[float]] = {}
        for smu_name, key in (("SMU1", "max_smu1"), ("SMU2", "max_smu2")):
            cfg = run.snapshot["smu"][smu_name]
            if not cfg["enabled"]:
                continue
            max_si = unit_to_si(cfg["source_mode"], float(s[key]))
            one_cycle = cyclic_levels(max_si, steps)
            one_cycle_plans[smu_name] = one_cycle
        jitter_on = bool(run.snapshot["general"].get("jitter"))
        if cycles > 0:
            row_count, _ = estimate_sweep_from_plans(
                run.snapshot,
                {name: levels * cycles for name, levels in one_cycle_plans.items()},
                settle,
            )
            run.set_estimate(row_count, None)
        if run.snapshot["general"]["sequence"] == SEQUENCE_SEQUENTIAL:
            completed = 0
            while not run.stop_event.is_set() and (cycles <= 0 or completed < cycles):
                for smu_name in ("SMU1", "SMU2"):
                    levels = one_cycle_plans.get(smu_name, [])
                    if jitter_on and completed > 0:
                        levels = jitter_nominal_levels(levels)
                    execute_one_smu_plan(self.app, run, smu_name, levels, settle)
                completed += 1
                run.save_iv_round(f"cycle_{completed:04d}")
            return
        if cycles <= 0:
            completed = 0
            while not run.stop_event.is_set():
                plans = one_cycle_plans
                if jitter_on and completed > 0:
                    plans = {name: jitter_nominal_levels(levels) for name, levels in one_cycle_plans.items()}
                execute_sweep_plans(self.app, run, plans, settle, repeat_shorter=True)
                completed += 1
                run.save_iv_round(f"cycle_{completed:04d}")
        else:
            for cycle_idx in range(cycles):
                if run.stop_event.is_set():
                    break
                plans = one_cycle_plans
                if jitter_on and cycle_idx > 0:
                    plans = {name: jitter_nominal_levels(levels) for name, levels in one_cycle_plans.items()}
                execute_sweep_plans(self.app, run, plans, settle)
                run.save_iv_round(f"cycle_{cycle_idx + 1:04d}")


class TemperatureTargetEditor(ttk.LabelFrame):
    def __init__(self, master: tk.Widget, values: List[Dict[str, Any]], with_stable: bool = False):
        super().__init__(master, text="Temperature Targets")
        self.with_stable = with_stable
        btns = ttk.Frame(self)
        btns.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Button(btns, text="Add", command=lambda: self.add_row({})).pack(side=tk.LEFT)
        ttk.Button(btns, text="Remove", command=self.remove_row).pack(side=tk.LEFT, padx=4)
        self.rows_frame = ttk.Frame(self)
        self.rows_frame.pack(fill=tk.X, padx=4, pady=4)
        self.rows: List[Tuple[Any, ...]] = []
        for row in values or [{"target": "4.0", "ramp": "1.0", "pre_regen": "0", "stable_s": "30"}]:
            self.add_row(row)

    def add_row(self, data: Dict[str, Any]) -> None:
        frame = ttk.Frame(self.rows_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text="Target [K]").grid(row=0, column=0, padx=3)
        e_target = ttk.Entry(frame, width=10)
        e_target.insert(0, str(data.get("target", "4.0")))
        e_target.grid(row=0, column=1, padx=3)
        ttk.Label(frame, text="Ramp [K/min]").grid(row=0, column=2, padx=3)
        e_ramp = ttk.Entry(frame, width=10)
        e_ramp.insert(0, str(data.get("ramp", "1.0")))
        e_ramp.grid(row=0, column=3, padx=3)
        regen = tk.IntVar(value=int(str(data.get("pre_regen", "0")) or "0"))
        ttk.Checkbutton(frame, text="Pre Regenerate", variable=regen).grid(row=0, column=4, padx=3)
        if self.with_stable:
            ttk.Label(frame, text="Stable [s]").grid(row=0, column=5, padx=3)
            e_stable = ttk.Entry(frame, width=8)
            e_stable.insert(0, str(data.get("stable_s", "30")))
            e_stable.grid(row=0, column=6, padx=3)
            self.rows.append((frame, e_target, e_ramp, regen, e_stable))
        else:
            self.rows.append((frame, e_target, e_ramp, regen))
        self.after_idle(lambda: refit_parent_window(self))

    def remove_row(self) -> None:
        if len(self.rows) <= 1:
            return
        row = self.rows.pop()
        row[0].destroy()
        self.after_idle(lambda: refit_parent_window(self))

    def get_targets(self) -> List[Dict[str, Any]]:
        out = []
        for row in self.rows:
            if self.with_stable:
                _, e_target, e_ramp, regen, e_stable = row
                out.append(
                    {
                        "target": e_target.get(),
                        "ramp": e_ramp.get(),
                        "pre_regen": str(regen.get()),
                        "stable_s": e_stable.get(),
                    }
                )
            else:
                _, e_target, e_ramp, regen = row
                out.append({"target": e_target.get(), "ramp": e_ramp.get(), "pre_regen": str(regen.get())})
        return out


class TcMeasurementWindow(MeasurementWindowBase):
    title = "Tc Measurement"
    state_key = "tc_measurement"

    def build(self) -> None:
        d = self.app.state.get_measurement(
            self.state_key,
            {"current_smu1": "100", "current_smu2": "100", "sample_ms": "200", "targets": []},
        )
        self.entries: Dict[str, ttk.Entry] = {}
        for row, (key, label) in enumerate(
            [("current_smu1", "SMU1 Current [uA]"), ("current_smu2", "SMU2 Current [uA]"), ("sample_ms", "Sample [ms]")]
        ):
            ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w")
            ent = ttk.Entry(self.body, width=12)
            ent.insert(0, d[key])
            ent.grid(row=row, column=1, sticky="w")
            self.entries[key] = ent
        self.targets = TemperatureTargetEditor(self.body, d["targets"], with_stable=False)
        self.targets.grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(self.body, text="Skip Current Temperature", command=self.skip_temperature).grid(
            row=4, column=0, sticky="w"
        )

    def start_validator(self) -> Tuple[bool, str]:
        bad = []
        cfg = self.app.state.general
        if cfg.smu1.enabled and cfg.smu1.source_mode != SOURCE_CURRENT:
            bad.append("SMU1 Voltage")
        if cfg.smu2.enabled and cfg.smu2.source_mode != SOURCE_CURRENT:
            bad.append("SMU2 Voltage")
        if bad:
            return False, "&".join(bad)
        return True, "Start Measurement"

    def skip_temperature(self) -> None:
        run = self.app.coordinator.active_run
        if run:
            run.skip_temperature_event.set()

    def collect(self) -> Dict[str, Any]:
        return {
            "current_smu1": self.entries["current_smu1"].get(),
            "current_smu2": self.entries["current_smu2"].get(),
            "sample_ms": self.entries["sample_ms"].get(),
            "targets": self.targets.get_targets(),
        }

    def worker(self, run: MeasurementRun) -> None:
        s = run.measurement_settings
        sample_s = max(0.05, float(s["sample_ms"]) * 1e-3)
        currents = {"SMU1": float(s["current_smu1"]) * 1e-6, "SMU2": float(s["current_smu2"]) * 1e-6}
        active = active_smu_names(run.snapshot)
        for smu_name in active:
            dev = self.app.hardware.get_smu(smu_name)
            if dev:
                dev.apply_source(SOURCE_CURRENT, currents[smu_name])
        for target in parse_targets(s["targets"]):
            if run.stop_event.is_set():
                break

            def measure_once(temp: float) -> None:
                for smu_name in active:
                    if run.stop_event.is_set():
                        return
                    dev = self.app.hardware.get_smu(smu_name)
                    if not dev:
                        continue
                    voltage, current = dev.measure(SOURCE_CURRENT, currents[smu_name])
                    run.add_point(smu_name, voltage, current, temp)
                time.sleep(sample_s)

            def measure_during_fast_cooldown(temp: float) -> bool:
                if run.skip_temperature_event.is_set():
                    return False
                measure_once(temp)
                return not run.stop_event.is_set()

            self.app.cryo.start_target(
                target["target"],
                target["ramp"],
                target["pre_regen"],
                run.snapshot["general"]["fast_cooldown"],
                run.stop_event,
                self.set_status,
                measure_during_fast_cooldown,
            )

            self.app.cryo.wait_for_target(
                target["target"], 0.05, run.stop_event, run.skip_temperature_event, self.set_status, measure_once, 0.01
            )


class IVTempContinuousWindow(MeasurementWindowBase):
    title = "IVTemp continuous"
    state_key = "ivtemp_continuous"

    def build(self) -> None:
        d = self.app.state.get_measurement(
            self.state_key,
            {
                "settle_ms": "20",
                "targets": [],
                "segments_smu1": [{"min": "0", "max": "100", "steps": "51"}],
                "segments_smu2": [{"min": "0", "max": "100", "steps": "51"}],
            },
        )
        ttk.Label(self.body, text="Settle [ms]").grid(row=0, column=0, sticky="w")
        self.settle_entry = ttk.Entry(self.body, width=12)
        self.settle_entry.insert(0, d["settle_ms"])
        self.settle_entry.grid(row=0, column=1, sticky="w")
        self.targets = TemperatureTargetEditor(self.body, d["targets"], with_stable=False)
        self.targets.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        self.seg1 = SegmentEditor(self.body, "SMU1 IV", lambda: source_unit(self.app.state.general.smu1.source_mode), d["segments_smu1"])
        self.seg1.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        self.seg2 = SegmentEditor(self.body, "SMU2 IV", lambda: source_unit(self.app.state.general.smu2.source_mode), d["segments_smu2"])
        self.seg2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(self.body, text="Skip Current Temperature", command=self.skip_temperature).grid(row=4, column=0, sticky="w")

    def skip_temperature(self) -> None:
        run = self.app.coordinator.active_run
        if run:
            run.skip_temperature_event.set()

    def collect(self) -> Dict[str, Any]:
        return {
            "settle_ms": self.settle_entry.get(),
            "targets": self.targets.get_targets(),
            "segments_smu1": self.seg1.get_segments(),
            "segments_smu2": self.seg2.get_segments(),
        }

    def worker(self, run: MeasurementRun) -> None:
        run.enable_iv_round_saves()
        s = run.measurement_settings
        settle = max(0.0, float(s["settle_ms"]) * 1e-3)
        plans = build_segment_plans(run.snapshot, s)
        for target in parse_targets(s["targets"]):
            if run.stop_event.is_set():
                break

            def measure_during_fast_cooldown(temp: float) -> bool:
                if run.skip_temperature_event.is_set():
                    return False
                execute_sweep_plans(self.app, run, plans, settle, repeat_shorter=True)
                run.save_iv_round(f"fastcool_T{temp:.3f}K")
                return not run.stop_event.is_set()

            self.app.cryo.start_target(
                target["target"], target["ramp"], target["pre_regen"], run.snapshot["general"]["fast_cooldown"],
                run.stop_event, self.set_status, measure_during_fast_cooldown
            )
            while not run.stop_event.is_set():
                temp = self.app.hardware.read_temperature()
                if run.skip_temperature_event.is_set():
                    run.skip_temperature_event.clear()
                    break
                if abs(temp - target["target"]) <= 0.05:
                    break
                execute_sweep_plans(self.app, run, plans, settle, repeat_shorter=True)
                run.save_iv_round(f"T{temp:.3f}K_to_{target['target']:.3f}K")
                self.set_status(f"T={temp:.3f} K -> {target['target']:.3f} K")


class IVTempAtTemperaturesWindow(IVTempContinuousWindow):
    title = "IVTemp at Temperatures"
    state_key = "ivtemp_at_temperatures"

    def build(self) -> None:
        d = self.app.state.get_measurement(
            self.state_key,
            {
                "settle_ms": "20",
                "targets": [],
                "segments_smu1": [{"min": "0", "max": "100", "steps": "51"}],
                "segments_smu2": [{"min": "0", "max": "100", "steps": "51"}],
            },
        )
        ttk.Label(self.body, text="Settle [ms]").grid(row=0, column=0, sticky="w")
        self.settle_entry = ttk.Entry(self.body, width=12)
        self.settle_entry.insert(0, d["settle_ms"])
        self.settle_entry.grid(row=0, column=1, sticky="w")
        self.targets = TemperatureTargetEditor(self.body, d["targets"], with_stable=True)
        self.targets.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        self.seg1 = SegmentEditor(self.body, "SMU1 IV", lambda: source_unit(self.app.state.general.smu1.source_mode), d["segments_smu1"])
        self.seg1.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        self.seg2 = SegmentEditor(self.body, "SMU2 IV", lambda: source_unit(self.app.state.general.smu2.source_mode), d["segments_smu2"])
        self.seg2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(self.body, text="Skip Current Temperature", command=self.skip_temperature).grid(row=4, column=0, sticky="w")

    def worker(self, run: MeasurementRun) -> None:
        run.enable_iv_round_saves()
        s = run.measurement_settings
        settle = max(0.0, float(s["settle_ms"]) * 1e-3)
        plans = build_segment_plans(run.snapshot, s)
        for target in parse_targets(s["targets"]):
            if run.stop_event.is_set():
                break
            if not run.check_temperature_guard_setpoint(target["target"]):
                break
            self.app.cryo.start_target(
                target["target"], target["ramp"], target["pre_regen"], run.snapshot["general"]["fast_cooldown"],
                run.stop_event, self.set_status
            )
            self.app.cryo.wait_for_target(
                target["target"],
                0.05,
                run.stop_event,
                run.skip_temperature_event,
                self.set_status,
                None,
                0.25,
            )
            stable_until = time.time() + max(0.0, target.get("stable_s", 0.0))
            while time.time() < stable_until and not run.stop_event.is_set():
                self.set_status(f"Stabilizing at {target['target']:.3f} K")
                time.sleep(0.25)
            execute_sweep_plans(self.app, run, plans, settle)
            run.save_iv_round(f"T{target['target']:.3f}K")


class FindJJWindow(MeasurementWindowBase):
    title = "Find JJ"
    state_key = "find_jj"

    def build(self) -> None:
        d = self.app.state.get_measurement(
            self.state_key,
            {
                "v_threshold_uV": "100",
                "v_retrap_uV": "50",
                "i_start_uA": "0.05",
                "i_max_uA": "1000",
                "grow": "1.3",
                "min_step_uA": "0.02",
                "settle_ms": "2",
                "pretty_points": "180",
                "pretty_overdrive": "1.05",
                "cycles": "1",
                "branch": "Positive and Negative",
                "manual_range_after_first_find": "0",
            },
        )
        self.entries: Dict[str, ttk.Entry] = {}
        rows = [
            ("v_threshold_uV", "Threshold [uV]"),
            ("v_retrap_uV", "Retrap [uV]"),
            ("i_start_uA", "I start [uA]"),
            ("i_max_uA", "I max [uA]"),
            ("grow", "Grow factor"),
            ("min_step_uA", "Min step [uA]"),
            ("settle_ms", "Settle [ms]"),
            ("pretty_points", "Fine sweep points"),
            ("pretty_overdrive", "Fine overdrive x Ic"),
            ("cycles", "Cycles"),
        ]
        for row, (key, label) in enumerate(rows):
            ttk.Label(self.body, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ent = ttk.Entry(self.body, width=12)
            ent.insert(0, d[key])
            ent.grid(row=row, column=1, sticky="w")
            self.entries[key] = ent
        ttk.Label(self.body, text="Measure Branch").grid(row=len(rows), column=0, sticky="w")
        self.branch_var = tk.StringVar(value=d["branch"])
        ttk.Combobox(
            self.body,
            textvariable=self.branch_var,
            values=["Positive", "Negative", "Positive and Negative"],
            state="readonly",
            width=24,
        ).grid(row=len(rows), column=1, sticky="w")
        self.manual_range_after_first_find = tk.IntVar(value=int(str(d.get("manual_range_after_first_find", "0")) or "0"))
        ttk.Checkbutton(
            self.body,
            text="Set manual current range after first Find-JJ sweep",
            variable=self.manual_range_after_first_find,
        ).grid(row=len(rows) + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def start_validator(self) -> Tuple[bool, str]:
        bad = []
        cfg = self.app.state.general
        if cfg.smu1.enabled and cfg.smu1.source_mode != SOURCE_CURRENT:
            bad.append("SMU1 Voltage")
        if cfg.smu2.enabled and cfg.smu2.source_mode != SOURCE_CURRENT:
            bad.append("SMU2 Voltage")
        if bad:
            return False, "&".join(bad)
        return True, "Start Measurement"

    def collect(self) -> Dict[str, Any]:
        out = {key: ent.get() for key, ent in self.entries.items()}
        out["branch"] = self.branch_var.get()
        out["manual_range_after_first_find"] = str(self.manual_range_after_first_find.get())
        return out

    def worker(self, run: MeasurementRun) -> None:
        params = jj_params(run.measurement_settings)
        for smu_name in active_smu_names(run.snapshot):
            if run.stop_event.is_set():
                break
            dev = self.app.hardware.get_smu(smu_name)
            if dev:
                self.set_status(f"{smu_name}: Find JJ")
                find_jj_full(run, dev, smu_name, params)


class TempJJContinuousWindow(FindJJWindow):
    title = "TempJJ continuous"
    state_key = "tempjj_continuous"

    def build(self) -> None:
        super().build()
        d = self.app.state.get_measurement(self.state_key, {})
        self.targets = TemperatureTargetEditor(self.body, d.get("targets", []), with_stable=False)
        self.targets.grid(row=12, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Label(self.body, text="Fine window [% Ic]").grid(row=13, column=0, sticky="w")
        self.fine_window = ttk.Entry(self.body, width=12)
        self.fine_window.insert(0, d.get("fine_window_pct", "20"))
        self.fine_window.grid(row=13, column=1, sticky="w")
        ttk.Button(self.body, text="Skip Current Temperature", command=self.skip_temperature).grid(row=14, column=0, sticky="w")

    def skip_temperature(self) -> None:
        run = self.app.coordinator.active_run
        if run:
            run.skip_temperature_event.set()

    def collect(self) -> Dict[str, Any]:
        out = super().collect()
        out["targets"] = self.targets.get_targets()
        out["fine_window_pct"] = self.fine_window.get()
        return out

    def worker(self, run: MeasurementRun) -> None:
        params = jj_params(run.measurement_settings)
        fine_window = max(0.01, float(run.measurement_settings["fine_window_pct"]) / 100.0)
        previous: Dict[str, Dict[str, Optional[float]]] = {}
        for smu_name in active_smu_names(run.snapshot):
            dev = self.app.hardware.get_smu(smu_name)
            if dev:
                previous[smu_name] = find_jj_full(run, dev, smu_name, params)
        for target in parse_targets(run.measurement_settings["targets"]):
            if run.stop_event.is_set():
                break

            def measure_during_fast_cooldown(temp: float) -> bool:
                if run.skip_temperature_event.is_set():
                    return False
                for smu_name in active_smu_names(run.snapshot):
                    dev = self.app.hardware.get_smu(smu_name)
                    if not dev:
                        continue
                    crossed, new_markers = find_jj_fine(run, dev, smu_name, params, previous.get(smu_name, {}), fine_window)
                    if not crossed:
                        new_markers = find_jj_full(run, dev, smu_name, params)
                    previous[smu_name] = new_markers
                return not run.stop_event.is_set()

            self.app.cryo.start_target(
                target["target"], target["ramp"], target["pre_regen"], run.snapshot["general"]["fast_cooldown"],
                run.stop_event, self.set_status, measure_during_fast_cooldown
            )
            while not run.stop_event.is_set():
                temp = self.app.hardware.read_temperature()
                if run.skip_temperature_event.is_set():
                    run.skip_temperature_event.clear()
                    break
                if abs(temp - target["target"]) <= 0.05:
                    break
                for smu_name in active_smu_names(run.snapshot):
                    dev = self.app.hardware.get_smu(smu_name)
                    if not dev:
                        continue
                    crossed, new_markers = find_jj_fine(run, dev, smu_name, params, previous.get(smu_name, {}), fine_window)
                    if not crossed:
                        new_markers = find_jj_full(run, dev, smu_name, params)
                    previous[smu_name] = new_markers


class TempJJAtTemperaturesWindow(TempJJContinuousWindow):
    title = "TempJJ at Temperatures"
    state_key = "tempjj_at_temperatures"

    def worker(self, run: MeasurementRun) -> None:
        params = jj_params(run.measurement_settings)
        for target in parse_targets(run.measurement_settings["targets"]):
            if run.stop_event.is_set():
                break
            if not run.check_temperature_guard_setpoint(target["target"]):
                break
            self.app.cryo.start_target(
                target["target"], target["ramp"], target["pre_regen"], run.snapshot["general"]["fast_cooldown"],
                run.stop_event, self.set_status
            )
            self.app.cryo.wait_for_target(
                target["target"],
                0.05,
                run.stop_event,
                run.skip_temperature_event,
                self.set_status,
                None,
                0.25,
            )
            for smu_name in active_smu_names(run.snapshot):
                dev = self.app.hardware.get_smu(smu_name)
                if dev and not run.stop_event.is_set():
                    find_jj_full(run, dev, smu_name, params)


class FraunhoferWindow(FindJJWindow):
    title = "Fraunhofer Pattern"
    state_key = "fraunhofer"

    def build(self) -> None:
        super().build()
        d = self.app.state.get_measurement(
            self.state_key,
            {
                "coil_min_mA": "-10",
                "coil_max_mA": "10",
                "coil_steps": "41",
                "coil_settle_s": "0.1",
                "field_per_A": "1.0",
                "fine_window_pct": "20",
            },
        )
        self.extra_entries: Dict[str, ttk.Entry] = {}
        for idx, (key, label) in enumerate(
            [
                ("coil_min_mA", "Coil min [mA]"),
                ("coil_max_mA", "Coil max [mA]"),
                ("coil_steps", "Coil steps"),
                ("coil_settle_s", "Coil settle [s]"),
                ("field_per_A", "Conversion field/A"),
                ("fine_window_pct", "Fine window [% Ic]"),
            ],
            start=12,
        ):
            ttk.Label(self.body, text=label).grid(row=idx, column=0, sticky="w", pady=2)
            ent = ttk.Entry(self.body, width=12)
            ent.insert(0, d[key])
            ent.grid(row=idx, column=1, sticky="w")
            self.extra_entries[key] = ent

    def save_kind(self) -> str:
        return "fraunhofer"

    def collect(self) -> Dict[str, Any]:
        out = super().collect()
        for key, ent in self.extra_entries.items():
            out[key] = ent.get()
        return out

    def worker(self, run: MeasurementRun) -> None:
        if not self.app.hardware.smu1 or not self.app.hardware.smu2:
            raise RuntimeError("Fraunhofer needs SMU1 and SMU2")
        params = jj_params(run.measurement_settings)
        cmin = float(run.measurement_settings["coil_min_mA"]) * 1e-3
        cmax = float(run.measurement_settings["coil_max_mA"]) * 1e-3
        n = int(float(run.measurement_settings["coil_steps"]))
        coil_points = center_out(np.linspace(cmin, cmax, max(1, n)))
        settle = max(0.0, float(run.measurement_settings["coil_settle_s"]))
        conversion = float(run.measurement_settings["field_per_A"])
        fine_window = max(0.01, float(run.measurement_settings["fine_window_pct"]) / 100.0)
        smu1, smu2 = self.app.hardware.smu1, self.app.hardware.smu2
        previous: Dict[str, Optional[float]] = {}
        for idx, coil_A in enumerate(coil_points):
            if run.stop_event.is_set():
                break
            self.set_status(f"Coil {idx + 1}/{len(coil_points)}: {coil_A * 1e3:.4g} mA")
            smu2.apply_source(SOURCE_CURRENT, coil_A)
            time.sleep(settle)
            field = coil_A * conversion
            if idx == 0 or not previous:
                previous = find_jj_full(run, smu1, "SMU1", params, coil_device=smu2, magnetic_field=field)
            else:
                crossed, previous_new = find_jj_fine(
                    run, smu1, "SMU1", params, previous, fine_window, coil_device=smu2, magnetic_field=field
                )
                previous = previous_new if crossed else find_jj_full(
                    run, smu1, "SMU1", params, coil_device=smu2, magnetic_field=field
                )


def parse_targets(raw_targets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in raw_targets:
        target = float(row["target"])
        ramp = float(row.get("ramp", "1") or "1")
        pre_regen = bool(int(str(row.get("pre_regen", "0") or "0")))
        stable_s = float(row.get("stable_s", "0") or "0")
        out.append({"target": target, "ramp": ramp, "pre_regen": pre_regen, "stable_s": stable_s})
    if not out:
        raise ValueError("At least one temperature target is required")
    return out


def active_smu_names(snapshot: Dict[str, Any]) -> List[str]:
    return [name for name in ("SMU1", "SMU2") if snapshot["smu"][name]["enabled"]]


def build_segment_plans(snapshot: Dict[str, Any], settings: Dict[str, Any]) -> Dict[str, List[float]]:
    plans: Dict[str, List[float]] = {}
    for smu_name, key in (("SMU1", "segments_smu1"), ("SMU2", "segments_smu2")):
        cfg = snapshot["smu"][smu_name]
        if not cfg["enabled"]:
            continue
        levels: List[float] = []
        for seg in settings.get(key, []):
            start = unit_to_si(cfg["source_mode"], float(seg["min"]))
            end = unit_to_si(cfg["source_mode"], float(seg["max"]))
            steps = int(float(seg["steps"]))
            levels.extend(linspace_inclusive(start, end, steps))
        plans[smu_name] = levels
    if not plans:
        raise ValueError("No active SMU is enabled in the general settings")
    return plans


def estimate_sweep_from_plans(
    snapshot: Dict[str, Any], plans: Dict[str, List[float]], settle_s: float
) -> Tuple[int, float]:
    row_count = sum(len(values) for values in plans.values())
    if snapshot["general"]["sequence"] == SEQUENCE_SEQUENTIAL:
        setpoint_count = row_count
    else:
        setpoint_count = max((len(values) for values in plans.values()), default=0)
    nplc = snapshot["general"].get("nplc") or 0.0
    seconds = setpoint_count * (max(0.0, settle_s) + max(0.0, float(nplc)) * 0.02)
    return row_count, seconds


def tsp_send(dev: Any, cmd: str) -> bool:
    for method_name in ("write", "send", "sendcmd", "execute", "exec", "write_raw"):
        method = getattr(dev, method_name, None)
        if callable(method):
            try:
                method(cmd)
                return True
            except Exception:
                pass
    for path in ("visa", "instrument", "resource"):
        handle = getattr(dev, path, None)
        if handle is not None and hasattr(handle, "write"):
            try:
                handle.write(cmd)
                return True
            except Exception:
                pass
    return False


def tsp_query(dev: Any, cmd: str) -> Optional[str]:
    for method_name in ("query", "ask"):
        method = getattr(dev, method_name, None)
        if callable(method):
            try:
                return str(method(cmd))
            except Exception:
                pass
    if tsp_send(dev, cmd):
        for method_name in ("read", "read_raw"):
            method = getattr(dev, method_name, None)
            if callable(method):
                try:
                    return str(method())
                except Exception:
                    pass
    for path in ("visa", "instrument", "resource"):
        handle = getattr(dev, path, None)
        if handle is not None and hasattr(handle, "query"):
            try:
                return str(handle.query(cmd))
            except Exception:
                pass
    return None


def parse_tsp_floats(text: Optional[str], expected: int) -> Optional[List[float]]:
    if text is None:
        return None
    tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf", str(text), re.I)
    if len(tokens) < expected:
        return None
    try:
        return [float(tok) for tok in tokens[:expected]]
    except Exception:
        return None


def tsp_source_command(channel: str, source_mode: str, level: float) -> str:
    func = 2 if source_mode == SOURCE_CURRENT else 1
    attr = "leveli" if source_mode == SOURCE_CURRENT else "levelv"
    return f"{channel}.source.output=1; {channel}.source.func={func}; {channel}.source.{attr}={float(level):.17g}"


def try_tsp_parallel_dual_point(
    run: MeasurementRun,
    requests: Sequence[Tuple[str, SMUDevice, str, float]],
    settle_s: float,
) -> bool:
    if not run.check_temperature_guard():
        return True
    pause_count = run.temperature_guard_pause_count
    if len(requests) != 2:
        return False
    (_, dev1, _, _), (_, dev2, _, _) = requests
    if dev1.instrument is not dev2.instrument:
        return False
    if not dev1.tsp_channel or not dev2.tsp_channel:
        return False
    cmd_parts = [tsp_source_command(dev.tsp_channel, mode, level) for _, dev, mode, level in requests]
    if settle_s > 0:
        cmd_parts.append(f"delay({float(settle_s):.9g})")
    measure_parts = []
    for _, dev, _, _ in requests:
        ch = dev.tsp_channel
        measure_parts.extend([f"{ch}.measure.v()", f"{ch}.measure.i()"])
    cmd_parts.append("print(" + ", ".join(measure_parts) + ")")
    response = tsp_query(dev1.instrument, "; ".join(cmd_parts))
    values = parse_tsp_floats(response, 4)
    if values is None:
        return False
    temp = run.app.hardware.read_temperature()
    if not run.check_temperature_guard(temp):
        return True
    if run.temperature_guard_pause_count != pause_count:
        return False
    for idx, (smu_name, _dev, _mode, level) in enumerate(requests):
        voltage = values[2 * idx]
        current = values[2 * idx + 1]
        run.add_point(smu_name, voltage, current, temp, extra={"source_level": level, "parallel_tsp": True})
    return True


def execute_sweep_plans(
    app: "MeasurementApp",
    run: MeasurementRun,
    plans: Dict[str, List[float]],
    settle_s: float,
    repeat_shorter: bool = False,
) -> None:
    sequence = run.snapshot["general"]["sequence"]
    if sequence == SEQUENCE_SEQUENTIAL:
        for smu_name in ("SMU1", "SMU2"):
            execute_one_smu_plan(app, run, smu_name, plans.get(smu_name, []), settle_s)
    else:
        max_len = max((len(v) for v in plans.values()), default=0)
        idx = 0
        while idx < max_len:
            if run.stop_event.is_set():
                return
            if not run.check_temperature_guard():
                return
            pause_count = run.temperature_guard_pause_count
            to_measure: List[Tuple[str, SMUDevice, str, float]] = []
            for smu_name in ("SMU1", "SMU2"):
                levels = plans.get(smu_name, [])
                if not levels:
                    continue
                if idx >= len(levels) and not repeat_shorter:
                    continue
                dev = app.hardware.get_smu(smu_name)
                if not dev:
                    continue
                mode = run.snapshot["smu"][smu_name]["source_mode"]
                level = levels[idx % len(levels)] if repeat_shorter else levels[idx]
                to_measure.append((smu_name, dev, mode, level))
            if try_tsp_parallel_dual_point(run, to_measure, settle_s):
                idx += 1
                continue
            pause_count = run.temperature_guard_pause_count
            for _smu_name, dev, mode, level in to_measure:
                dev.apply_source(mode, level)
            time.sleep(settle_s)
            temp = app.hardware.read_temperature()
            if not run.check_temperature_guard(temp):
                return
            if run.temperature_guard_pause_count != pause_count:
                continue
            for smu_name, dev, mode, level in to_measure:
                voltage, current = dev.measure(mode, level)
                run.add_point(smu_name, voltage, current, temp, extra={"source_level": level})
            idx += 1


def execute_one_smu_plan(app: "MeasurementApp", run: MeasurementRun, smu_name: str, levels: List[float], settle_s: float) -> None:
    if not levels:
        return
    dev = app.hardware.get_smu(smu_name)
    if not dev:
        return
    mode = run.snapshot["smu"][smu_name]["source_mode"]
    idx = 0
    while idx < len(levels):
        level = levels[idx]
        if run.stop_event.is_set():
            return
        if not run.check_temperature_guard():
            return
        pause_count = run.temperature_guard_pause_count
        dev.apply_source(mode, level)
        time.sleep(settle_s)
        temp = app.hardware.read_temperature()
        if not run.check_temperature_guard(temp):
            return
        if run.temperature_guard_pause_count != pause_count:
            continue
        voltage, current = dev.measure(mode, level)
        run.add_point(smu_name, voltage, current, temp, extra={"source_level": level})
        idx += 1


def jj_params(settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "v_threshold": float(settings["v_threshold_uV"]) * 1e-6,
        "v_retrap": float(settings["v_retrap_uV"]) * 1e-6,
        "i_start": float(settings["i_start_uA"]),
        "i_max": float(settings["i_max_uA"]),
        "grow": float(settings["grow"]),
        "min_step": float(settings["min_step_uA"]),
        "settle_s": max(0.0, float(settings["settle_ms"]) * 1e-3),
        "pretty_points": max(20, int(float(settings["pretty_points"]))),
        "pretty_overdrive": float(settings["pretty_overdrive"]),
        "cycles": max(1, int(float(settings["cycles"] or "1"))),
        "branch": settings.get("branch", "Positive and Negative"),
        "manual_range_after_first_find": bool(int(str(settings.get("manual_range_after_first_find", "0") or "0"))),
    }


def soft_ramp_sequence(i_start_uA: float, i_max_uA: float, grow: float, min_step_uA: float) -> List[float]:
    current = max(float(i_start_uA), float(min_step_uA))
    out: List[float] = []
    while current <= float(i_max_uA) * (1 + 1e-9):
        out.append(current * 1e-6)
        current = max(current + float(min_step_uA), current * float(grow))
    return out


def branch_signs(branch: str) -> List[int]:
    if branch == "Positive":
        return [1]
    if branch == "Negative":
        return [-1]
    return [1, -1]


def find_jj_full(
    run: MeasurementRun,
    dev: SMUDevice,
    smu_name: str,
    params: Dict[str, Any],
    coil_device: Optional[SMUDevice] = None,
    magnetic_field: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    markers: Dict[str, Optional[float]] = {"Ic_plus": None, "Ic_minus": None}
    dev.apply_source(SOURCE_CURRENT, 0.0)
    time.sleep(params["settle_s"])
    manual_range_set = False
    for sign in branch_signs(params["branch"]):
        seq = soft_ramp_sequence(params["i_start"], params["i_max"], params["grow"], params["min_step"])
        found: Optional[float] = None
        for level_abs in seq:
            if run.stop_event.is_set():
                return markers
            level = sign * level_abs
            row = jj_apply_measure(run, dev, smu_name, level, params["settle_s"], coil_device, magnetic_field)
            if abs(row["voltage"]) >= params["v_threshold"]:
                found = level
                break
        if sign > 0:
            markers["Ic_plus"] = found
        else:
            markers["Ic_minus"] = found
        if found is not None:
            if params.get("manual_range_after_first_find") and not manual_range_set:
                apply_jj_post_find_manual_ranges(run, dev, smu_name, abs(found), params)
                manual_range_set = True
            down = list(reversed(soft_ramp_sequence(params["i_start"], abs(found) * 1e6, params["grow"], params["min_step"])))
            for level_abs in down:
                if run.stop_event.is_set():
                    return markers
                row = jj_apply_measure(run, dev, smu_name, sign * level_abs, params["settle_s"], coil_device, magnetic_field)
                if abs(row["voltage"]) <= params["v_retrap"]:
                    break
            pretty_jj_sweep(run, dev, smu_name, sign, abs(found), params, coil_device, magnetic_field)
    dev.apply_source(SOURCE_CURRENT, 0.0)
    return markers


def apply_jj_post_find_manual_ranges(
    run: MeasurementRun,
    dev: SMUDevice,
    smu_name: str,
    ic_abs: float,
    params: Dict[str, Any],
) -> None:
    i_range = max(abs(ic_abs) * params["pretty_overdrive"] * 1.25, params["min_step"] * 1e-6, 1e-12)
    v_range = max(abs(params["v_threshold"]) * 5.0, abs(params["v_retrap"]) * 5.0, 1e-6)
    warnings: List[str] = []
    try:
        dev.channel.measure.autorangei = dev.channel.AUTORANGE_OFF
        dev.channel.measure.rangei = float(i_range)
    except Exception as exc:
        warnings.append(f"{smu_name}: post Find-JJ current measure range was not applied ({exc})")
    try:
        dev.channel.source.autorangev = dev.channel.AUTORANGE_OFF
        dev.channel.source.rangev = float(v_range)
    except Exception as exc:
        warnings.append(f"{smu_name}: post Find-JJ voltage source range was not applied ({exc})")
    if warnings:
        run.app.root.after(0, lambda text="\n".join(warnings): messagebox.showerror("Keithley configuration", text))


def pretty_jj_sweep(
    run: MeasurementRun,
    dev: SMUDevice,
    smu_name: str,
    sign: int,
    ic_abs: float,
    params: Dict[str, Any],
    coil_device: Optional[SMUDevice],
    magnetic_field: Optional[float],
) -> None:
    if ic_abs <= 0:
        return
    imax = abs(ic_abs) * params["pretty_overdrive"]
    points = [float(x) for x in np.linspace(0.0, imax, params["pretty_points"])]
    cycle = points + list(reversed(points))
    for _ in range(params["cycles"]):
        for level in cycle:
            if run.stop_event.is_set():
                return
            jj_apply_measure(run, dev, smu_name, sign * level, params["settle_s"], coil_device, magnetic_field)


def find_jj_fine(
    run: MeasurementRun,
    dev: SMUDevice,
    smu_name: str,
    params: Dict[str, Any],
    previous: Dict[str, Optional[float]],
    fine_window: float,
    coil_device: Optional[SMUDevice] = None,
    magnetic_field: Optional[float] = None,
) -> Tuple[bool, Dict[str, Optional[float]]]:
    new_markers = dict(previous)
    crossed = False
    for sign in branch_signs(params["branch"]):
        key = "Ic_plus" if sign > 0 else "Ic_minus"
        ic = previous.get(key)
        if not ic:
            continue
        width = max(abs(ic) * fine_window, params["min_step"] * 1e-6)
        points = np.linspace(ic - width, ic + width, params["pretty_points"])
        best = None
        for level in points:
            if run.stop_event.is_set():
                return crossed, new_markers
            row = jj_apply_measure(run, dev, smu_name, float(level), params["settle_s"], coil_device, magnetic_field)
            if abs(row["voltage"]) >= params["v_threshold"] and best is None:
                best = float(level)
                crossed = True
        if best is not None:
            new_markers[key] = best
    return crossed, new_markers


def jj_apply_measure(
    run: MeasurementRun,
    dev: SMUDevice,
    smu_name: str,
    current_A: float,
    settle_s: float,
    coil_device: Optional[SMUDevice] = None,
    magnetic_field: Optional[float] = None,
) -> Dict[str, Any]:
    guard_row = {
        "time": now_text(),
        "elapsed_s": time.time() - run.start_time,
        "smu": smu_name,
        "voltage": float("nan"),
        "current": float("nan"),
        "temperature": run.app.hardware.read_temperature(),
        "magnetic_field": magnetic_field if magnetic_field is not None else float("nan"),
        "jj_current_set": current_A,
    }
    if not run.check_temperature_guard(guard_row["temperature"]):
        return guard_row
    while not run.stop_event.is_set():
        pause_count = run.temperature_guard_pause_count
        dev.apply_source(SOURCE_CURRENT, current_A)
        time.sleep(settle_s)
        temp = run.app.hardware.read_temperature()
        if not run.check_temperature_guard(temp):
            guard_row["temperature"] = temp
            return guard_row
        if run.temperature_guard_pause_count == pause_count:
            break
    else:
        return guard_row
    voltage, current = dev.measure(SOURCE_CURRENT, current_A)
    if coil_device is not None:
        coil_v, coil_i = coil_device.measure(SOURCE_CURRENT, None)
        field = magnetic_field if magnetic_field is not None else float("nan")
        run.add_fraunhofer_point(voltage, current, coil_v, coil_i, field, temp)
        return {
            "time": now_text(),
            "elapsed_s": time.time() - run.start_time,
            "smu": smu_name,
            "voltage": voltage,
            "current": current,
            "temperature": temp,
            "magnetic_field": field,
            "jj_current_set": current_A,
        }
    return run.add_point(smu_name, voltage, current, temp, magnetic_field, {"jj_current_set": current_A})


class GeneralSettingsWindow:
    def __init__(self, app: "MeasurementApp"):
        self.app = app
        self.root = app.root
        self.root.title("Settings")
        self.root.protocol("WM_DELETE_WINDOW", app.shutdown)
        self.vars: Dict[str, Any] = {}
        self.loading = False
        self.build()
        self.refresh_from_state()
        fit_window_to_content(self.root, min_w=760, min_h=300, max_w=1120, max_h=540, x=40, y=40)

    def build(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        devices = ttk.LabelFrame(main, text="SMU Setup")
        devices.pack(fill=tk.X, pady=6)
        self._smu_row(devices, "SMU1", 0)
        self._smu_row(devices, "SMU2", 1)
        sequence = ttk.Frame(devices)
        sequence.grid(row=2, column=0, columnspan=8, sticky="w", pady=6)
        ttk.Label(sequence, text="SMUs").pack(side=tk.LEFT, padx=(4, 8))
        self.sequence_seg = SegmentedControl(
            sequence,
            [("Parallel", SEQUENCE_PARALLEL), ("Sequential", SEQUENCE_SEQUENTIAL)],
            SEQUENCE_PARALLEL,
            lambda _: self.save_to_state(),
        )
        self.sequence_seg.pack(side=tk.LEFT)
        ttk.Label(sequence, text="Current Limit [A]").pack(side=tk.LEFT, padx=(20, 3))
        self.current_limit = ttk.Entry(sequence, width=12)
        self.current_limit.pack(side=tk.LEFT)
        ttk.Label(sequence, text="Voltage Limit [V]").pack(side=tk.LEFT, padx=(10, 3))
        self.voltage_limit = ttk.Entry(sequence, width=12)
        self.voltage_limit.pack(side=tk.LEFT)
        self.header_seg = SegmentedControl(
            sequence,
            [("Header Yes", "1"), ("Header No", "0")],
            "1",
            lambda _: self.save_to_state(),
        )
        self.header_seg.pack(side=tk.LEFT, padx=14)

        other = ttk.LabelFrame(main, text="Measurement Defaults")
        other.pack(fill=tk.X, pady=6)
        ttk.Label(other, text="NPLC").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.nplc = ttk.Entry(other, width=12)
        self.nplc.grid(row=0, column=1, sticky="w")
        ttk.Label(other, text="Autozero").grid(row=0, column=2, padx=4, sticky="e")
        self.autozero_seg = SegmentedControl(
            other,
            [("Once", "Once"), ("Automatic", "Automatic"), ("Now", "Now")],
            "Once",
            self.autozero_changed,
        )
        self.autozero_seg.grid(row=0, column=3, sticky="w")
        self.fast_cool_seg = SegmentedControl(
            other,
            [("Fast Cool Yes", "1"), ("Fast Cool No", "0")],
            "1",
            lambda _: self.save_to_state(),
        )
        self.fast_cool_seg.grid(row=0, column=4, padx=10, sticky="w")
        self.jitter_seg = SegmentedControl(
            other,
            [("Jitter Off", "0"), ("Jitter On", "1")],
            "0",
            lambda _: self.save_to_state(),
        )
        self.jitter_seg.grid(row=0, column=5, padx=10, sticky="w")
        self.temperature_guard_seg = SegmentedControl(
            other,
            [("Temp Guard Off", "0"), ("Temp Guard On", "1")],
            "0",
            lambda _: self.save_to_state(),
        )
        self.temperature_guard_seg.grid(row=1, column=0, columnspan=2, padx=4, pady=4, sticky="w")
        ttk.Label(other, text="Min [K]").grid(row=1, column=2, padx=(12, 3), sticky="e")
        self.temperature_guard_min = ttk.Entry(other, width=10)
        self.temperature_guard_min.grid(row=1, column=3, sticky="w")
        ttk.Label(other, text="Max [K]").grid(row=1, column=4, padx=(12, 3), sticky="e")
        self.temperature_guard_max = ttk.Entry(other, width=10)
        self.temperature_guard_max.grid(row=1, column=5, sticky="w")
        self.temperature_guard_mode_seg = SegmentedControl(
            other,
            [("Save and stop", TEMP_GUARD_SAVE_STOP), ("Wait and continue", TEMP_GUARD_WAIT_CONTINUE)],
            TEMP_GUARD_SAVE_STOP,
            lambda _: self.save_to_state(),
        )
        self.temperature_guard_mode_seg.grid(row=2, column=0, columnspan=4, padx=4, pady=(0, 4), sticky="w")

        ranges = ttk.LabelFrame(main, text="Autorange and Manual Ranges")
        ranges.pack(fill=tk.X, pady=6)
        ttk.Label(ranges, text="Autorange Voltage").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        self.autorange_voltage_seg = SegmentedControl(
            ranges,
            [("On", "1"), ("Manual", "0")],
            "1",
            lambda _: self.save_to_state(),
        )
        self.autorange_voltage_seg.grid(row=0, column=1, sticky="w")
        ttk.Label(ranges, text="SMU1 [mV]").grid(row=0, column=2, padx=(16, 3), sticky="e")
        self.voltage_range_smu1 = ttk.Entry(ranges, width=10)
        self.voltage_range_smu1.grid(row=0, column=3, sticky="w")
        ttk.Label(ranges, text="SMU2 [mV]").grid(row=0, column=4, padx=(10, 3), sticky="e")
        self.voltage_range_smu2 = ttk.Entry(ranges, width=10)
        self.voltage_range_smu2.grid(row=0, column=5, sticky="w")

        ttk.Label(ranges, text="Autorange Current").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        self.autorange_current_seg = SegmentedControl(
            ranges,
            [("On", "1"), ("Manual", "0")],
            "1",
            lambda _: self.save_to_state(),
        )
        self.autorange_current_seg.grid(row=1, column=1, sticky="w")
        ttk.Label(ranges, text="SMU1 [uA]").grid(row=1, column=2, padx=(16, 3), sticky="e")
        self.current_range_smu1 = ttk.Entry(ranges, width=10)
        self.current_range_smu1.grid(row=1, column=3, sticky="w")
        ttk.Label(ranges, text="SMU2 [uA]").grid(row=1, column=4, padx=(10, 3), sticky="e")
        self.current_range_smu2 = ttk.Entry(ranges, width=10)
        self.current_range_smu2.grid(row=1, column=5, sticky="w")

        paths = ttk.LabelFrame(main, text="Paths and Actions")
        paths.pack(fill=tk.X, pady=6)
        ttk.Button(paths, text="Set Savepath", command=self.choose_save_path).grid(row=0, column=0, padx=4, pady=4)
        self.save_path_label = ttk.Label(paths, text="", width=80)
        self.save_path_label.grid(row=0, column=1, sticky="w")
        ttk.Button(paths, text="Reset Temperature Plot", command=self.app.temperature_plot.reset).grid(
            row=1, column=0, padx=4, pady=4
        )
        ttk.Button(paths, text="Save Now", command=self.app.coordinator.save_now).grid(row=1, column=1, sticky="w")
        ttk.Button(paths, text="Reset To Standard", command=self.reset_standard).grid(row=1, column=2, padx=4)
        ttk.Button(paths, text="Info", command=self.app.open_info_window).grid(row=1, column=3, padx=4)

        for ent in (
            self.current_limit,
            self.voltage_limit,
            self.nplc,
            self.voltage_range_smu1,
            self.voltage_range_smu2,
            self.current_range_smu1,
            self.current_range_smu2,
            self.temperature_guard_min,
            self.temperature_guard_max,
        ):
            ent.bind("<FocusOut>", lambda _e: self.save_to_state())
            ent.bind("<Return>", lambda _e: self.save_to_state())

    def _smu_row(self, parent: tk.Widget, name: str, row: int) -> None:
        ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        on = SegmentedControl(parent, [("On", "1"), ("Off", "0")], "1", lambda _v: self.save_to_state())
        on.grid(row=row, column=1, padx=4)
        sense = SegmentedControl(parent, [("2 Point", "2"), ("4 Point", "4")], "4", lambda _v: self.save_to_state())
        sense.grid(row=row, column=2, padx=4)
        source = SegmentedControl(
            parent, [("Source Voltage mV", SOURCE_VOLTAGE), ("Source Current uA", SOURCE_CURRENT)],
            SOURCE_CURRENT, lambda _v: self.save_to_state()
        )
        source.grid(row=row, column=3, padx=4)
        ttk.Label(parent, text="Filename").grid(row=row, column=4, padx=4)
        filename = ttk.Entry(parent, width=28)
        filename.grid(row=row, column=5, sticky="w", padx=4)
        filename.bind("<FocusOut>", lambda _e: self.save_to_state())
        filename.bind("<Return>", lambda _e: self.save_to_state())
        self.vars[name] = {"on": on, "sense": sense, "source": source, "filename": filename}

    def refresh_from_state(self) -> None:
        self.loading = True
        cfg = self.app.state.general
        for name, smu_cfg in (("SMU1", cfg.smu1), ("SMU2", cfg.smu2)):
            widgets = self.vars[name]
            widgets["on"].set("1" if smu_cfg.enabled else "0")
            widgets["sense"].set("4" if smu_cfg.four_point else "2")
            widgets["source"].set(smu_cfg.source_mode)
            widgets["filename"].delete(0, tk.END)
            widgets["filename"].insert(0, smu_cfg.filename)
        self.sequence_seg.set(cfg.sequence)
        self.header_seg.set("1" if cfg.put_header else "0")
        self.fast_cool_seg.set("1" if cfg.fast_cooldown else "0")
        self.jitter_seg.set("1" if cfg.jitter else "0")
        self.temperature_guard_seg.set("1" if cfg.temperature_guard_enabled else "0")
        self.temperature_guard_mode_seg.set(cfg.temperature_guard_mode)
        self.autorange_voltage_seg.set("1" if cfg.autorange_voltage else "0")
        self.autorange_current_seg.set("1" if cfg.autorange_current else "0")
        self.autozero_seg.set(cfg.autozero)
        self._set_entry(self.current_limit, cfg.current_limit)
        self._set_entry(self.voltage_limit, cfg.voltage_limit)
        self._set_entry(self.nplc, cfg.nplc)
        self._set_entry(self.voltage_range_smu1, cfg.voltage_range_smu1_mV)
        self._set_entry(self.voltage_range_smu2, cfg.voltage_range_smu2_mV)
        self._set_entry(self.current_range_smu1, cfg.current_range_smu1_uA)
        self._set_entry(self.current_range_smu2, cfg.current_range_smu2_uA)
        self._set_entry(self.temperature_guard_min, cfg.temperature_guard_min_K)
        self._set_entry(self.temperature_guard_max, cfg.temperature_guard_max_K)
        self.save_path_label.configure(text=cfg.save_path)
        self.loading = False

    def _set_entry(self, entry: ttk.Entry, value: Optional[float]) -> None:
        entry.delete(0, tk.END)
        if value is not None:
            entry.insert(0, str(value))

    def save_to_state(self) -> None:
        if self.loading:
            return
        cfg = self.app.state.general
        for name, attr in (("SMU1", "smu1"), ("SMU2", "smu2")):
            widgets = self.vars[name]
            smu_cfg = getattr(cfg, attr)
            smu_cfg.enabled = widgets["on"].get() == "1"
            smu_cfg.four_point = widgets["sense"].get() == "4"
            smu_cfg.source_mode = widgets["source"].get()
            smu_cfg.filename = widgets["filename"].get() or name
        cfg.sequence = self.sequence_seg.get()
        cfg.put_header = self.header_seg.get() == "1"
        cfg.fast_cooldown = self.fast_cool_seg.get() == "1"
        cfg.jitter = self.jitter_seg.get() == "1"
        cfg.temperature_guard_enabled = self.temperature_guard_seg.get() == "1"
        cfg.temperature_guard_mode = self.temperature_guard_mode_seg.get()
        cfg.autorange_voltage = self.autorange_voltage_seg.get() == "1"
        cfg.autorange_current = self.autorange_current_seg.get() == "1"
        cfg.autozero = self.autozero_seg.get()
        try:
            cfg.current_limit = parse_optional_float(self.current_limit.get())
            cfg.voltage_limit = parse_optional_float(self.voltage_limit.get())
            cfg.nplc = parse_optional_float(self.nplc.get())
            cfg.voltage_range_smu1_mV = parse_optional_float(self.voltage_range_smu1.get())
            cfg.voltage_range_smu2_mV = parse_optional_float(self.voltage_range_smu2.get())
            cfg.current_range_smu1_uA = parse_optional_float(self.current_range_smu1.get())
            cfg.current_range_smu2_uA = parse_optional_float(self.current_range_smu2.get())
            cfg.temperature_guard_min_K = parse_optional_float(self.temperature_guard_min.get())
            cfg.temperature_guard_max_K = parse_optional_float(self.temperature_guard_max.get())
        except ValueError:
            return
        self.app.state.save()
        self.app.coordinator.refresh_buttons()

    def autozero_changed(self, value: str) -> None:
        if self.loading:
            return
        self.save_to_state()
        if value == "Now":
            self.app.perform_autozero_now()

    def choose_save_path(self) -> None:
        path = filedialog.askdirectory(initialdir=self.app.state.general.save_path or DEFAULT_SAVE_PATH)
        if path:
            self.app.state.general.save_path = path
            self.app.state.save()
            self.save_path_label.configure(text=path)

    def reset_standard(self) -> None:
        self.app.state.reset_general()
        self.refresh_from_state()


class MeasurementLauncherWindow:
    def __init__(self, app: "MeasurementApp"):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Measurements")
        self.window.protocol("WM_DELETE_WINDOW", app.shutdown)
        self.build()
        fit_window_to_content(self.window, min_w=300, min_h=260, max_w=460, max_h=620, x=980, y=40)

    def build(self) -> None:
        sections = [
            ("DC Measurements", [("Simple DC", SimpleDCWindow), ("Cyclic DC", CyclicDCWindow)]),
            (
                "Temperature dependent DC Measurements",
                [
                    ("Tc Measurement", TcMeasurementWindow),
                    ("IVTemp continuous", IVTempContinuousWindow),
                    ("IVTemp at Temperatures", IVTempAtTemperaturesWindow),
                ],
            ),
            (
                "Josephson Junctions",
                [
                    ("Find JJ", FindJJWindow),
                    ("TempJJ continuous", TempJJContinuousWindow),
                    ("TempJJ at Temperatures", TempJJAtTemperaturesWindow),
                    ("Fraunhofer Pattern", FraunhoferWindow),
                ],
            ),
        ]
        for title, buttons in sections:
            frame = ttk.LabelFrame(self.window, text=title)
            frame.pack(fill=tk.X, padx=10, pady=8)
            for label, cls in buttons:
                ttk.Button(frame, text=label, command=lambda c=cls: c(self.app)).pack(fill=tk.X, padx=8, pady=3)


class MeasurementApp:
    def __init__(self):
        self.root = tk.Tk()
        self._shutting_down = False
        self.root.report_callback_exception = self.report_callback_exception
        self.state = PersistentState(SETTINGS_FILE)
        self.hardware = Hardware()
        EmergencyOutputGuard.register(self.hardware)
        self.cryo = CryoController(self.hardware)
        self.saver = TextSaver(self)
        self.event_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.temperature_start = time.time()
        self.temperature_history: List[Tuple[float, float]] = []
        self.session_rows: Dict[str, List[Dict[str, Any]]] = {"SMU1": [], "SMU2": []}
        self.coordinator = MeasurementCoordinator(self)
        self.temperature_plot = TemperaturePlotWindow(self)
        self.session_plots = {
            "SMU1": SessionPlotWindow(self, "SMU1"),
            "SMU2": SessionPlotWindow(self, "SMU2"),
        }
        self.settings_window = GeneralSettingsWindow(self)
        self.launcher_window = MeasurementLauncherWindow(self)
        if self.hardware.errors:
            self.root.after(300, lambda: messagebox.showwarning("Hardware", "\n".join(self.hardware.errors)))
        self.root.after(200, self.poll_events)
        self.root.after(1000, self.poll_temperature)

    def settings_snapshot(self) -> Dict[str, Any]:
        self.settings_window.save_to_state()
        cfg = self.state.general
        return {
            "general": {
                "sequence": cfg.sequence,
                "current_limit": cfg.current_limit,
                "voltage_limit": cfg.voltage_limit,
                "put_header": cfg.put_header,
                "nplc": cfg.nplc,
                "autozero": cfg.autozero,
                "jitter": cfg.jitter,
                "autorange_voltage": cfg.autorange_voltage,
                "voltage_range_smu1_mV": cfg.voltage_range_smu1_mV,
                "voltage_range_smu2_mV": cfg.voltage_range_smu2_mV,
                "autorange_current": cfg.autorange_current,
                "current_range_smu1_uA": cfg.current_range_smu1_uA,
                "current_range_smu2_uA": cfg.current_range_smu2_uA,
                "save_path": cfg.save_path,
                "backup_path": cfg.backup_path,
                "fast_cooldown": cfg.fast_cooldown,
                "temperature_guard_enabled": cfg.temperature_guard_enabled,
                "temperature_guard_mode": cfg.temperature_guard_mode,
                "temperature_guard_min_K": cfg.temperature_guard_min_K,
                "temperature_guard_max_K": cfg.temperature_guard_max_K,
            },
            "smu": {"SMU1": asdict(cfg.smu1), "SMU2": asdict(cfg.smu2)},
        }

    def apply_snapshot_to_hardware(self, snapshot: Dict[str, Any]) -> None:
        cfg = GeneralConfig.from_dict(
            {
                **snapshot["general"],
                "smu1": snapshot["smu"]["SMU1"],
                "smu2": snapshot["smu"]["SMU2"],
            }
        )
        warnings: List[str] = []
        for smu_name in active_smu_names(snapshot):
            dev = self.hardware.get_smu(smu_name)
            if not dev:
                raise RuntimeError(f"{smu_name} is not connected")
            smu_cfg = cfg.smu1 if smu_name == "SMU1" else cfg.smu2
            warnings.extend(dev.configure(smu_cfg, cfg))
            if cfg.autozero in ("Once", "Automatic"):
                dev.set_autozero(cfg.autozero)
        if warnings:
            self.root.after(0, lambda text="\n".join(warnings): messagebox.showerror("Keithley configuration", text))

    def perform_autozero_now(self) -> None:
        cfg = self.state.general
        for name, smu_cfg in (("SMU1", cfg.smu1), ("SMU2", cfg.smu2)):
            if smu_cfg.enabled:
                dev = self.hardware.get_smu(name)
                if dev:
                    dev.set_autozero("Once")

    def poll_events(self) -> None:
        changed = False
        try:
            while True:
                kind, payload = self.event_queue.get_nowait()
                if kind == "point":
                    smu = payload.get("smu")
                    if smu in self.session_rows:
                        self.session_rows[smu].append(payload)
                    run = self.coordinator.active_run
                    if run and run.live_plot:
                        run.live_plot.redraw()
                    changed = True
        except queue.Empty:
            pass
        if changed:
            for plot in self.session_plots.values():
                plot.redraw()
        self.root.after(200, self.poll_events)

    def poll_temperature(self) -> None:
        temp = self.hardware.read_temperature()
        self.temperature_history.append((time.time() - self.temperature_start, temp))
        if len(self.temperature_history) > 20000:
            self.temperature_history = self.temperature_history[-20000:]
        self.temperature_plot.redraw()
        self.root.after(1000, self.poll_temperature)

    def save_session_snapshot(self) -> None:
        run = MeasurementRun(
            self,
            "Session Save",
            self.settings_snapshot(),
            {"manual": "session"},
            "standard",
            create_live_plot=False,
        )
        for rows in self.session_rows.values():
            for row in rows:
                with run.lock:
                    run.rows.append(row)
        paths = run.save()
        msg = "Saved session snapshot." if paths else "No session data to save."
        messagebox.showinfo("Save Now", msg)

    def open_info_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Info")
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        text = tk.Text(frame, wrap=tk.WORD, width=92, height=32)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        text.insert(tk.END, INFO_TEXT)
        text.configure(state=tk.DISABLED)
        fit_window_to_content(win, min_w=620, min_h=420, max_w=820, max_h=680, x=160, y=120)

    def report_callback_exception(self, exc_type: type, exc: BaseException, tb: Any) -> None:
        EmergencyOutputGuard.output_off()
        traceback.print_exception(exc_type, exc, tb)
        try:
            messagebox.showerror("Program Error", "".join(traceback.format_exception(exc_type, exc, tb)))
        except Exception:
            pass

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self.coordinator.active_run:
                self.coordinator.active_run.request_save_now()
            self.hardware.output_off_all()
            EmergencyOutputGuard.output_off()
            self.state.save()
        finally:
            try:
                self.root.destroy()
            except Exception:
                pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app: Optional[MeasurementApp] = None
    try:
        app = MeasurementApp()
        app.run()
    finally:
        EmergencyOutputGuard.output_off()
        if app is not None:
            try:
                app.state.save()
            except Exception:
                pass


if __name__ == "__main__":
    main()
