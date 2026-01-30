# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:02:23 2023

@author: ga63raz
"""

#%% Cell 1: Import Libraries
import numpy as np
import sys
import os
import matplotlib
import time
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import json
import matplotlib.pyplot as plt
import threading
import datetime
import timeout_decorator
from dataclasses import dataclass
from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keithley2600 import  ResultTable
from matplotlib.animation import FuncAnimation
from tkinter import messagebox
#Callable Functions:
from api_client import KiutraClient
from controller_interfaces import ContinuousTemperatureControl, MagnetControl, ADRControl,SampleControl
from Connection_Codes import ConnectKeithley, ConnectKiutra, ConnectKeithley_ASRL5_TSP
import tkinter.font as tkFont
from pymeasure.instruments.keithley import Keithley2600

#%% Cell 2: Safe File
#Safe or load File
DEFAULT_PATH = "//nas.ads.mwn.de/tuei/lab/ZEITlab-Equipment/00314_Kiutra_Cryostat_MOL_EG.058/Userdata"
SAVE_PATH = "//nas.ads.mwn.de/tuei/lab/ZEITlab-Equipment/00314_Kiutra_Cryostat_MOL_EG.058/Userdata/SweepSafes"
SAVE_PATH_2 = "C:/Users/ge36kuc/Desktop/SafesBackup"

def get_filename(user_input):
    # Get the user's filename input
    
    
    # Check if the user provided a filename
    if user_input:
        # Get the current date and time
        now = datetime.datetime.now()
        
        # Format the current date and time as a string: YYYYMMDD_HHMMSS
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        
        if integrationtime != 0.2:
            # Append integrationtime, timestamp to the user's filename input
            filename = f"{user_input}_integrationtime={integrationtime}s_{timestamp}"
        else:
            # Append only timestamp to the user's filename input if integrationtime is not set
            filename = f"{user_input}_{timestamp}"
        
        return filename
    else:
        return None
def safe_file(data_table,user_input):
    
    # Speichern der Daten
    filename = get_filename(user_input)  # Ruft den Dateinamen vom Benutzer ab
    if filename:  # Überprüft, ob der Benutzer einen Dateinamen eingegeben hat
        if not filename.endswith('.txt'):
            filename += '.txt'
        file_path = os.path.join(SAVE_PATH_2, filename)
        try:
            file_path2 = os.path.join(SAVE_PATH, filename)
            np.savetxt(file_path2, data_table, fmt='%s', header="Time\tCurrent (A)\tVoltage (V)\tTemperature (K)", comments='', delimiter='\t')
        except: print("Path not found")
        np.savetxt(file_path, data_table, fmt='%s', header="Time\tCurrent (A)\tVoltage (V)\tTemperature (K)", comments='', delimiter='\t')
    else:
        print("Kein Dateiname angegeben. Daten wurden nicht gespeichert.")
    

# Function to set the default path
def set_default_path():
    global DEFAULT_PATH
    new_path = filedialog.askdirectory(initialdir=DEFAULT_PATH)
    if new_path:
        DEFAULT_PATH = new_path
        print(f"Default Path set to: {DEFAULT_PATH}")

# Function to set the save path
def set_save_path():
    global SAVE_PATH
    new_path = filedialog.askdirectory(initialdir=SAVE_PATH)
    if new_path:
        SAVE_PATH = new_path
        print(f"Save Path set to: {SAVE_PATH}")
def on_closing(closingwindow):
    global measurement
    measurement = 0
    global rt 
    if rt:
        data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
        safe_file(data_numeric,"onclosingSMU1")    
    rt = False
    global rt2
    if rt2:
        data_numeric2 = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt2.data])
        safe_file(data_numeric2,"onclosingSMU2")    
    rt2 = False
    k.smua.source.output = 0
    k2.smua.source.output = 0
    closingwindow.destroy()
def checkformeasurement(function):
    if measurement == 1:
        pass
    else:
       start_sweep_thread(function)
#%% Cell 3: set up constants

integrationtime = 0.2
measurement = 0
global starttime
starttime=time.time()
global aniT
permanent_temperature_data=[]
live_temperature_data = []
live_resistance_data = []
global rt
global rt2
rt = False
rt2 = False
def set_integration_time():
    user_input = simpledialog.askstring("Integration Time", "Please set integration time in seconds:")
    try:
        if user_input:
            integration_time_value = float(user_input)
            if integration_time_value > 0:
                return integration_time_value  # Return the value instead of setting a global variable
            else:
                messagebox.showerror("Fehler", "Bitte geben Sie eine positive Zahl ein.")
    except ValueError:
        messagebox.showerror("Fehler", "Ungültige Eingabe. Bitte geben Sie eine Zahl ein.")
    return None  # Return None if there's an error or invalid input
def update_integration_time():
    global integrationtime
    result = set_integration_time()
    if result is not None:
        integrationtime = result
        print(f"Integration time set to: {integrationtime} seconds")
def thread_safe_update():
    root.after(0, update_integration_time)  # Schedule the update to run in the main thread

def update_temperature_label():
    global permanent_temperature_data
    global starttime
    try:
        # Query the current temperature
        current_temperature = client.query('T_sample.kelvin') or 300
        permanent_temperature_data.append([current_temperature,time.time()-starttime])
        # Update the label text with the current temperature
        temperature_label.config(text=f"Current Temperature: {current_temperature} K")
    except Exception as e:
        # Handle any exceptions that occur while querying the temperature
        temperature_label.config(text=f"Error: {str(e)}")
    finally:
        # Schedule the update_temperature_label method to run again after 1000 milliseconds (1 second)
        root.after(1000, update_temperature_label)
def whipe_data():
    global permanent_temperature_data
    global aniT
    global starttime
    permanent_temperature_data = [] 
    # Create a new FuncAnimation object with the updated data or state
    starttime=time.time()

    


def update_temp_plot(_):
        global permanent_temperature_data
        if permanent_temperature_data:
            temperature_values, time_values = zip(*permanent_temperature_data)
            lineT.set_data(time_values, temperature_values)
            axT.relim()
            axT.autoscale_view()
            figT.canvas.flush_events() 
    
def Errormessage(message):
    messagebox.showerror("Bad Entry", message)

#%% Cell 4: Magnetic Measurements (UPDATED)

#%% BLOCK 1/4 — CONFIG, STATE, HELPERS (gemeinsam für Log, Linear & Rough)

import os, time, datetime, threading, queue
import numpy as np
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------- Dual-save paths --------
SAVE_PATH_2 = r"C:/Users/ge36kuc/Desktop/SafesBackup"  # secondary backup path

# -------- CONSTANTS --------
COIL_ZERO_THRESHOLD_MA = 1e-3     # < 1 mA gilt als "praktisch 0" fürs echte Coil-OFF

# -------- UI/logic defaults --------
DEFAULTS = {
    # Coil sweep
    "coil_max_mA":               100.0,         # hard clamp ±100 mA
    "coil_use_log":              True,
    "log_tier_counts":           (12, 12, 10, 8, 6),  # [1–10]µA,[10–100]µA,[0.1–1]mA,[1–10]mA,[10–100]mA
    "coil_step_mA":              1.0,           # linear step
    "coil_start_mA":             0.0,           # Startstrom für Resume/Anchor

    # Temperature target
    "target_temp_K":             "",

    # Ic threshold & timing
    "v_thresh":                  2e-4,          # |V| >= 2e-4 V => switched
    "nplc":                      0.01,
    "pulse_width_ms":            2.0,
    "pulse_settle_ms":           0.3,
    "cooldown_ms":               2.0,

    # Guided Ic search bounds
    "ic_start_uA":               0.1,
    "ic_growth":                 1.5,
    "ic_max_uA":                 1000.0,        # 1 mA cap
    "refine_bits":               12,
    "refine_bits_highfield":     8,

    # Robustness / Constraints
    "switch_confirm":            3,             # ≥3 aufeinanderfolgende Over-Thresholds
    "ic0_min_uA":                1.0,

    # Guided behavior (heat-minimizing)
    "guided_lower_frac":         0.20,
    "guided_step_rel":           0.05,
    "guided_min_step_uA":        0.05,
    "guided_window_frac":        0.25,

    # Zero-field IV refinement (pretty trace)
    "iv0_refine":                True,
    "iv0_pts_per_branch":        60,
    "iv0_above_frac":            0.30,
    "iv0_window_frac":           0.25,

    # Band refinement [½·Ic … ~Ic] — B=0 und nur bei kleinen |B|
    "band_refine_enable":        True,
    "band_refine_frac_low":      0.5,
    "band_refine_pts":           25,
    "band_refine_top_margin":    0.995,         # ≤ 0.995*Ic
    "band_refine_coil_uA_max":   2000.0,        # refine nur, wenn |I_coil| ≤ 2 mA

    # Coil output behavior
    "coil_settle_ms":            5.0,
    "coil_ramp_steps":           5,

    # Temperature guard & ADR
    "dT_stop":                   0.2,           # pause if T > T0 + 0.2 K
    "dT_resume":                 0.1,           # resume when T <= 0.1 K above T0
    "cooldown_total_min":        90.0,
    "adr_trigger_min":           50.0,
    "adr_settle_K":              0.05,
    "post_adr_settle_min":       2.0,
    "adr_max_resets":            0,             # 0 = unlimited

    # Measure both Ic branches
    "measure_negative":          True,

    # Error-Queue Handling
    "error_drain_every_pulses":  200,

    # Rough-scan defaults
    "rough_max_mA":              100.0,
    "rough_step_mA":             1.0,
    "rough_refine_bits":         6,
    "rough_smooth_pts":          5,             # gleitendes Mittel (ungerade gut)
    "rough_min_prominence":      0.05,          # relativ zu max(Ic); 5%

    # really-vthresh confirm
    "confirm_extra_pulses":      2,
    "confirm_accept_min":        2,
    "confirm_rel_thresh":        0.9,

    # log sweep anchor-mode
    "log_anchor_mode":           True,

    # Burst control (Queue-full vermeiden, Speed behalten)
    "burst_ops_before_wait":     25,   # wie viele Kommandos vor waitcomplete()
    "burst_backoff_s":           0.02,  # kurze Pause nach Queue-full
}

# -------- Thread-safe state --------
class FFState:
    def __init__(self):
        self.stop_event = threading.Event()
        self.progress_q = queue.Queue()
        self.rows_pos = []               # [timestamp, T, Icoil[A], Ic_pos[A], limit]
        self.rows_neg = []               # [timestamp, T, Icoil[A], Ic_neg[A], limit]
        self.iv_points = []              # live IV (B=0)
        self.iv_first_points = []        # first IV curve (B=0)
        self.iv_latest_points = []       # latest IV curve (B=0)
        self.iv_current_points = None    # active IV curve collector
        self.iv_curve_index = 0
        self.iv_search_rows = []         # ALL pulses: [timestamp,T[K],Icoil[A],branch,I_junc[A],V[V]]
        self.coil_points_A = []          # x-values for Ic plot
        self.icp_points_A = []           # y+ scatter
        self.icn_points_A = []           # y- scatter
        self.base = None
        self.T0 = None
        self.settings = None
        self.thread = None
        self.adr_events = 0
        self.command_q = queue.Queue()   # Live-Controls: set_step / jump (Linear)
        self.current_coil_uA = None      # zuletzt gesetzter Coil-Strom (µA)

_ff_state = None
_ERR_DRAIN_EVERY = DEFAULTS["error_drain_every_pulses"]
_pulse_counter = 0

# -------- Low-level TSP I/O --------
def _tsp_send(dev, cmd: str):
    """Best-effort write of a TSP command."""
    for m in ("write", "send", "sendcmd", "execute", "exec", "write_raw"):
        if hasattr(dev, m):
            try:
                getattr(dev, m)(cmd); return True
            except Exception: pass
    for path in ("visa", "instrument", "resource"):
        h = getattr(dev, path, None)
        if h is not None and hasattr(h, "write"):
            try:
                h.write(cmd); return True
            except Exception: pass
    return False

def _tsp_batch(dev, *cmds):
    """Send multiple TSP commands in one write to reduce queue load."""
    parts = []
    for cmd in cmds:
        if not cmd:
            continue
        text = str(cmd).strip().rstrip(";")
        if text:
            parts.append(text)
    if not parts:
        return False
    return _tsp_send(dev, "; ".join(parts))

def _tsp_query(dev, cmd: str):
    """TSP query returning a string (or None)."""
    for m in ("query", "ask"):
        if hasattr(dev, m):
            try:
                return getattr(dev, m)(cmd)
            except Exception:
                pass
    # Fallback: write + read
    if _tsp_send(dev, cmd):
        for m in ("read", "read_raw"):
            if hasattr(dev, m):
                try:
                    return getattr(dev, m)()
                except Exception:
                    pass
    for path in ("visa", "instrument", "resource"):
        h = getattr(dev, path, None)
        if h is not None and hasattr(h, "query"):
            try:
                return h.query(cmd)
            except Exception:
                pass
    return None

def _get_error_count(dev):
    try:
        resp = _tsp_query(dev, "print(errorqueue.count)")
        if resp is None: return None
        return int(str(resp).strip().splitlines()[-1])
    except Exception:
        return None

def _drain_errorqueue(dev):
    try:
        _tsp_send(dev, "errorqueue.clear()")
    except Exception:
        pass

def _drain_errorqueue_with_log(dev, label="SMU"):
    """Read and drain errorqueue; log if queue full (-350) was reported."""
    queue_full = False
    count = _get_error_count(dev)
    if count is None or count <= 0:
        return False
    for _ in range(int(count)):
        resp = _tsp_query(dev, "print(errorqueue.next())")
        if resp is None:
            continue
        msg = str(resp).strip()
        if "-350" in msg or "Queue full" in msg or "queue full" in msg:
            queue_full = True
    if queue_full:
        print(f"[WARN] {label} reported -350 Queue full in errorqueue.")
    try:
        _tsp_send(dev, "status.reset()")
    except Exception:
        pass
    return queue_full

def _waitcomplete(dev):
    """Robustes wait, ohne hart zu blockieren wenn Treiber anders heißt."""
    for m in ("waitcomplete", "wait_complete", "waitFinish", "waitfinish"):
        if hasattr(dev, m):
            try:
                getattr(dev, m)(); return
            except Exception:
                pass
    try:
        _tsp_send(dev, "waitcomplete()")
    except Exception:
        pass

def _clear_k2600(dev):
    try:
        _tsp_send(dev, "errorqueue.clear()")
        _tsp_send(dev, "status.reset()")
    except Exception:
        pass

def _clear_k(dev):
    """Legacy clear; nutzt jetzt TSP-Helper."""
    _clear_k2600(dev)
    try: _waitcomplete(dev)
    except Exception: pass

# ---- Burst control / queue-safe wrappers ----
_burst_counter_k  = 0
_burst_counter_k2 = 0

def _tsp_waitcomplete(dev):
    try:
        if hasattr(dev, "waitcomplete"):
            dev.waitcomplete()
        else:
            _tsp_send(dev, "delay(0)")
    except Exception:
        pass

def _maybe_wait(dev, which='k', every=None):
    """Nach 'every' Kommandos einmal waitcomplete()."""
    global _burst_counter_k, _burst_counter_k2
    every = int(every if every is not None else DEFAULTS["burst_ops_before_wait"])
    if every <= 0:
        return
    if which == 'k':
        _burst_counter_k += 1
        if _burst_counter_k >= every:
            _tsp_waitcomplete(dev)
            _burst_counter_k = 0
    else:
        _burst_counter_k2 += 1
        if _burst_counter_k2 >= every:
            _tsp_waitcomplete(dev)
            _burst_counter_k2 = 0

def _recover_queue_full(dev, ex, *, which='k'):
    """Handle -350 „Queue full“: wait, clear, kurzer Backoff, Zähler resetten."""
    msg = str(ex)
    if "-350" in msg or "Queue full" in msg or "queue" in msg.lower():
        _tsp_waitcomplete(dev)
        _clear_k2600(dev)
        time.sleep(float(DEFAULTS["burst_backoff_s"]))
        if which == 'k':
            global _burst_counter_k; _burst_counter_k = 0
        else:
            global _burst_counter_k2; _burst_counter_k2 = 0
        return True
    return False

def _queue_safe_call(dev, which, op, *, retries=2, fallback=None):
    """Retry op on -350 queue full errors and return fallback on final failure."""
    for attempt in range(int(retries) + 1):
        try:
            return op()
        except Exception as ex:
            if _recover_queue_full(dev, ex, which=which):
                continue
            if attempt >= int(retries):
                if fallback is not None:
                    return fallback
                raise
    return fallback

# -------- Helpers (paths, saving, timing, plotting) --------
def _get_primary_path():
    try:
        p = SAVE_PATH  # evtl. im Usercode definiert
    except NameError:
        p = os.getcwd()
    return p

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _dated_base(base: str) -> str:
    return f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

def _save_csv_both(base, suffix, header, arr):
    fname = f"{base}{suffix}.csv"
    p1 = _ensure_dir(_get_primary_path()); p2 = _ensure_dir(SAVE_PATH_2)
    f1 = os.path.join(p1, fname); f2 = os.path.join(p2, fname)
    np.savetxt(f1, arr, fmt='%s', delimiter=',', header=header, comments='')
    try: np.savetxt(f2, arr, fmt='%s', delimiter=',', header=header, comments='')
    except Exception: pass
    return f1, f2

def _save_dual(base, suffix, header, arr):
    _save_csv_both(base, suffix, header, arr)

def _get_T():
    try:
        return float(client.query('T_sample.kelvin') or 300.0)
    except Exception:
        return 300.0

def _set_measure_speeds(nplc):
    try: k.smua.measure.nplc = float(nplc)
    except Exception: pass
    try: k2.smua.measure.nplc = float(nplc)
    except Exception: pass

def _make_plot_window(title, xlab, ylab, with_two_series=False,
                      label1=None, label2=None, color1=None, color2=None):
    win = tk.Toplevel(root); win.title(title)
    fig = Figure(figsize=(6, 4), dpi=100); ax = fig.add_subplot(111)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    if label1 is None:
        label1 = "Ic+" if with_two_series else "data"
    line1, = ax.plot([], [], linestyle='None', marker='o', markersize=3,
                     label=label1, color=color1)
    line2 = None
    if with_two_series:
        if label2 is None:
            label2 = "Ic−"
        line2, = ax.plot([], [], linestyle='None', marker='o', markersize=3,
                         label=label2, color=color2)
        ax.legend(loc="best")
    else:
        try: ax.legend().remove()
        except Exception: pass
    canvas = FigureCanvasTkAgg(fig, master=win); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return win, fig, ax, line1, line2, canvas

def _parse_counts(s: str, fallback=(12,12,10,8,6)):
    try:
        parts = [int(x.strip()) for x in s.split(',')]
        if len(parts) != 5: return fallback
        return tuple(max(0, int(v)) for v in parts)
    except Exception:
        return fallback

# ---- Coil setpoints: generators + center-out ORDERING AROUND START ----
def _generate_coil_setpoints_log_center_out_uA(coil_max_uA: float, counts_per_decade, include_zero=True):
    max_uA = float(min(abs(coil_max_uA), 1e5))  # ≤ 100 mA
    decades = [(1.0, 10.0), (10.0, 100.0), (100.0, 1e3), (1e3, 1e4), (1e4, 1e5)]
    mags = []
    first = True
    for (lo, hi), n in zip(decades, counts_per_decade):
        n = int(max(0, n)); 
        if n == 0: continue
        hi_cap = min(hi, max_uA); 
        if hi_cap <= lo: continue
        vals = np.logspace(np.log10(lo), np.log10(hi_cap), num=n, endpoint=True)
        if not first: vals = [v for v in vals if v > lo * (1 + 1e-12)]
        mags.extend(vals); first = False
    mags = sorted(set(float(x) for x in mags))
    seq = [0.0] if include_zero else []
    for m in mags: seq.append(+m); seq.append(-m)
    seq = [x for x in seq if abs(x) <= max_uA + 1e-9]
    # unique
    out, seen = [], set()
    for v in seq:
        key = round(v, 12)
        if key not in seen:
            out.append(v); seen.add(key)
    return out

def _generate_coil_setpoints_linear_center_out_uA(coil_max_mA: float, step_mA: float, include_zero=True):
    max_mA = float(min(abs(coil_max_mA), 100.0))
    step_mA = float(abs(step_mA))
    if step_mA <= 0 or max_mA <= 0:
        return [0.0] if include_zero else []
    n = int(np.floor(max_mA / step_mA))
    mags_mA = [i * step_mA for i in range(1, n + 1)]
    seq_uA = [0.0] if include_zero else []
    for m in mags_mA:
        uA = m * 1000.0; seq_uA.append(+uA); seq_uA.append(-uA)
    # unique
    out, seen = [], set()
    for v in seq_uA:
        key = round(v, 9)
        if key not in seen:
            out.append(v); seen.add(key)
    return out

def _order_center_out_keyfun(x, start_uA):
    return (abs(x - start_uA), 0 if (x - start_uA) >= 0 else 1)

def _order_center_out_around_start(uA_points, start_mA):
    """Reihenfolge center-out um Startstrom (mA). Startpunkt zuerst, dann zunehmend entfernte Punkte."""
    start_uA = float(start_mA) * 1000.0
    uniq = sorted(set(float(x) for x in uA_points))
    if start_uA not in uniq:
        uniq.append(start_uA); uniq = sorted(uniq)
    ordered = sorted(uniq, key=partial(_order_center_out_keyfun, start_uA=start_uA))
    return ordered

def _append_unique_value(seq, seen, v, max_uA):
    key = round(float(v), 9)
    if abs(v) <= max_uA + 1e-9 and key not in seen:
        seq.append(float(v))
        seen.add(key)

def _use_temperature_control(target_temp):
    try:
        return float(target_temp) > 2.9
    except Exception:
        return False

def _start_temperature_control(setpoint, ramp=0.2):
    try:
        if getattr(temperature_control, "is_active", False):
            temperature_control.stop()
            time.sleep(0.5)
        temperature_control.start((float(setpoint), float(ramp)))
        return True
    except Exception:
        return False

# ---- Temperature/ADR guard (inkl. settle) ----
def wait_for_temperature_settle(start_temp, s, *, label_prefix="PAUSE"):
    t0 = time.time(); adr_sent = False
    window = []
    wnd_len = max(1, int(max(1.0, s["post_adr_settle_s"]) / 0.5))  # 0.5 s sampling
    while not _ff_state.stop_event.is_set():
        time.sleep(0.5)
        T = _get_T()
        elapsed = time.time() - t0
        eta_adr = max(0, s["adr_trigger_s"] - elapsed)
        eta_total = max(0, s["cooldown_total_s"] - elapsed)
        txt = f"{label_prefix}: T={T:.3f} K | ADR in ≥{eta_adr/60:.1f} min | cooldown ~{eta_total/60:.1f} min"
        txt += f" | ADR events: {_ff_state.adr_events}/{('∞' if s['adr_max_resets']==0 else s['adr_max_resets'])}"
        _ff_state.progress_q.put(('status', txt))

        if (not adr_sent) and (elapsed >= s["adr_trigger_s"]):
            _ff_state.adr_events += 1
            unlimited = (s["adr_max_resets"] == 0)
            final_now = (not unlimited) and (_ff_state.adr_events >= s["adr_max_resets"])
            try:
                _ff_state.progress_q.put(('status', f"{'Final ' if final_now else ''}ADR… (event {_ff_state.adr_events})"))
                try:
                    adr_control.start_adr(setpoint=start_temp, ramp=0.2, adr_mode=None, operation_mode='cadr',
                                          auto_regenerate=True, pre_regenerate=True)
                except Exception as ex:
                    _ff_state.progress_q.put(('status', f"ADR call failed/absent: {ex}"))
                adr_sent = True
            except Exception as ex:
                _ff_state.progress_q.put(('status', f"ADR error: {ex}"))
            if final_now:
                _ff_state.progress_q.put(('status', "Final ADR sent — stopping and saving results…"))
                _save_results(_ff_state)
                _ff_state.progress_q.put(('done', "Stopped after final ADR. Results saved."))
                _ff_state.stop_event.set()
                return

        if adr_sent and abs(T - start_temp) <= s["adr_settle_K"]:
            _ff_state.progress_q.put(('status', f"ADR reached setpoint; extra settling…"))
            _stable_block_s(start_temp, s["post_adr_settle_s"])
            return

        if T <= start_temp + s["dT_resume"]:
            window.append(T)
            if len(window) > wnd_len: window.pop(0)
            if len(window) == wnd_len and (max(window) - min(window) < 0.01):
                time.sleep(max(0.0, s["post_adr_settle_s"]))
                return

def _stable_block_s(ref_T, duration_s):
    if duration_s <= 0: return
    buf = []
    t_end = time.time() + duration_s
    while time.time() < t_end and not _ff_state.stop_event.is_set():
        T = _get_T(); buf.append(T)
        if len(buf) > 20: buf.pop(0)
        time.sleep(0.5)
    if len(buf) >= 2 and (max(buf) - min(buf) >= 0.02):
        time.sleep(duration_s * 0.5)

def _prep_pulse_smua(k, nplc=0.01, limit_v=0.01):
    """SMU1 (Junction) vorbereiten: schneller Measure, Ausgang auf 0 A, Output AUS."""
    try:
        k.smua.source.func = 2              # current mode
        k.smua.source.autorangei = 1        # autorange current
        k.smua.measure.nplc = float(nplc)
        k.smua.source.limitv = float(limit_v)
        k.smua.source.leveli = 0.0
        k.smua.source.output = 0            # wichtig: AUS bis zum Puls
        _waitcomplete(k)
    except Exception:
        pass


#%% BLOCK 2/4 — PULSED PRIMITIVES, IC SEARCH, REFINEMENTS + UI FRAMES

def _junction_pulse_measure_v(I_uA, pulse_width_s, pulse_settle_s, cooldown_s):
    """
    Saubere Pulse auf SMU1 (k): Output nur für die Pulse EIN,
    Level nach Messung wieder 0, Output AUS. Nutzt leveli statt apply_current().
    """
    I_A = float(I_uA) * 1e-6
    def _pulse_output_on():
        if not _tsp_batch(k, "smua.source.func=2", "smua.source.limitv=0.01", "smua.source.output=1"):
            setattr(k.smua.source, "func", 2)  # current
            setattr(k.smua.source, "limitv", 0.01)
            setattr(k.smua.source, "output", 1)
        _maybe_wait(k, which='k')

    _queue_safe_call(k, 'k', _pulse_output_on, retries=2)

    # Level setzen
    def _pulse_set_level():
        if not _tsp_batch(k, f"smua.source.leveli={I_A}"):
            setattr(k.smua.source, "leveli", I_A)
        _maybe_wait(k, which='k')

    _queue_safe_call(k, 'k', _pulse_set_level, retries=2)

    if pulse_settle_s > 0:
        time.sleep(pulse_settle_s)

    # Spannung messen
    v = _queue_safe_call(
        k, 'k',
        lambda: k.smua.measure.v(),
        retries=2,
        fallback=float('nan')
    )

    remain = max(0.0, pulse_width_s - pulse_settle_s)
    if remain > 0:
        time.sleep(remain)

    # Ausgang und Level zurücknehmen (wirklich aus!)
    def _pulse_output_off():
        if not _tsp_batch(k, "smua.source.leveli=0", "smua.source.output=0"):
            setattr(k.smua.source, "leveli", 0.0)
            setattr(k.smua.source, "output", 0)
        _maybe_wait(k, which='k')

    _queue_safe_call(k, 'k', _pulse_output_off, retries=1)

    if cooldown_s > 0:
        time.sleep(cooldown_s)

    # Fehler gelegentlich drainen
    _drain_errors_periodic(_ERR_DRAIN_EVERY)

    return v


def _junction_pulse_measure_v_guarded(I_uA, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard=None):
    if temp_guard:
        ok = temp_guard()
        if not ok:
            return None
    return _junction_pulse_measure_v(I_uA, pulse_width_s, pulse_settle_s, cooldown_s)
# --- Helper Patch: add missing helpers if not already defined ---

import numpy as np, datetime, os

# Fallbacks auf bereits vorhandene Globals
try:
    DEFAULTS
except NameError:
    DEFAULTS = {"error_drain_every_pulses": 200, "rough_smooth_pts": 5, "rough_min_prominence": 0.05}

# _ERR_DRAIN_EVERY & _pulse_counter
try:
    _ERR_DRAIN_EVERY
except NameError:
    _ERR_DRAIN_EVERY = int(DEFAULTS.get("error_drain_every_pulses", 200))
try:
    _pulse_counter
except NameError:
    _pulse_counter = 0

# ---- _drain_errors_periodic(every_pulses) ----
try:
    _drain_errors_periodic
except NameError:
    def _drain_errors_periodic(every_pulses: int):
        """Alle 'every_pulses' Impulse: Fehlerqueues der K2600er leeren (best effort)."""
        global _pulse_counter
        _pulse_counter += 1
        if every_pulses and every_pulses > 0 and (_pulse_counter % int(every_pulses) == 0):
            try:
                _drain_errorqueue_with_log(k, "SMU1")
            except Exception:
                pass
            try:
                _drain_errorqueue_with_log(k2, "SMU2")
            except Exception:
                pass

# ---- _save_dual Fallback (falls nicht vorhanden) ----
try:
    _save_dual
except NameError:
    def _save_dual(base, suffix, header, arr):
        """Schreibt CSV in aktuelles Verzeichnis; minimaler Fallback."""
        fname = f"{base}{suffix}.csv"
        try:
            np.savetxt(fname, arr, fmt='%s', delimiter=',', header=header, comments='')
        except Exception as ex:
            print(f"[WARN] save_dual failed for {fname}: {ex}")

# ---- Smoother / Extrema / Period (nur falls fehlen) ----
try:
    _smooth_moving_avg
except NameError:
    def _smooth_moving_avg(y, w):
        w = int(max(1, w))
        if w == 1 or len(y) <= 2: return np.array(y, float)
        k = w // 2
        padL = np.full(k, y[0], dtype=float)
        padR = np.full(k, y[-1], dtype=float)
        arr = np.concatenate([padL, np.asarray(y, float), padR])
        return np.convolve(arr, np.ones(w)/w, mode='valid')

try:
    _find_extrema_centered
except NameError:
    def _extrema_prominent(idxs, y, thr, invert=False):
        keep = []
        for idx in idxs:
            val = y[idx]
            left = y[max(0, idx-1)]
            right = y[min(len(y)-1, idx+1)]
            prom = (val - min(left, right)) if not invert else (max(left, right) - val)
            if prom >= thr:
                keep.append(idx)
        return keep

    def _find_extrema_centered(x, y, min_prom_rel=0.05):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 5: return [], []
        dy = np.diff(y)
        mins, maxs = [], []
        for i in range(1, len(dy)):
            if dy[i-1] < 0 and dy[i] > 0: mins.append(i)
            if dy[i-1] > 0 and dy[i] < 0: maxs.append(i)
        if len(y) > 0:
            yr = max(1e-12, (np.nanmax(y) - np.nanmin(y)))
            thr = float(min_prom_rel) * yr
            maxs = _extrema_prominent(maxs, y, thr, invert=False)
            mins = _extrema_prominent(mins, y, thr, invert=True)
        return mins, maxs

try:
    _estimate_period_from_extrema
except NameError:
    def _estimate_period_from_extrema(x, idxs):
        if len(idxs) < 2: return float('nan'), 0
        xs = np.array([x[i] for i in idxs], float)
        diffs = np.diff(np.sort(xs))
        if len(diffs) == 0: return float('nan'), len(idxs)
        return float(np.nanmedian(diffs)), len(idxs)

# ---- _save_rough_arrays ----
try:
    _save_rough_arrays
except NameError:
    def _save_rough_arrays(state, xs_A, icA, smooth_pts=None, suffix=""):
        """Speichert Rough-Arrays + einfache Diagnostik (Periode) an beide Speicherorte (via _save_dual)."""
        if xs_A is None or icA is None or len(xs_A) == 0:
            return
        if smooth_pts is None:
            smooth_pts = int(getattr(state, "settings", {}).get("rough_smooth_pts", DEFAULTS.get("rough_smooth_pts", 5)))
        ic_smooth = _smooth_moving_avg(icA, max(1, smooth_pts))
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        arr = np.array([(now, xs_A[j], icA[j], ic_smooth[j]) for j in range(len(xs_A))], dtype=object)
        _save_dual(state.base, f"_RoughScan_Ic{suffix}", "timestamp,I_coil[A],Ic_raw[A],Ic_smooth[A]", arr)

        # Diagnose
        try:
            mins_idx, maxs_idx = _find_extrema_centered(xs_A, ic_smooth,
                min_prom_rel=getattr(state, "settings", {}).get("rough_min_prominence", DEFAULTS.get("rough_min_prominence", 0.05)))
            per_min, nmin = _estimate_period_from_extrema(xs_A, mins_idx)
            per_max, nmax = _estimate_period_from_extrema(xs_A, maxs_idx)
        except Exception:
            per_min, nmin, per_max, nmax = float('nan'), 0, float('nan'), 0

        diag = np.array([
            ("min_period_estimate[A]", f"{per_min:.6e}", "count_minima", str(nmin)),
            ("max_period_estimate[A]", f"{per_max:.6e}", "count_maxima", str(nmax)),
        ], dtype=object)
        _save_dual(state.base, f"_RoughScan_Diagnostics{suffix}", "key,value,key2,value2", diag)

# ---- _build_log_anchor_sequence ----
try:
    _build_log_anchor_sequence
except NameError:
    def _build_log_anchor_sequence(coil_max_uA, counts_per_decade, start_mA):
        """
        Sequenz: +Istart, -Istart, dann log-mäßig weiter nach außen: +I2, -I2, ...
        Nutzt _generate_coil_setpoints_log_center_out_uA falls vorhanden, sonst lokalen Fallback.
        """
        start_uA = abs(float(start_mA)) * 1000.0
        if start_uA <= 0:
            return None  # -> Caller fällt auf center-out zurück

        # versuche bestehende Generatorfunktion zu verwenden
        try:
            raw = _generate_coil_setpoints_log_center_out_uA(
                coil_max_uA, counts_per_decade, include_zero=False
            )
        except Exception:
            # Minimal-Fallback: 5 Dekaden, gleich wie Defaults typisch
            max_uA = float(min(abs(coil_max_uA), 1e5))
            decades = [(1.0, 10.0), (10.0, 100.0), (100.0, 1e3), (1e3, 1e4), (1e4, 1e5)]
            mags = []
            for (lo, hi), n in zip(decades, counts_per_decade):
                n = int(max(0, n))
                if n == 0: 
                    continue
                hi_cap = min(hi, max_uA)
                if hi_cap <= lo:
                    continue
                vals = np.logspace(np.log10(lo), np.log10(hi_cap), num=n, endpoint=True)
                mags.extend(vals)
            mags = sorted(set(float(x) for x in mags))
            raw = []
            for m in mags:
                raw.extend([+m, -m])

        seq = [ +start_uA, -start_uA ]
        for v in raw:
            if abs(v) > start_uA + 1e-12:
                seq.extend([ +float(v), -float(v) ])

        # unique + clamp
        out, seen = [], set()
        cap = float(coil_max_uA)
        for v in seq:
            if abs(v) <= cap + 1e-9:
                key = round(float(v), 12)
                if key not in seen:
                    out.append(float(v)); seen.add(key)
        return out


def _coil_set_current_uA(target_uA, ramp_steps, settle_s):
    """Coil (SMU2=k2) setzen, Queue-sicher; sehr kleine Steps zusammenfassen."""
    # Sicherstellen: Current-Mode, Output EIN
    def _coil_output_on():
        if not _tsp_batch(k2, "smub.source.func=2", "smub.source.limitv=10.0", "smub.source.output=1"):
            setattr(k2.smua.source, "func", 2)
            setattr(k2.smua.source, "limitv", 10.0)
            setattr(k2.smua.source, "output", 1)
        _maybe_wait(k2, which='k2')

    _queue_safe_call(k2, 'k2', _coil_output_on, retries=2)

    target_uA = float(target_uA)
    ramp_steps = int(max(1, ramp_steps))

    if ramp_steps > 1:
        # don’t ramp below ~50 nA per step
        min_increment_uA = 0.05  # 50 nA
        est_inc = abs(target_uA) / ramp_steps
        if est_inc < min_increment_uA:
            ramp_steps = max(1, int(abs(target_uA) / min_increment_uA))

    if ramp_steps <= 1:
        def _coil_set_level():
            level = target_uA * 1e-6
            if not _tsp_batch(k2, f"smub.source.leveli={level}"):
                setattr(k2.smua.source, "leveli", level)
            _maybe_wait(k2, which='k2')

        _queue_safe_call(k2, 'k2', _coil_set_level, retries=2)
    else:
        for frac in np.linspace(0.0, 1.0, ramp_steps):
            def _coil_ramp_step(frac=frac):
                level = (frac * target_uA) * 1e-6
                if not _tsp_batch(k2, f"smub.source.leveli={level}"):
                    setattr(k2.smua.source, "leveli", level)
                _maybe_wait(k2, which='k2')

            _queue_safe_call(k2, 'k2', _coil_ramp_step, retries=2)

    if settle_s > 0:
        time.sleep(settle_s)


def _coil_off():
    """Coil wirklich AUS: Level 0, wait, Output 0."""
    def _coil_output_off():
        if not _tsp_batch(k2, "smub.source.leveli=0", "smub.source.output=0"):
            k2.smua.source.leveli = 0.0
            k2.smua.source.output = 0
        _waitcomplete(k2)

    try:
        _queue_safe_call(k2, 'k2', _coil_output_off, retries=1)
    except Exception as ex:
        print(f"[WARN] Coil off failed: {ex}")


# --- Ic-Suche: adaptive + guided ---
def _find_ic_uA_adaptive(
    v_thresh, start_uA, growth, max_uA, refine_bits,
    pulse_width_s, pulse_settle_s, cooldown_s,
    *, sign=+1, collect_iv=False, stop_event=None,
    switch_confirm=3, min_ic_uA=None, temp_guard=None
):
    iv = []
    start_uA = float(max(0.0, start_uA))
    if min_ic_uA is not None: start_uA = max(start_uA, float(min_ic_uA))
    current = start_uA; last_ok = 0.0; last_bad = None; consec_over = 0

    while current <= max_uA:
        if stop_event and stop_event.is_set(): return (sign * last_ok, True, iv)
        I_try = sign * current
        v = _junction_pulse_measure_v_guarded(I_try, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
        if v is None: return (sign * last_ok, True, iv)
        if collect_iv: iv.append((I_try, v))

        confirm_extra      = int(DEFAULTS.get("confirm_extra_pulses", 2))
        confirm_accept_min = int(DEFAULTS.get("confirm_accept_min",   2))
        confirm_rel        = float(DEFAULTS.get("confirm_rel_thresh", 0.9))

        if np.isfinite(v) and abs(v) >= v_thresh:
            ok = _confirm_switch_same_I(sign, current,
                                        pulse_width_s, pulse_settle_s, cooldown_s,
                                        v_thresh, confirm_rel, confirm_extra, confirm_accept_min,
                                        collect_iv=collect_iv, iv_sink=iv, temp_guard=temp_guard)
            if ok:
                consec_over += 1
                if consec_over >= int(switch_confirm):
                    last_bad = current
                    break
            else:
                consec_over = 0
                last_ok = current
            current = max(current * float(growth), current + 1e-6)
        else:
            consec_over = 0
            last_ok = current
            current = max(current * float(growth), current + 1e-6)

    else:
        return (sign * current, True, iv)

    low, high = last_ok, last_bad
    for _ in range(int(refine_bits)):
        if stop_event and stop_event.is_set(): return (sign * (0.5*(low+high)), True, iv)
        mid = 0.5 * (low + high)
        v = _junction_pulse_measure_v_guarded(sign * mid, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
        if v is None: return (sign * (0.5*(low+high)), True, iv)
        if collect_iv: iv.append((sign * mid, v))
        if abs(v) >= v_thresh: high = mid
        else: low = mid
    return (sign * 0.5*(low+high), False, iv)


def _confirm_switch_same_I(sign, IuA, pulse_width_s, pulse_settle_s, cooldown_s,
                           v_thresh, rel_thresh, extra_pulses, accept_min,
                           *, collect_iv=False, iv_sink=None, temp_guard=None):
    """Mehrfachmessung am selben I: akzepte nur, wenn genügend Wiederholungen ≥ rel*Vthresh."""
    thr = float(v_thresh) * float(rel_thresh)
    hits = 0
    for _ in range(int(extra_pulses)):
        v2 = _junction_pulse_measure_v_guarded(sign * float(IuA), pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
        if v2 is None:
            return False
        if collect_iv and iv_sink is not None:
            iv_sink.append((sign * float(IuA), v2))
        if np.isfinite(v2) and abs(v2) >= thr:
            hits += 1
    return (hits >= int(accept_min))

def _guided_pulse(sign, IuA, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard=None):
    if stop_event and stop_event.is_set():
        return None
    v = _junction_pulse_measure_v_guarded(sign * IuA, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
    if v is None:
        return None
    if collect_iv:
        iv.append((sign * IuA, v))
    return v


def _find_ic_uA_guided(
    v_thresh, start_uA, growth, max_uA, refine_bits,
    pulse_width_s, pulse_settle_s, cooldown_s,
    *, sign=+1, collect_iv=False, stop_event=None,
    prev_ic_uA=None, lower_frac=0.10, step_rel=0.05, min_step_uA=0.05, window_frac=0.25,
    switch_confirm=3, min_ic_uA=None, temp_guard=None
):
    if not prev_ic_uA or prev_ic_uA <= 0:
        return _find_ic_uA_adaptive(
            v_thresh, max(start_uA, (min_ic_uA or 0.0)), growth, max_uA, refine_bits,
            pulse_width_s, pulse_settle_s, cooldown_s,
            sign=sign, collect_iv=collect_iv, stop_event=stop_event,
            switch_confirm=switch_confirm, min_ic_uA=min_ic_uA, temp_guard=temp_guard
        )

    iv = []
    prev_mag = float(prev_ic_uA)
    if min_ic_uA is not None: prev_mag = max(prev_mag, float(min_ic_uA))
    low_start = max(start_uA, prev_mag * (1.0 - float(lower_frac)))
    if min_ic_uA is not None: low_start = max(low_start, float(min_ic_uA))
    hi_cap    = min(max_uA,  prev_mag * (1.0 + float(window_frac)))
    step_uA   = max(min_step_uA, prev_mag * float(step_rel))

    v0 = _guided_pulse(sign, low_start, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard)
    if v0 is None: return (sign * low_start, True, iv)

    if abs(v0) >= v_thresh:
        last_bad = low_start; cur = max(start_uA, low_start * (1.0 - step_rel)); last_ok = None
        for _ in range(200):
            v = _guided_pulse(sign, cur, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard)
            if v is None: return (sign * (last_ok if last_ok else cur), True, iv)
            if abs(v) < v_thresh: last_ok = cur; break
            last_bad = cur; cur = max(start_uA, cur * (1.0 - step_rel))
        if last_ok is None: return (sign * max(start_uA, float(min_ic_uA or start_uA)), True, iv)
        low, high = last_ok, last_bad
    else:
        last_ok = low_start; last_bad = None; cur = low_start + step_uA; consec_over = 0
        for _ in range(2000):
            if cur > hi_cap: break
            v = _guided_pulse(sign, cur, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard)
            if v is None: return (sign * last_ok, True, iv)

            confirm_extra      = int(DEFAULTS.get("confirm_extra_pulses", 2))
            confirm_accept_min = int(DEFAULTS.get("confirm_accept_min",   2))
            confirm_rel        = float(DEFAULTS.get("confirm_rel_thresh", 0.9))

            if np.isfinite(v) and abs(v) >= v_thresh:
                ok = _confirm_switch_same_I(sign, cur,
                                            pulse_width_s, pulse_settle_s, cooldown_s,
                                            v_thresh, confirm_rel, confirm_extra, confirm_accept_min,
                                            collect_iv=collect_iv, iv_sink=iv, temp_guard=temp_guard)
                if ok:
                    consec_over += 1
                    if consec_over >= int(switch_confirm):
                        last_bad = cur
                        break
                else:
                    consec_over = 0
                    last_ok = cur
            else:
                consec_over = 0
                last_ok = cur

            cur = cur + step_uA

        # Expand, falls noch kein last_bad
        expand_rounds = 0; cur_hi = max(low_start + step_uA, last_ok + step_uA)
        while last_bad is None and cur_hi <= max_uA and expand_rounds < 10:
            v = _guided_pulse(sign, cur_hi, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard)
            if v is None: return (sign * last_ok, True, iv)
            if np.isfinite(v) and abs(v) >= v_thresh:
                confirm_extra      = int(DEFAULTS.get("confirm_extra_pulses", 2))
                confirm_accept_min = int(DEFAULTS.get("confirm_accept_min",   2))
                confirm_rel        = float(DEFAULTS.get("confirm_rel_thresh", 0.9))
                ok = _confirm_switch_same_I(sign, cur_hi,
                                            pulse_width_s, pulse_settle_s, cooldown_s,
                                            v_thresh, confirm_rel, confirm_extra, confirm_accept_min,
                                            collect_iv=collect_iv, iv_sink=iv, temp_guard=temp_guard)
                if ok:
                    last_bad = cur_hi
                    break
                else:
                    last_ok = cur_hi
                    cur_hi = min(max_uA, cur_hi * (1.0 + step_rel))
                    expand_rounds += 1
            else:
                last_ok = cur_hi
                cur_hi = min(max_uA, cur_hi * (1.0 + step_rel))
                expand_rounds += 1

        if last_bad is None: return (sign * last_ok, True, iv)
        low, high = last_ok, last_bad

    for _ in range(int(refine_bits)):
        if stop_event and stop_event.is_set(): return (sign * (0.5*(low+high)), True, iv)
        mid = 0.5 * (low + high)
        v = _guided_pulse(sign, mid, pulse_width_s, pulse_settle_s, cooldown_s, collect_iv, iv, stop_event, temp_guard)
        if v is None: return (sign * (0.5*(low+high)), True, iv)
        if abs(v) >= v_thresh: high = mid
        else: low = mid
    return (sign * 0.5*(low+high), False, iv)


# --- Band/IV-Refinements ---
def _refine_band_below_ic(sign, ic_uA, frac_low, npts, top_margin,
                          start_uA, max_uA, pulse_width_s, pulse_settle_s, cooldown_s,
                          collect_iv, stop_event, temp_guard=None):
    """Refine knapp unter Ic; garantiert mindestens einen Punkt < ~Vthresh."""
    if ic_uA is None or ic_uA <= 0 or npts <= 1:
        return []
    lo_uA = max(start_uA, ic_uA * float(frac_low))
    hi_uA = max(start_uA, ic_uA - float(top_margin))
    if hi_uA <= lo_uA:
        return []

    currents = np.linspace(lo_uA, hi_uA, int(npts)) * float(sign)
    iv_list = []

    for I in currents:
        if stop_event.is_set():
            break
        v = _junction_pulse_measure_v_guarded(I, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
        if v is None:
            break
        if collect_iv:
            iv_list.append((I, v))

    # Unterkante prüfen: wirklich unter Schwelle?
    if collect_iv and len(iv_list) > 0:
        Imin, Vmin = iv_list[0]
        vthresh = float(DEFAULTS.get("v_thresh", 2e-4))
        if not (np.isfinite(Vmin) and abs(Vmin) < 0.9 * vthresh):
            extra_I = Imin * 0.8
            if abs(extra_I) >= start_uA:
                v_extra = _junction_pulse_measure_v_guarded(extra_I, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
                if v_extra is None:
                    return iv_list
                iv_list.insert(0, (extra_I, v_extra))

    return iv_list


def _iv0_refine_around_ic(sign, Ic_uA, pts, above_frac, window_frac,
                          start_uA, max_uA, pulse_width_s, pulse_settle_s, cooldown_s,
                          *, collect_iv=True, stop_event=None, temp_guard=None):
    if Ic_uA is None or Ic_uA <= 0: return []
    Ic = float(Ic_uA)
    lo = max(start_uA, Ic * (1.0 - float(window_frac)))
    hi = min(max_uA,  Ic * (1.0 + float(window_frac)))
    pts = max(4, int(pts))
    n_above = max(1, int(round(pts * float(above_frac))))
    n_below = max(3, pts - n_above)
    below = np.linspace(lo, Ic, n_below, endpoint=False)
    above = np.linspace(Ic, hi, n_above, endpoint=True)
    grid  = np.concatenate([below, above]); out = []
    for IuA in grid:
        if stop_event and stop_event.is_set(): break
        v = _junction_pulse_measure_v_guarded(sign * IuA, pulse_width_s, pulse_settle_s, cooldown_s, temp_guard)
        if v is None:
            break
        if collect_iv: out.append((sign * IuA, v))
    return out


# --- UI Frames (Settings) ---
def _add_common_entries_with_start(frame, row0=0):
    e = frame._entries
    tk.Label(frame, text="Start coil (mA)").grid(row=row0, column=0, padx=4, pady=2, sticky="e")
    e0s = tk.Entry(frame, width=10); e0s.insert(0, str(DEFAULTS["coil_start_mA"]))
    e0s.grid(row=row0, column=1); e["coil_start_mA"] = e0s
    tk.Label(frame, text="Target T (K)").grid(row=row0, column=2, padx=4, pady=2, sticky="e")
    e0t = tk.Entry(frame, width=10); e0t.insert(0, str(DEFAULTS["target_temp_K"]))
    e0t.grid(row=row0, column=3); e["target_temp_K"] = e0t

def _parse_target_temp(raw_value):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if text == "" or text.lower() in {"auto", "current", "now"}:
        return None
    return float(text)

def create_auto_fraunhofer_frame(parent):
    frame = tk.LabelFrame(parent, text="Automatic Fraunhofer (Log Sweep) Settings")
    frame._entries = {}
    # Row 0: Startstrom
    _add_common_entries_with_start(frame, row0=0)
    # Row 1: coil sweep (log tiers)
    tk.Label(frame, text="Coil Max (mA)").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    e0 = tk.Entry(frame, width=10); e0.insert(0, str(DEFAULTS["coil_max_mA"]))
    e0.grid(row=1, column=1); frame._entries["coil_max_mA"] = e0
    tk.Label(frame, text="Log tier counts (5)").grid(row=1, column=2, padx=4, pady=2, sticky="e")
    e1a = tk.Entry(frame, width=18); e1a.insert(0, ",".join(str(x) for x in DEFAULTS["log_tier_counts"]))
    e1a.grid(row=1, column=3); frame._entries["log_tier_counts"] = e1a
    # Row 2: timing and threshold
    tk.Label(frame, text="NPLC").grid(row=2, column=0, padx=4, pady=2, sticky="e")
    e2 = tk.Entry(frame, width=10); e2.insert(0, str(DEFAULTS["nplc"]))
    e2.grid(row=2, column=1); frame._entries["nplc"] = e2
    tk.Label(frame, text="V_thresh (V)").grid(row=2, column=2, padx=4, pady=2, sticky="e")
    e3 = tk.Entry(frame, width=10); e3.insert(0, f"{DEFAULTS['v_thresh']:.1e}")
    e3.grid(row=2, column=3); frame._entries["v_thresh"] = e3
    # Row 3: pulsing
    tk.Label(frame, text="Pulse width (ms)").grid(row=3, column=0, padx=4, pady=2, sticky="e")
    e4 = tk.Entry(frame, width=10); e4.insert(0, str(DEFAULTS["pulse_width_ms"]))
    e4.grid(row=3, column=1); frame._entries["pulse_width_ms"] = e4
    tk.Label(frame, text="Pulse settle (ms)").grid(row=3, column=2, padx=4, pady=2, sticky="e")
    e5 = tk.Entry(frame, width=10); e5.insert(0, str(DEFAULTS["pulse_settle_ms"]))
    e5.grid(row=3, column=3); frame._entries["pulse_settle_ms"] = e5
    tk.Label(frame, text="Cooldown (ms)").grid(row=3, column=4, padx=4, pady=2, sticky="e")
    e5c = tk.Entry(frame, width=10); e5c.insert(0, str(DEFAULTS["cooldown_ms"]))
    e5c.grid(row=3, column=5); frame._entries["cooldown_ms"] = e5c
    # Row 4: guided search params
    tk.Label(frame, text="Ic start (µA)").grid(row=4, column=0, padx=4, pady=2, sticky="e")
    e6 = tk.Entry(frame, width=10); e6.insert(0, str(DEFAULTS["ic_start_uA"]))
    e6.grid(row=4, column=1); frame._entries["ic_start_uA"] = e6
    tk.Label(frame, text="Growth ×").grid(row=4, column=2, padx=4, pady=2, sticky="e")
    e7 = tk.Entry(frame, width=10); e7.insert(0, str(DEFAULTS["ic_growth"]))
    e7.grid(row=4, column=3); frame._entries["ic_growth"] = e7
    tk.Label(frame, text="Ic max (µA)").grid(row=4, column=4, padx=4, pady=2, sticky="e")
    e8 = tk.Entry(frame, width=10); e8.insert(0, str(DEFAULTS["ic_max_uA"]))
    e8.grid(row=4, column=5); frame._entries["ic_max_uA"] = e8
    tk.Label(frame, text="Refine bits").grid(row=4, column=6, padx=4, pady=2, sticky="e")
    e9 = tk.Entry(frame, width=10); e9.insert(0, str(DEFAULTS["refine_bits"]))
    e9.grid(row=4, column=7); frame._entries["refine_bits"] = e9
    # Row 5: robustness / min-Ic
    tk.Label(frame, text="Switch confirm (N)").grid(row=5, column=0, padx=4, pady=2, sticky="e")
    e9b = tk.Entry(frame, width=10); e9b.insert(0, str(DEFAULTS["switch_confirm"]))
    e9b.grid(row=5, column=1); frame._entries["switch_confirm"] = e9b
    tk.Label(frame, text="Ic0 min (µA)").grid(row=5, column=2, padx=4, pady=2, sticky="e")
    e9c = tk.Entry(frame, width=10); e9c.insert(0, str(DEFAULTS["ic0_min_uA"]))
    e9c.grid(row=5, column=3); frame._entries["ic0_min_uA"] = e9c
    # Row 6: coil & temperature guard
    tk.Label(frame, text="Coil settle (ms)").grid(row=6, column=0, padx=4, pady=2, sticky="e")
    e10 = tk.Entry(frame, width=10); e10.insert(0, str(DEFAULTS["coil_settle_ms"]))
    e10.grid(row=6, column=1); frame._entries["coil_settle_ms"] = e10
    tk.Label(frame, text="Coil ramp steps").grid(row=6, column=2, padx=4, pady=2, sticky="e")
    e11 = tk.Entry(frame, width=10); e11.insert(0, str(DEFAULTS["coil_ramp_steps"]))
    e11.grid(row=6, column=3); frame._entries["coil_ramp_steps"] = e11
    tk.Label(frame, text="ΔT stop (K)").grid(row=6, column=4, padx=4, pady=2, sticky="e")
    e12 = tk.Entry(frame, width=10); e12.insert(0, str(DEFAULTS["dT_stop"]))
    e12.grid(row=6, column=5); frame._entries["dT_stop"] = e12
    tk.Label(frame, text="ΔT resume (K)").grid(row=6, column=6, padx=4, pady=2, sticky="e")
    e13 = tk.Entry(frame, width=10); e13.insert(0, str(DEFAULTS["dT_resume"]))
    e13.grid(row=6, column=7); frame._entries["dT_resume"] = e13
    # Row 7: ADR timers & counter
    tk.Label(frame, text="Cooldown total (min)").grid(row=7, column=0, padx=4, pady=2, sticky="e")
    e14 = tk.Entry(frame, width=10); e14.insert(0, str(DEFAULTS["cooldown_total_min"]))
    e14.grid(row=7, column=1); frame._entries["cooldown_total_min"] = e14
    tk.Label(frame, text="ADR trigger (min)").grid(row=7, column=2, padx=4, pady=2, sticky="e")
    e15 = tk.Entry(frame, width=10); e15.insert(0, str(DEFAULTS["adr_trigger_min"]))
    e15.grid(row=7, column=3); frame._entries["adr_trigger_min"] = e15
    tk.Label(frame, text="ADR settle (K)").grid(row=7, column=4, padx=4, pady=2, sticky="e")
    e16 = tk.Entry(frame, width=10); e16.insert(0, str(DEFAULTS["adr_settle_K"]))
    e16.grid(row=7, column=5); frame._entries["adr_settle_K"] = e16
    tk.Label(frame, text="Post-ADR settle (min)").grid(row=7, column=6, padx=4, pady=2, sticky="e")
    e17 = tk.Entry(frame, width=12); e17.insert(0, str(DEFAULTS["post_adr_settle_min"]))
    e17.grid(row=7, column=7); frame._entries["post_adr_settle_min"] = e17
    tk.Label(frame, text="ADR max resets").grid(row=8, column=0, padx=4, pady=2, sticky="e")
    e18 = tk.Entry(frame, width=10); e18.insert(0, str(DEFAULTS["adr_max_resets"]))
    e18.grid(row=8, column=1); frame._entries["adr_max_resets"] = e18
    tk.Label(frame, text="Refine if |Icoil| ≤ (µA)").grid(row=8, column=2, padx=4, pady=2, sticky="e")
    e19 = tk.Entry(frame, width=10); e19.insert(0, str(DEFAULTS["band_refine_coil_uA_max"]))
    e19.grid(row=8, column=3); frame._entries["band_refine_coil_uA_max"] = e19
    neg_var = tk.BooleanVar(value=DEFAULTS["measure_negative"])
    cb = tk.Checkbutton(frame, text="Measure negative Ic", variable=neg_var)
    cb.grid(row=9, column=0, columnspan=2, padx=4, pady=4, sticky="w")
    frame._neg_var = neg_var
    return frame

def create_linear_frame(parent):
    frame = tk.LabelFrame(parent, text="Magnetic Measurement (Linear) Settings")
    frame._entries = {}
    _add_common_entries_with_start(frame, row0=0)
    tk.Label(frame, text="Coil Max (mA)").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    e0 = tk.Entry(frame, width=10); e0.insert(0, str(DEFAULTS["coil_max_mA"]))
    e0.grid(row=1, column=1); frame._entries["coil_max_mA"] = e0
    tk.Label(frame, text="Coil Step (mA)").grid(row=1, column=2, padx=4, pady=2, sticky="e")
    e1 = tk.Entry(frame, width=10); e1.insert(0, str(DEFAULTS["coil_step_mA"]))
    e1.grid(row=1, column=3); frame._entries["coil_step_mA"] = e1
    tk.Label(frame, text="NPLC").grid(row=2, column=0, padx=4, pady=2, sticky="e")
    e2 = tk.Entry(frame, width=10); e2.insert(0, str(DEFAULTS["nplc"]))
    e2.grid(row=2, column=1); frame._entries["nplc"] = e2
    tk.Label(frame, text="V_thresh (V)").grid(row=2, column=2, padx=4, pady=2, sticky="e")
    e3 = tk.Entry(frame, width=10); e3.insert(0, f"{DEFAULTS['v_thresh']:.1e}")
    e3.grid(row=2, column=3); frame._entries["v_thresh"] = e3
    tk.Label(frame, text="Pulse width (ms)").grid(row=3, column=0, padx=4, pady=2, sticky="e")
    e4 = tk.Entry(frame, width=10); e4.insert(0, str(DEFAULTS["pulse_width_ms"]))
    e4.grid(row=3, column=1); frame._entries["pulse_width_ms"] = e4
    tk.Label(frame, text="Pulse settle (ms)").grid(row=3, column=2, padx=4, pady=2, sticky="e")
    e5 = tk.Entry(frame, width=10); e5.insert(0, str(DEFAULTS["pulse_settle_ms"]))
    e5.grid(row=3, column=3); frame._entries["pulse_settle_ms"] = e5
    tk.Label(frame, text="Cooldown (ms)").grid(row=3, column=4, padx=4, pady=2, sticky="e")
    e6 = tk.Entry(frame, width=10); e6.insert(0, str(DEFAULTS["cooldown_ms"]))
    e6.grid(row=3, column=5); frame._entries["cooldown_ms"] = e6
    tk.Label(frame, text="Ic start (µA)").grid(row=4, column=0, padx=4, pady=2, sticky="e")
    e7 = tk.Entry(frame, width=10); e7.insert(0, str(DEFAULTS["ic_start_uA"]))
    e7.grid(row=4, column=1); frame._entries["ic_start_uA"] = e7
    tk.Label(frame, text="Growth ×").grid(row=4, column=2, padx=4, pady=2, sticky="e")
    e8 = tk.Entry(frame, width=10); e8.insert(0, str(DEFAULTS["ic_growth"]))
    e8.grid(row=4, column=3); frame._entries["ic_growth"] = e8
    tk.Label(frame, text="Ic max (µA)").grid(row=4, column=4, padx=4, pady=2, sticky="e")
    e9 = tk.Entry(frame, width=10); e9.insert(0, str(DEFAULTS["ic_max_uA"]))
    e9.grid(row=4, column=5); frame._entries["ic_max_uA"] = e9
    tk.Label(frame, text="Refine bits").grid(row=4, column=6, padx=4, pady=2, sticky="e")
    e10 = tk.Entry(frame, width=10); e10.insert(0, str(DEFAULTS["refine_bits"]))
    e10.grid(row=4, column=7); frame._entries["refine_bits"] = e10
    tk.Label(frame, text="Switch confirm (N)").grid(row=5, column=0, padx=4, pady=2, sticky="e")
    e11 = tk.Entry(frame, width=10); e11.insert(0, str(DEFAULTS["switch_confirm"]))
    e11.grid(row=5, column=1); frame._entries["switch_confirm"] = e11
    tk.Label(frame, text="Ic0 min (µA, nur B=0)").grid(row=5, column=2, padx=4, pady=2, sticky="e")
    e12 = tk.Entry(frame, width=10); e12.insert(0, str(DEFAULTS["ic0_min_uA"]))
    e12.grid(row=5, column=3); frame._entries["ic0_min_uA"] = e12
    tk.Label(frame, text="Coil settle (ms)").grid(row=6, column=0, padx=4, pady=2, sticky="e")
    e13 = tk.Entry(frame, width=10); e13.insert(0, str(DEFAULTS["coil_settle_ms"]))
    e13.grid(row=6, column=1); frame._entries["coil_settle_ms"] = e13
    tk.Label(frame, text="Coil ramp steps").grid(row=6, column=2, padx=4, pady=2, sticky="e")
    e14 = tk.Entry(frame, width=10); e14.insert(0, str(DEFAULTS["coil_ramp_steps"]))
    e14.grid(row=6, column=3); frame._entries["coil_ramp_steps"] = e14
    tk.Label(frame, text="ΔT stop (K)").grid(row=6, column=4, padx=4, pady=2, sticky="e")
    e15 = tk.Entry(frame, width=10); e15.insert(0, str(DEFAULTS["dT_stop"]))
    e15.grid(row=6, column=5); frame._entries["dT_stop"] = e15
    tk.Label(frame, text="ΔT resume (K)").grid(row=6, column=6, padx=4, pady=2, sticky="e")
    e16 = tk.Entry(frame, width=10); e16.insert(0, str(DEFAULTS["dT_resume"]))
    e16.grid(row=6, column=7); frame._entries["dT_resume"] = e16
    tk.Label(frame, text="Cooldown total (min)").grid(row=7, column=0, padx=4, pady=2, sticky="e")
    e17 = tk.Entry(frame, width=10); e17.insert(0, str(DEFAULTS["cooldown_total_min"]))
    e17.grid(row=7, column=1); frame._entries["cooldown_total_min"] = e17
    tk.Label(frame, text="ADR trigger (min)").grid(row=7, column=2, padx=4, pady=2, sticky="e")
    e18 = tk.Entry(frame, width=10); e18.insert(0, str(DEFAULTS["adr_trigger_min"]))
    e18.grid(row=7, column=3); frame._entries["adr_trigger_min"] = e18
    tk.Label(frame, text="ADR settle (K)").grid(row=7, column=4, padx=4, pady=2, sticky="e")
    e19 = tk.Entry(frame, width=10); e19.insert(0, str(DEFAULTS["adr_settle_K"]))
    e19.grid(row=7, column=5); frame._entries["adr_settle_K"] = e19
    tk.Label(frame, text="Post-ADR settle (min)").grid(row=7, column=6, padx=4, pady=2, sticky="e")
    e20 = tk.Entry(frame, width=10); e20.insert(0, str(DEFAULTS["post_adr_settle_min"]))
    e20.grid(row=7, column=7); frame._entries["post_adr_settle_min"] = e20
    tk.Label(frame, text="ADR max resets").grid(row=8, column=0, padx=4, pady=2, sticky="e")
    e21 = tk.Entry(frame, width=10); e21.insert(0, str(DEFAULTS["adr_max_resets"]))
    e21.grid(row=8, column=1); frame._entries["adr_max_resets"] = e21
    tk.Label(frame, text="Refine if |Icoil| ≤ (µA)").grid(row=8, column=2, padx=4, pady=2, sticky="e")
    e22 = tk.Entry(frame, width=10); e22.insert(0, str(DEFAULTS["band_refine_coil_uA_max"]))
    e22.grid(row=8, column=3); frame._entries["band_refine_coil_uA_max"] = e22
    neg_var = tk.BooleanVar(value=DEFAULTS["measure_negative"])
    cb = tk.Checkbutton(frame, text="Measure negative Ic", variable=neg_var)
    cb.grid(row=9, column=0, columnspan=2, padx=4, pady=4, sticky="w")
    frame._neg_var = neg_var
    return frame

def create_roughscan_frame(parent):
    frame = tk.LabelFrame(parent, text="Rough Scan Settings (fast Fraunhofer search)")
    frame._entries = {}
    _add_common_entries_with_start(frame, row0=0)
    tk.Label(frame, text="Max (mA)").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    e0 = tk.Entry(frame, width=10); e0.insert(0, str(DEFAULTS["rough_max_mA"]))
    e0.grid(row=1, column=1); frame._entries["rough_max_mA"] = e0
    tk.Label(frame, text="Step (mA)").grid(row=1, column=2, padx=4, pady=2, sticky="e")
    e1 = tk.Entry(frame, width=10); e1.insert(0, str(DEFAULTS["rough_step_mA"]))
    e1.grid(row=1, column=3); frame._entries["rough_step_mA"] = e1
    tk.Label(frame, text="NPLC").grid(row=2, column=0, padx=4, pady=2, sticky="e")
    e2 = tk.Entry(frame, width=10); e2.insert(0, str(DEFAULTS["nplc"]))
    e2.grid(row=2, column=1); frame._entries["nplc"] = e2
    tk.Label(frame, text="V_thresh (V)").grid(row=2, column=2, padx=4, pady=2, sticky="e")
    e3 = tk.Entry(frame, width=10); e3.insert(0, f"{DEFAULTS['v_thresh']:.1e}")
    e3.grid(row=2, column=3); frame._entries["v_thresh"] = e3
    tk.Label(frame, text="Pulse width (ms)").grid(row=3, column=0, padx=4, pady=2, sticky="e")
    e4 = tk.Entry(frame, width=10); e4.insert(0, str(DEFAULTS["pulse_width_ms"]))
    e4.grid(row=3, column=1); frame._entries["pulse_width_ms"] = e4
    tk.Label(frame, text="Pulse settle (ms)").grid(row=3, column=2, padx=4, pady=2, sticky="e")
    e5 = tk.Entry(frame, width=10); e5.insert(0, str(DEFAULTS["pulse_settle_ms"]))
    e5.grid(row=3, column=3); frame._entries["pulse_settle_ms"] = e5
    tk.Label(frame, text="Cooldown (ms)").grid(row=3, column=4, padx=4, pady=2, sticky="e")
    e6 = tk.Entry(frame, width=10); e6.insert(0, str(DEFAULTS["cooldown_ms"]))
    e6.grid(row=3, column=5); frame._entries["cooldown_ms"] = e6
    tk.Label(frame, text="Rough refine bits").grid(row=4, column=0, padx=4, pady=2, sticky="e")
    e7 = tk.Entry(frame, width=10); e7.insert(0, str(DEFAULTS["rough_refine_bits"]))
    e7.grid(row=4, column=1); frame._entries["rough_refine_bits"] = e7
    tk.Label(frame, text="Smooth pts").grid(row=4, column=2, padx=4, pady=2, sticky="e")
    e8 = tk.Entry(frame, width=10); e8.insert(0, str(DEFAULTS["rough_smooth_pts"]))
    e8.grid(row=4, column=3); frame._entries["rough_smooth_pts"] = e8
    tk.Label(frame, text="Min prominence").grid(row=4, column=4, padx=4, pady=2, sticky="e")
    e9 = tk.Entry(frame, width=10); e9.insert(0, str(DEFAULTS["rough_min_prominence"]))
    e9.grid(row=4, column=5); frame._entries["rough_min_prominence"] = e9
    return frame


#Bloxk3und4hiereinfügen
# --- Backward-compat alias (falls alter Name irgendwo noch verwendet wird) ---
def drain_errorperiodic(every_pulses):
    # ruft die neue Schreibweise auf
    return _drain_errors_periodic(every_pulses)
#%% BLOCK 3/4 — UI (LOG, LINEAR, ROUGH): Fraunhofer_Pattern(), MagneticMeasurementLinear(), RoughScan()

def _make_common_progress(title):
    prog = tk.Toplevel(root); prog.title(title)
    bar = ttk.Progressbar(prog, maximum=100, length=360); bar.pack(padx=10, pady=6)
    status_lbl = tk.Label(prog, text="Idle."); status_lbl.pack(padx=10, pady=4)
    return prog, bar, status_lbl

def _start_stop_buttons(parent):
    controls = tk.Frame(parent); controls.pack(fill=tk.X, padx=8, pady=8)
    start_btn = tk.Button(controls, text="Start", width=10)
    stop_btn  = tk.Button(controls, text="Stop",  width=10, state=tk.DISABLED)
    start_btn.pack(side=tk.RIGHT, padx=4); stop_btn.pack(side=tk.RIGHT, padx=4)
    return start_btn, stop_btn

def _poll_queues(state, status_lbl, bar, ic_line_pos, ic_line_neg, ic_ax, ic_canvas,
                 iv_line=None, iv_ax=None, iv_canvas=None, iv_line_latest=None):
    try:
        while True:
            msg = state.progress_q.get_nowait()
            tag = msg[0]

            if tag == 'status':
                status_lbl.config(text=msg[1])

            elif tag == 'progress':
                i, total = msg[1], msg[2]
                bar['maximum'] = max(1, total)
                bar['value'] = max(0, min(i, total))

            elif tag == 'point':
                x, yp, yn = msg[1], msg[2], msg[3]
                state.coil_points_A.append(x); state.icp_points_A.append(yp)
                if yn is not None:
                    state.icn_points_A.append(yn)

                ic_line_pos.set_data(state.coil_points_A, state.icp_points_A)

                if state.settings.get("measure_negative", False) and len(state.icn_points_A):
                    n = min(len(state.coil_points_A), len(state.icn_points_A))
                    ic_line_neg.set_data(state.coil_points_A[:n], state.icn_points_A[:n])

                ic_ax.relim(); ic_ax.autoscale_view()
                ic_canvas.draw_idle()

            elif tag == 'iv_start' and iv_line is not None and iv_ax is not None and iv_canvas is not None:
                state.iv_curve_index += 1
                state.iv_current_points = []
                if state.iv_curve_index == 1:
                    state.iv_first_points = state.iv_current_points
                    state.iv_latest_points = state.iv_current_points
                else:
                    state.iv_latest_points = state.iv_current_points

            elif tag == 'iv' and iv_line is not None and iv_ax is not None and iv_canvas is not None:
                I, V = msg[1], msg[2]
                if iv_line_latest is None:
                    state.iv_points.append((I, V))
                    if len(state.iv_points) >= 2:
                        xs = [p[0] for p in state.iv_points]
                        ys = [p[1] for p in state.iv_points]
                        iv_line.set_data(xs, ys)
                        iv_ax.relim(); iv_ax.autoscale_view()
                        iv_canvas.draw_idle()
                else:
                    if state.iv_current_points is None:
                        state.iv_curve_index += 1
                        state.iv_current_points = []
                        state.iv_first_points = state.iv_current_points
                        state.iv_latest_points = state.iv_current_points
                    state.iv_current_points.append((I, V))
                    if len(state.iv_first_points) >= 2:
                        xs = [p[0] for p in state.iv_first_points]
                        ys = [p[1] for p in state.iv_first_points]
                        iv_line.set_data(xs, ys)
                    if len(state.iv_latest_points) >= 2 and iv_line_latest is not None:
                        xs = [p[0] for p in state.iv_latest_points]
                        ys = [p[1] for p in state.iv_latest_points]
                        iv_line_latest.set_data(xs, ys)
                    iv_ax.relim(); iv_ax.autoscale_view()
                    iv_canvas.draw_idle()

            elif tag == 'error':
                messagebox.showerror("Error", msg[1])

            elif tag == 'done':
                status_lbl.config(text=msg[1])
                # Fortschrittsbalken „voll“
                bar['value'] = bar['maximum']

            state.progress_q.task_done()

    except queue.Empty:
        pass

    # Solange der Thread läuft, weiter pollen
    if state and state.thread and state.thread.is_alive():
        root.after(100, lambda: _poll_queues(state, status_lbl, bar, ic_line_pos, ic_line_neg,
                                             ic_ax, ic_canvas, iv_line, iv_ax, iv_canvas, iv_line_latest))

@dataclass
class FFUI:
    frame: object
    start_btn: object
    stop_btn: object
    status_lbl: object
    bar: object
    ic_line_pos: object
    ic_line_neg: object
    ic_ax: object
    ic_canvas: object
    iv_line: object
    iv_ax: object
    iv_canvas: object
    iv_line_latest: object = None
    step_entry: object = None
    jump_entry: object = None

def _fraunhofer_start_clicked(ui: FFUI):
    base = simpledialog.askstring("Save name", "Enter a safe base filename (no extension):")
    if not base:
        messagebox.showerror("Error", "Please provide a base filename.")
        return
    base = _dated_base(base)

    e = ui.frame._entries
    try:
        target_temp = _parse_target_temp(e["target_temp_K"].get())
        if target_temp is None:
            target_temp = _get_T()
        settings = dict(
            coil_start_mA     = float(e["coil_start_mA"].get()),
            target_temp_K     = float(target_temp),
            coil_max_mA       = float(e["coil_max_mA"].get()),
            log_tier_counts   = _parse_counts(e["log_tier_counts"].get(), DEFAULTS["log_tier_counts"]),
            nplc              = float(e["nplc"].get()),
            v_thresh          = float(e["v_thresh"].get()),
            pulse_width_s     = float(e["pulse_width_ms"].get())  * 1e-3,
            pulse_settle_s    = float(e["pulse_settle_ms"].get()) * 1e-3,
            cooldown_s        = float(e["cooldown_ms"].get())     * 1e-3,
            ic_start_uA       = float(e["ic_start_uA"].get()),
            ic_growth         = float(e["ic_growth"].get()),
            ic_max_uA         = float(e["ic_max_uA"].get()),
            refine_bits       = int(e["refine_bits"].get()),
            switch_confirm    = int(e["switch_confirm"].get()),
            ic0_min_uA        = float(e["ic0_min_uA"].get()),
            coil_settle_s     = float(e["coil_settle_ms"].get())  * 1e-3,
            coil_ramp_steps   = int(e["coil_ramp_steps"].get()),
            dT_stop           = float(e["dT_stop"].get()),
            dT_resume         = float(e["dT_resume"].get()),
            cooldown_total_s  = float(e["cooldown_total_min"].get())*60.0,
            adr_trigger_s     = float(e["adr_trigger_min"].get()) *60.0,
            adr_settle_K      = float(e["adr_settle_K"].get()),
            post_adr_settle_s = float(e["post_adr_settle_min"].get())*60.0,
            adr_max_resets    = int(e["adr_max_resets"].get()),
            band_refine_coil_uA_max = float(e["band_refine_coil_uA_max"].get()),
            measure_negative  = bool(ui.frame._neg_var.get()),
            guided_lower_frac = float(DEFAULTS["guided_lower_frac"]),
            guided_step_rel   = float(DEFAULTS["guided_step_rel"]),
            guided_min_step_uA= float(DEFAULTS["guided_min_step_uA"]),
            guided_window_frac= float(DEFAULTS["guided_window_frac"]),
            iv0_refine        = bool(DEFAULTS["iv0_refine"]),
            iv0_pts_per_branch= int(DEFAULTS["iv0_pts_per_branch"]),
            iv0_above_frac    = float(DEFAULTS["iv0_above_frac"]),
            iv0_window_frac   = float(DEFAULTS["iv0_window_frac"]),
            band_refine_enable   = bool(DEFAULTS["band_refine_enable"]),
            band_refine_frac_low = float(DEFAULTS["band_refine_frac_low"]),
            band_refine_pts      = int(DEFAULTS["band_refine_pts"]),
            band_refine_top_margin=float(DEFAULTS["band_refine_top_margin"]),
            error_drain_every_pulses = int(DEFAULTS["error_drain_every_pulses"]),
            log_anchor_mode   = bool(ui.frame._anchor_var.get()),
            burst_ops_before_wait = int(DEFAULTS["burst_ops_before_wait"]),
            burst_backoff_s       = float(DEFAULTS["burst_backoff_s"]),
        )
    except Exception as ex:
        messagebox.showerror("Error", f"Invalid setting: {ex}")
        return

    _ff_state.stop_event.clear()
    _ff_state.base = base
    _ff_state.settings = settings
    _ff_state.T0 = float(settings["target_temp_K"])
    _ff_state.rows_pos.clear(); _ff_state.rows_neg.clear()
    _ff_state.iv_points.clear(); _ff_state.iv_search_rows.clear()
    _ff_state.iv_first_points.clear(); _ff_state.iv_latest_points.clear()
    _ff_state.iv_current_points = None; _ff_state.iv_curve_index = 0
    _ff_state.iv_first_points.clear(); _ff_state.iv_latest_points.clear()
    _ff_state.iv_current_points = None; _ff_state.iv_curve_index = 0
    _ff_state.coil_points_A.clear(); _ff_state.icp_points_A.clear(); _ff_state.icn_points_A.clear()
    _ff_state.adr_events = 0

    global _ERR_DRAIN_EVERY, _pulse_counter
    _ERR_DRAIN_EVERY = int(settings.get("error_drain_every_pulses", 200))
    _pulse_counter = 0

    ui.start_btn.config(state=tk.DISABLED)
    ui.stop_btn.config(state=tk.NORMAL)
    ui.status_lbl.config(text=f"Starting… Ttarget = {_ff_state.T0:.3f} K")
    _ff_state.thread = threading.Thread(target=_worker_run_log, args=(_ff_state,), daemon=True)
    _ff_state.thread.start()
    _poll_queues(_ff_state, ui.status_lbl, ui.bar, ui.ic_line_pos, ui.ic_line_neg, ui.ic_ax,
                 ui.ic_canvas, ui.iv_line, ui.iv_ax, ui.iv_canvas, ui.iv_line_latest)

def _fraunhofer_stop_clicked(ui: FFUI):
    if _ff_state and _ff_state.thread and _ff_state.thread.is_alive():
        _ff_state.stop_event.set()
        ui.status_lbl.config(text="Stopping… (saving partial data)")

    ui.start_btn.config(state=tk.NORMAL)
    ui.stop_btn.config(state=tk.DISABLED)

def _linear_apply_step(ui: FFUI):
    try:
        val = float(ui.step_entry.get())
        if _ff_state and _ff_state.thread and _ff_state.thread.is_alive():
            _ff_state.command_q.put({"op": "set_step", "step_mA": val})
            ui.status_lbl.config(text=f"Live: new step {val} mA queued")
    except Exception as ex:
        messagebox.showerror("Step error", str(ex))

def _linear_do_jump(ui: FFUI):
    try:
        val = float(ui.jump_entry.get())
        if _ff_state and _ff_state.thread and _ff_state.thread.is_alive():
            _ff_state.command_q.put({"op": "jump", "target_mA": val})
            ui.status_lbl.config(text=f"Live: jump to {val} mA queued")
    except Exception as ex:
        messagebox.showerror("Jump error", str(ex))

def _linear_start_clicked(ui: FFUI):
    base = simpledialog.askstring("Save name", "Enter a safe base filename (no extension):")
    if not base:
        messagebox.showerror("Error", "Please provide a base filename.")
        return
    base = _dated_base(base)

    e = ui.frame._entries
    try:
        target_temp = _parse_target_temp(e["target_temp_K"].get())
        if target_temp is None:
            target_temp = _get_T()
        settings = dict(
            coil_start_mA     = float(e["coil_start_mA"].get()),
            target_temp_K     = float(target_temp),
            coil_max_mA       = float(e["coil_max_mA"].get()),
            coil_step_mA      = float(e["coil_step_mA"].get()),
            nplc              = float(e["nplc"].get()),
            v_thresh          = float(e["v_thresh"].get()),
            pulse_width_s     = float(e["pulse_width_ms"].get())  * 1e-3,
            pulse_settle_s    = float(e["pulse_settle_ms"].get()) * 1e-3,
            cooldown_s        = float(e["cooldown_ms"].get())     * 1e-3,
            ic_start_uA       = float(e["ic_start_uA"].get()),
            ic_growth         = float(e["ic_growth"].get()),
            ic_max_uA         = float(e["ic_max_uA"].get()),
            refine_bits       = int(e["refine_bits"].get()),
            switch_confirm    = int(e["switch_confirm"].get()),
            ic0_min_uA        = float(e["ic0_min_uA"].get()),
            coil_settle_s     = float(e["coil_settle_ms"].get())  * 1e-3,
            coil_ramp_steps   = int(e["coil_ramp_steps"].get()),
            dT_stop           = float(e["dT_stop"].get()),
            dT_resume         = float(e["dT_resume"].get()),
            cooldown_total_s  = float(e["cooldown_total_min"].get())*60.0,
            adr_trigger_s     = float(e["adr_trigger_min"].get()) *60.0,
            adr_settle_K      = float(e["adr_settle_K"].get()),
            post_adr_settle_s = float(e["post_adr_settle_min"].get())*60.0,
            adr_max_resets    = int(e["adr_max_resets"].get()),
            band_refine_coil_uA_max = float(e["band_refine_coil_uA_max"].get()),
            measure_negative  = bool(ui.frame._neg_var.get()),
            guided_lower_frac = float(DEFAULTS["guided_lower_frac"]),
            guided_step_rel   = float(DEFAULTS["guided_step_rel"]),
            guided_min_step_uA= float(DEFAULTS["guided_min_step_uA"]),
            guided_window_frac= float(DEFAULTS["guided_window_frac"]),
            iv0_refine        = bool(DEFAULTS["iv0_refine"]),
            iv0_pts_per_branch= int(DEFAULTS["iv0_pts_per_branch"]),
            iv0_above_frac    = float(DEFAULTS["iv0_above_frac"]),
            iv0_window_frac   = float(DEFAULTS["iv0_window_frac"]),
            band_refine_enable   = bool(DEFAULTS["band_refine_enable"]),
            band_refine_frac_low = float(DEFAULTS["band_refine_frac_low"]),
            band_refine_pts      = int(DEFAULTS["band_refine_pts"]),
            band_refine_top_margin=float(DEFAULTS["band_refine_top_margin"]),
            error_drain_every_pulses = int(DEFAULTS["error_drain_every_pulses"]),
            burst_ops_before_wait = int(DEFAULTS["burst_ops_before_wait"]),
            burst_backoff_s       = float(DEFAULTS["burst_backoff_s"]),
        )
    except Exception as ex:
        messagebox.showerror("Error", f"Invalid setting: {ex}")
        return

    _ff_state.stop_event.clear()
    _ff_state.base = base
    _ff_state.settings = settings
    _ff_state.T0 = float(settings["target_temp_K"])
    _ff_state.rows_pos.clear(); _ff_state.rows_neg.clear()
    _ff_state.iv_points.clear(); _ff_state.iv_search_rows.clear()
    _ff_state.iv_first_points.clear(); _ff_state.iv_latest_points.clear()
    _ff_state.iv_current_points = None; _ff_state.iv_curve_index = 0
    _ff_state.coil_points_A.clear(); _ff_state.icp_points_A.clear(); _ff_state.icn_points_A.clear()
    _ff_state.adr_events = 0

    global _ERR_DRAIN_EVERY, _pulse_counter
    _ERR_DRAIN_EVERY = int(settings.get("error_drain_every_pulses", 200))
    _pulse_counter = 0

    ui.start_btn.config(state=tk.DISABLED)
    ui.stop_btn.config(state=tk.NORMAL)
    ui.status_lbl.config(text=f"Starting (linear)… Ttarget = {_ff_state.T0:.3f} K")
    _ff_state.thread = threading.Thread(target=_worker_run_linear, args=(_ff_state,), daemon=True)
    _ff_state.thread.start()
    _poll_queues(_ff_state, ui.status_lbl, ui.bar, ui.ic_line_pos, ui.ic_line_neg, ui.ic_ax,
                 ui.ic_canvas, ui.iv_line, ui.iv_ax, ui.iv_canvas, ui.iv_line_latest)

def _linear_stop_clicked(ui: FFUI):
    if _ff_state and _ff_state.thread and _ff_state.thread.is_alive():
        _ff_state.stop_event.set()
        ui.status_lbl.config(text="Stopping… (saving partial data)")

    ui.start_btn.config(state=tk.NORMAL)
    ui.stop_btn.config(state=tk.DISABLED)

def _rough_start_clicked(ui: FFUI):
    base = simpledialog.askstring("Save name", "Enter a base filename (no extension) for ROUGH:")
    if not base:
        messagebox.showerror("Error", "Please provide a base filename.")
        return
    base = _dated_base(base)

    e = ui.frame._entries
    try:
        target_temp = _parse_target_temp(e["target_temp_K"].get())
        if target_temp is None:
            target_temp = _get_T()
        settings = dict(
            coil_start_mA     = float(e["coil_start_mA"].get()),
            target_temp_K     = float(target_temp),
            rough_max_mA      = float(e["rough_max_mA"].get()),
            rough_step_mA     = float(e["rough_step_mA"].get()),
            nplc              = float(e["nplc"].get()),
            v_thresh          = float(e["v_thresh"].get()),
            pulse_width_s     = float(e["pulse_width_ms"].get())  * 1e-3,
            pulse_settle_s    = float(e["pulse_settle_ms"].get()) * 1e-3,
            cooldown_s        = float(e["cooldown_ms"].get())     * 1e-3,
            ic_start_uA       = float(DEFAULTS["ic_start_uA"]),
            ic_growth         = float(DEFAULTS["ic_growth"]),
            ic_max_uA         = float(DEFAULTS["ic_max_uA"]),
            rough_refine_bits = int(e["rough_refine_bits"].get()),
            rough_smooth_pts  = int(e["rough_smooth_pts"].get()),
            rough_min_prominence = float(e["rough_min_prominence"].get()),
            coil_settle_s     = float(DEFAULTS["coil_settle_ms"]) * 1e-3,
            coil_ramp_steps   = int(DEFAULTS["coil_ramp_steps"]),
            dT_stop           = float(DEFAULTS["dT_stop"]),
            dT_resume         = float(DEFAULTS["dT_resume"]),
            cooldown_total_s  = float(DEFAULTS["cooldown_total_min"])*60.0,
            adr_trigger_s     = float(DEFAULTS["adr_trigger_min"]) *60.0,
            adr_settle_K      = float(DEFAULTS["adr_settle_K"]),
            post_adr_settle_s = float(DEFAULTS["post_adr_settle_min"])*60.0,
            adr_max_resets    = int(DEFAULTS["adr_max_resets"]),
            measure_negative  = False,
            error_drain_every_pulses = int(DEFAULTS["error_drain_every_pulses"]),
            burst_ops_before_wait = int(DEFAULTS["burst_ops_before_wait"]),
            burst_backoff_s       = float(DEFAULTS["burst_backoff_s"]),
        )
    except Exception as ex:
        messagebox.showerror("Error", f"Invalid setting: {ex}")
        return

    _ff_state.stop_event.clear()
    _ff_state.base = base
    _ff_state.settings = settings
    _ff_state.T0 = float(settings["target_temp_K"])
    _ff_state.rows_pos.clear(); _ff_state.rows_neg.clear()
    _ff_state.iv_points.clear(); _ff_state.iv_search_rows.clear()
    _ff_state.coil_points_A.clear(); _ff_state.icp_points_A.clear(); _ff_state.icn_points_A.clear()
    _ff_state.adr_events = 0

    global _ERR_DRAIN_EVERY, _pulse_counter
    _ERR_DRAIN_EVERY = int(settings.get("error_drain_every_pulses", 200))
    _pulse_counter = 0

    ui.start_btn.config(state=tk.DISABLED)
    ui.stop_btn.config(state=tk.NORMAL)
    ui.status_lbl.config(text=f"Starting (rough)… Ttarget = {_ff_state.T0:.3f} K")
    _ff_state.thread = threading.Thread(target=_worker_run_rough,
                                        args=(_ff_state, ui.ic_line_pos, ui.ic_ax, ui.ic_canvas,
                                              ui.status_lbl, ui.bar),
                                        daemon=True)
    _ff_state.thread.start()
    _poll_queues(_ff_state, ui.status_lbl, ui.bar, ui.ic_line_pos, ui.ic_line_neg, ui.ic_ax,
                 ui.ic_canvas, None, None, None)

def _rough_stop_clicked(ui: FFUI):
    if _ff_state and _ff_state.thread and _ff_state.thread.is_alive():
        _ff_state.stop_event.set()
        ui.status_lbl.config(text="Stopping… (saving partial data)")

    ui.start_btn.config(state=tk.NORMAL)
    ui.stop_btn.config(state=tk.DISABLED)

def Fraunhofer_Pattern():
    """Logarithmischer Coil-Sweep (center-out ODER Anchor ±Start), threaded UI, Liveplots."""
    closer = (lambda w: on_closing(w)) if 'on_closing' in globals() else (lambda w: w.destroy())

    win = tk.Toplevel(root); win.title("Automatic Fraunhofer Pattern (Log Sweep)")
    win.protocol("WM_DELETE_WINDOW", lambda: closer(win))

    frame = create_auto_fraunhofer_frame(win)
    frame.pack(padx=8, pady=8, fill=tk.X)

    # Checkbox für Anchor-Mode
    frame._anchor_var = tk.BooleanVar(value=False)
    tk.Checkbutton(frame,
                   text="Anchor at ±Start (instead of center-out)",
                   variable=frame._anchor_var).grid(row=99, column=0, columnspan=2, sticky="w", padx=4, pady=2)

    start_btn, stop_btn = _start_stop_buttons(win)
    prog, bar, status_lbl = _make_common_progress("Fraunhofer Progress")

    ic_win, ic_fig, ic_ax, ic_line_pos, ic_line_neg, ic_canvas = _make_plot_window(
        title="Ic vs I_coil (log, live)", xlab="I_coil (A)", ylab="Ic (A)", with_two_series=True
    )
    iv_win, iv_fig, iv_ax, iv_line, _, iv_canvas = _make_plot_window(
        title="Zero-field IV (B = 0)", xlab="I_junc (A)", ylab="V (V)", with_two_series=False
    )

    global _ff_state, _pulse_counter, _ERR_DRAIN_EVERY
    _ff_state = FFState()
    _pulse_counter = 0
    ui = FFUI(
        frame=frame,
        start_btn=start_btn,
        stop_btn=stop_btn,
        status_lbl=status_lbl,
        bar=bar,
        ic_line_pos=ic_line_pos,
        ic_line_neg=ic_line_neg,
        ic_ax=ic_ax,
        ic_canvas=ic_canvas,
        iv_line=iv_line,
        iv_ax=iv_ax,
        iv_canvas=iv_canvas,
    )

    start_btn.config(command=partial(_fraunhofer_start_clicked, ui))
    stop_btn.config(command=partial(_fraunhofer_stop_clicked, ui))
    return win


def MagneticMeasurementLinear():
    """Linear coil sweep mit Live-Step/JUMP, threaded UI, Liveplots."""
    closer = (lambda w: on_closing(w)) if 'on_closing' in globals() else (lambda w: w.destroy())

    win = tk.Toplevel(root); win.title("Magnetic Measurement — Linear Sweep")
    win.protocol("WM_DELETE_WINDOW", lambda: closer(win))

    frame = create_linear_frame(win)
    frame.pack(padx=8, pady=8, fill=tk.X)

    start_btn, stop_btn = _start_stop_buttons(win)
    prog, bar, status_lbl = _make_common_progress("Linear Sweep Progress")

    # --- Live Controls: Step ändern & Jump zu bestimmtem Strom ---
    live = tk.LabelFrame(win, text="Live controls")
    live.pack(fill=tk.X, padx=8, pady=4)

    tk.Label(live, text="New step (mA)").grid(row=0, column=0, padx=4, pady=2, sticky="e")
    step_entry = tk.Entry(live, width=10); step_entry.insert(0, str(DEFAULTS["coil_step_mA"]))
    step_entry.grid(row=0, column=1, padx=4, pady=2)
    apply_btn = tk.Button(live, text="Apply step", width=12)
    apply_btn.grid(row=0, column=2, padx=6, pady=2)

    tk.Label(live, text="Jump to (mA)").grid(row=1, column=0, padx=4, pady=2, sticky="e")
    jump_entry = tk.Entry(live, width=10); jump_entry.insert(0, "0.0")
    jump_entry.grid(row=1, column=1, padx=4, pady=2)
    jump_btn = tk.Button(live, text="Jump", width=12)
    jump_btn.grid(row=1, column=2, padx=6, pady=2)

    ic_win, ic_fig, ic_ax, ic_line_pos, ic_line_neg, ic_canvas = _make_plot_window(
        title="Ic vs I_coil (linear, live)", xlab="I_coil (A)", ylab="Ic (A)", with_two_series=True
    )
    iv_win, iv_fig, iv_ax, iv_line, iv_line_latest, iv_canvas = _make_plot_window(
        title="Zero-field IV (B = 0)", xlab="I_junc (A)", ylab="V (V)", with_two_series=True,
        label1="First curve", label2="Latest curve", color1="tab:blue", color2="tab:red"
    )

    global _ff_state, _pulse_counter, _ERR_DRAIN_EVERY
    _ff_state = FFState()
    _pulse_counter = 0
    ui = FFUI(
        frame=frame,
        start_btn=start_btn,
        stop_btn=stop_btn,
        status_lbl=status_lbl,
        bar=bar,
        ic_line_pos=ic_line_pos,
        ic_line_neg=ic_line_neg,
        ic_ax=ic_ax,
        ic_canvas=ic_canvas,
        iv_line=iv_line,
        iv_ax=iv_ax,
        iv_canvas=iv_canvas,
        iv_line_latest=iv_line_latest,
        step_entry=step_entry,
        jump_entry=jump_entry,
    )

    apply_btn.config(command=partial(_linear_apply_step, ui))
    jump_btn.config(command=partial(_linear_do_jump, ui))
    start_btn.config(command=partial(_linear_start_clicked, ui))
    stop_btn.config(command=partial(_linear_stop_clicked, ui))
    return win


def RoughScan():
    """Schneller 0..±Max mA Scan um Startstrom, findet Fraunhofer-Min/Max & Periode (nur |Ic|, +Branch)."""
    closer = (lambda w: on_closing(w)) if 'on_closing' in globals() else (lambda w: w.destroy())

    win = tk.Toplevel(root); win.title("Rough Scan — Fast Fraunhofer Search")
    win.protocol("WM_DELETE_WINDOW", lambda: closer(win))

    frame = create_roughscan_frame(win)
    frame.pack(padx=8, pady=8, fill=tk.X)

    start_btn, stop_btn = _start_stop_buttons(win)
    prog, bar, status_lbl = _make_common_progress("Rough Scan Progress")

    ic_win, ic_fig, ic_ax, ic_line_pos, ic_line_neg, ic_canvas = _make_plot_window(
        title="Ic vs I_coil (rough, live)", xlab="I_coil (A)", ylab="Ic (A)", with_two_series=False
    )

    global _ff_state, _pulse_counter, _ERR_DRAIN_EVERY
    _ff_state = FFState()
    _pulse_counter = 0
    ui = FFUI(
        frame=frame,
        start_btn=start_btn,
        stop_btn=stop_btn,
        status_lbl=status_lbl,
        bar=bar,
        ic_line_pos=ic_line_pos,
        ic_line_neg=ic_line_neg,
        ic_ax=ic_ax,
        ic_canvas=ic_canvas,
        iv_line=None,
        iv_ax=None,
        iv_canvas=None,
    )

    start_btn.config(command=partial(_rough_start_clicked, ui))
    stop_btn.config(command=partial(_rough_stop_clicked, ui))
    return win


#%% BLOCK 4/4 — WORKERS & SAVE LOGIC (log, linear, rough) — burst/waitfree

def _worker_run_log(state: 'FFState'):
    try:
        s = state.settings

        coil_max_uA = float(min(100.0, abs(s["coil_max_mA"])) * 1000.0)

        # --- Setpoints (ANCHOR vs. CENTER-OUT) ---
        if s.get("log_anchor_mode", False):
            seq = _build_log_anchor_sequence(coil_max_uA, s["log_tier_counts"], s["coil_start_mA"])
            if seq is None:
                raw = _generate_coil_setpoints_log_center_out_uA(
                    coil_max_uA, s["log_tier_counts"], include_zero=True
                )
                coil_setpoints_uA = _order_center_out_around_start(raw, s["coil_start_mA"])
            else:
                coil_setpoints_uA = seq
        else:
            raw = _generate_coil_setpoints_log_center_out_uA(
                coil_max_uA, s["log_tier_counts"], include_zero=True
            )
            coil_setpoints_uA = _order_center_out_around_start(raw, s["coil_start_mA"])

        # --- SMUs konfigurieren (ohne waits/bursts) ---
        try: k.smua.sense  = k.smua.SENSE_REMOTE
        except Exception: pass
        try: k2.smua.sense = k2.smua.SENSE_LOCAL
        except Exception: pass
        _set_measure_speeds(s["nplc"])
        try: k.smua.source.limitv  = 0.01
        except Exception: pass
        try: k2.smua.source.limitv = 10.0
        except Exception: pass
        try: _clear_k2600(k); _clear_k2600(k2)
        except Exception: pass
        _prep_pulse_smua(k, nplc=s["nplc"], limit_v=0.01)

        target_temp = float(s.get("target_temp_K", state.T0))
        state.progress_q.put(('status', f"Zero-field IV (log mode)… T={_get_T():.3f} K"))

        # ----- B=0 characterization -----
        try:
            if abs(s["coil_start_mA"]) < 1e-6:
                _coil_off()
            else:
                _coil_set_current_uA(s["coil_start_mA"]*1000.0, ramp_steps=1, settle_s=s["coil_settle_s"])

            Ic0p_uA, _, ivp = _find_ic_uA_guided(
                v_thresh=s["v_thresh"],
                start_uA=max(s["ic_start_uA"], s["ic0_min_uA"]),
                growth=s["ic_growth"],
                max_uA=s["ic_max_uA"],
                refine_bits=s["refine_bits"],
                pulse_width_s=s["pulse_width_s"],
                pulse_settle_s=s["pulse_settle_s"],
                cooldown_s=s["cooldown_s"],
                sign=+1, collect_iv=True, stop_event=state.stop_event,
                prev_ic_uA=None,
                lower_frac=DEFAULTS["guided_lower_frac"],
                step_rel=DEFAULTS["guided_step_rel"],
                min_step_uA=DEFAULTS["guided_min_step_uA"],
                window_frac=DEFAULTS["guided_window_frac"],
                switch_confirm=s["switch_confirm"],
                min_ic_uA=s["ic0_min_uA"],
            )

            Ic0n_uA, _, ivn = _find_ic_uA_guided(
                v_thresh=s["v_thresh"],
                start_uA=max(s["ic_start_uA"], s["ic0_min_uA"]),
                growth=s["ic_growth"],
                max_uA=s["ic_max_uA"],
                refine_bits=s["refine_bits"],
                pulse_width_s=s["pulse_width_s"],
                pulse_settle_s=s["pulse_settle_s"],
                cooldown_s=s["cooldown_s"],
                sign=-1, collect_iv=True, stop_event=state.stop_event,
                prev_ic_uA=None,
                lower_frac=DEFAULTS["guided_lower_frac"],
                step_rel=DEFAULTS["guided_step_rel"],
                min_step_uA=DEFAULTS["guided_min_step_uA"],
                window_frac=DEFAULTS["guided_window_frac"],
                switch_confirm=s["switch_confirm"],
                min_ic_uA=s["ic0_min_uA"],
            )

            for (I,V) in (ivp + ivn):
                if state.stop_event.is_set(): break
                state.progress_q.put(('iv', I*1e-6, V))

            iv0_extra, iv0_band = [], []
            if s["iv0_refine"]:
                iv0_extra += _iv0_refine_around_ic(+1, abs(Ic0p_uA), s["iv0_pts_per_branch"],
                                                   s["iv0_above_frac"], s["iv0_window_frac"],
                                                   s["ic_start_uA"], s["ic_max_uA"],
                                                   s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                   collect_iv=True, stop_event=state.stop_event)
                iv0_extra += _iv0_refine_around_ic(-1, abs(Ic0n_uA), s["iv0_pts_per_branch"],
                                                   s["iv0_above_frac"], s["iv0_window_frac"],
                                                   s["ic_start_uA"], s["ic_max_uA"],
                                                   s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                   collect_iv=True, stop_event=state.stop_event)
            if s["band_refine_enable"]:
                iv0_band += _refine_band_below_ic(+1, abs(Ic0p_uA),
                                                  s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                  s["ic_start_uA"], s["ic_max_uA"],
                                                  s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                  collect_iv=True, stop_event=state.stop_event)
                iv0_band += _refine_band_below_ic(-1, abs(Ic0n_uA),
                                                  s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                  s["ic_start_uA"], s["ic_max_uA"],
                                                  s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                  collect_iv=True, stop_event=state.stop_event)

            for (I,V) in (iv0_extra + iv0_band):
                if state.stop_event.is_set(): break
                state.progress_q.put(('iv', I*1e-6, V))

            T_b0 = _get_T(); ts_b0 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for (I, V) in (ivp + ivn + iv0_extra + iv0_band):
                branch = '+' if I >= 0 else '-'
                state.iv_search_rows.append([ts_b0, T_b0, s["coil_start_mA"]*1e-3, branch, I*1e-6, V])

        finally:
            _coil_off()

        # Save IV_B0
        try:
            Tnow = _get_T(); nowstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            iv_b0_all = (ivp + ivn + iv0_extra + iv0_band)
            iv_arr = np.array([(nowstr, s["coil_start_mA"]*1e-3, I*1e-6, V, Tnow) for (I,V) in iv_b0_all], dtype=object)
            _save_dual(state.base, "_IV_B0", 'timestamp,I_coil[A],I_junc[A],V[V],T[K]', iv_arr)
        except Exception as ex:
            state.progress_q.put(('status', f"Warning: failed saving IV_B0: {ex}"))

        prev_ic_pos_uA = max(abs(Ic0p_uA), s["ic_start_uA"])
        prev_ic_neg_uA = max(abs(Ic0n_uA), s["ic_start_uA"])

        # ----- Coil sweep (log) -----
        total = len(coil_setpoints_uA)
        for i, coil_uA in enumerate(coil_setpoints_uA, 1):
            if state.stop_event.is_set(): break

            # Temperaturwache (resume/ADR/timeout)
            T = _get_T()
            if T > target_temp + s["dT_stop"]:
                t0 = time.time()
                use_temp_control = _use_temperature_control(target_temp)
                if use_temp_control:
                    _start_temperature_control(target_temp, ramp=0.2)
                    state.progress_q.put(('status', f"PAUSE (log): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — TC hold"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (log): T={T:.3f} K | TC hold {target_temp:.3f} K | "
                               f"cooldown ~{eta_total/60:.1f} min")
                        state.progress_q.put(('status', txt))
                        if T <= target_temp + s["dT_resume"]:
                            break
                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_results(state)
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return
                else:
                    adr_sent = False
                    state.progress_q.put(('status', f"PAUSE (log): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — waiting"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_adr = max(0, s["adr_trigger_s"] - elapsed)
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (log): T={T:.3f} K | ADR in ≥{eta_adr/60:.1f} min | "
                               f"cooldown ~{eta_total/60:.1f} min | ADR events: {state.adr_events}/"
                               f"{('∞' if s['adr_max_resets']==0 else s['adr_max_resets'])}")
                        state.progress_q.put(('status', txt))

                        if (not adr_sent) and (elapsed >= s["adr_trigger_s"]):
                            state.adr_events += 1
                            unlimited = (s["adr_max_resets"] == 0)
                            final_now = (not unlimited) and (state.adr_events >= s["adr_max_resets"])
                            try:
                                state.progress_q.put(('status', f"{'Final ' if final_now else ''}ADR… (event {state.adr_events})"))
                                try:
                                    adr_control.start_adr(setpoint=target_temp, ramp=0.2,
                                                          adr_mode=None, operation_mode='cadr',
                                                          auto_regenerate=True, pre_regenerate=True)
                                except Exception as ex:
                                    state.progress_q.put(('status', f"ADR call failed/absent: {ex}"))
                                adr_sent = True
                            except Exception as ex:
                                state.progress_q.put(('status', f"ADR error: {ex}"))
                            if final_now:
                                state.progress_q.put(('status', "Final ADR sent — stopping and saving results…"))
                                _save_results(state)
                                state.progress_q.put(('done', "Stopped after final ADR. Results saved."))
                                state.stop_event.set()
                                return

                        if adr_sent and abs(T - target_temp) <= s["adr_settle_K"]:
                            state.progress_q.put(('status', "ADR reached setpoint; extra settling…"))
                            time.sleep(max(0.0, s["post_adr_settle_s"]))
                            break

                        if (not adr_sent) and (T <= target_temp + s["dT_resume"]):
                            break

                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_results(state)
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return

                if state.stop_event.is_set():
                    _save_results(state)
                    state.progress_q.put(('done', "Stopped by user, results saved."))
                    return

            _coil_set_current_uA(coil_uA, ramp_steps=s["coil_ramp_steps"], settle_s=s["coil_settle_s"])

            lowfield = (abs(coil_uA) <= s["band_refine_coil_uA_max"])
            ref_bits = s["refine_bits"] if lowfield else int(s.get("refine_bits_highfield", 6))

            Ic_pos_uA, lim_p, ivp_list = _find_ic_uA_guided(
                v_thresh=s["v_thresh"], start_uA=s["ic_start_uA"], growth=s["ic_growth"], max_uA=s["ic_max_uA"],
                refine_bits=ref_bits,
                pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                sign=+1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=prev_ic_pos_uA,
                lower_frac=s["guided_lower_frac"], step_rel=s["guided_step_rel"], min_step_uA=s["guided_min_step_uA"],
                window_frac=s["guided_window_frac"], switch_confirm=s["switch_confirm"], temp_guard=_linear_temp_guard,
            )

            ivp_band = []
            if lowfield and s["band_refine_enable"]:
                ivp_band = _refine_band_below_ic(+1, abs(Ic_pos_uA),
                                                 s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                 s["ic_start_uA"], s["ic_max_uA"],
                                                 s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                 collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)

            T_now = _get_T(); ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for (I, V) in ivp_list:
                state.iv_search_rows.append([ts_now, T_now, coil_uA * 1e-6, '+', I*1e-6, V])
            for (I, V) in ivp_band:
                state.iv_search_rows.append([ts_now, T_now, coil_uA * 1e-6, '+', I*1e-6, V])

            state.rows_pos.append([ts_now, T_now, coil_uA * 1e-6, Ic_pos_uA * 1e-6, int(lim_p)])
            prev_ic_pos_uA = max(abs(Ic_pos_uA), s["ic_start_uA"])

            icn_A = None
            if s["measure_negative"]:
                Ic_neg_uA, lim_n, ivn_list = _find_ic_uA_guided(
                    v_thresh=s["v_thresh"], start_uA=s["ic_start_uA"], growth=s["ic_growth"], max_uA=s["ic_max_uA"],
                    refine_bits=ref_bits,
                    pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                    sign=-1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=prev_ic_neg_uA,
                    lower_frac=s["guided_lower_frac"], step_rel=s["guided_step_rel"], min_step_uA=s["guided_min_step_uA"],
                    window_frac=s["guided_window_frac"], switch_confirm=s["switch_confirm"], temp_guard=_linear_temp_guard,
                )

                ivn_band = []
                if lowfield and s["band_refine_enable"]:
                    ivn_band = _refine_band_below_ic(-1, abs(Ic_neg_uA),
                                                     s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                     s["ic_start_uA"], s["ic_max_uA"],
                                                     s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                     collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)

                T_now_n = _get_T(); ts_now_n = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for (I, V) in ivn_list:
                    state.iv_search_rows.append([ts_now_n, T_now_n, coil_uA * 1e-6, '-', I*1e-6, V])
                for (I, V) in ivn_band:
                    state.iv_search_rows.append([ts_now_n, T_now_n, coil_uA * 1e-6, '-', I*1e-6, V])

                state.rows_neg.append([ts_now_n, T_now_n, coil_uA * 1e-6, Ic_neg_uA * 1e-6, int(lim_n)])
                prev_ic_neg_uA = max(abs(Ic_neg_uA), s["ic_start_uA"])
                icn_A = Ic_neg_uA * 1e-6

            _coil_off()

            # Liveplot & Fortschritt
            state.progress_q.put(('point', coil_uA * 1e-6, Ic_pos_uA * 1e-6, icn_A))
            state.progress_q.put(('progress', i, total))
            if (i % 1) == 0:
                state.progress_q.put(('status', f"[log] I_coil={coil_uA*1e-6:.6f} A  Ic+={Ic_pos_uA*1e-6:.6e} A"))

        _save_results(state)
        state.progress_q.put(('done', "Done (log). Results saved."))

    except Exception as ex:
        try: _save_results(state)
        except Exception: pass
        state.progress_q.put(('error', f"Measurement error (log): {ex}"))


def _build_signed_pair_plan_from(current_uA, max_mA, step_mA, *, include_negative=True):
    """
    Erzeuge eine Liste (µA), die ab current_uA in folgender Paar-Reihenfolge weiterläuft:
    +M0, -M0, +(M0+step), -(M0+step), ...
    Bis |I| <= max_mA. Falls include_negative=False, dann nur +M0, +(M0+step), ...
    """
    max_uA = float(min(100.0, abs(max_mA)) * 1000.0)
    step_uA = float(max(1.0, abs(step_mA) * 1000.0))
    cur = float(current_uA)

    sign0 = +1 if cur >= 0 else -1
    m0 = abs(cur)

    seq = []
    seen = set()

    _append_unique_value(seq, seen, sign0 * m0, max_uA)
    if include_negative:
        _append_unique_value(seq, seen, -sign0 * m0, max_uA)

    kstep = 1
    while True:
        mag = m0 + kstep * step_uA
        if mag > max_uA + 1e-9:
            break
        _append_unique_value(seq, seen, sign0 * mag, max_uA)
        if include_negative:
            _append_unique_value(seq, seen, -sign0 * mag, max_uA)
        kstep += 1

    return seq


def _worker_run_linear(state: 'FFState'):
    try:
        s = state.settings

        raw_setpoints_uA = _generate_coil_setpoints_linear_center_out_uA(
            coil_max_mA=s["coil_max_mA"], step_mA=s["coil_step_mA"], include_zero=True
        )
        coil_setpoints_uA = _order_center_out_around_start(raw_setpoints_uA, s["coil_start_mA"])

        # --- SMUs (ohne waits/bursts) ---
        try: k.smua.sense  = k.smua.SENSE_REMOTE
        except Exception: pass
        try: k2.smua.sense = k2.smua.SENSE_LOCAL
        except Exception: pass
        _set_measure_speeds(s["nplc"])
        try: k.smua.source.limitv  = 0.01
        except Exception: pass
        try: k2.smua.source.limitv = 10.0
        except Exception: pass
        try: _clear_k2600(k); _clear_k2600(k2)
        except Exception: pass
        _prep_pulse_smua(k, nplc=s["nplc"], limit_v=0.01)

        target_temp = float(s.get("target_temp_K", state.T0))
        state.progress_q.put(('status', f"Zero-field IV (linear mode)… T={_get_T():.3f} K"))

        def _linear_temp_guard():
            if state.stop_event.is_set():
                return False
            T = _get_T()
            if T <= target_temp + s["dT_stop"]:
                return True
            try:
                k.smua.source.leveli = 0.0
                k.smua.source.output = 0
            except Exception:
                pass
            state.progress_q.put(('status', f"PAUSE (linear, pulse): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — SMU off"))
            while not state.stop_event.is_set():
                time.sleep(0.5)
                T = _get_T()
                try:
                    k.smua.source.output = 0
                except Exception:
                    pass
                if T <= target_temp + s["dT_resume"]:
                    return True
                state.progress_q.put(('status', f"PAUSE (linear, pulse): T={T:.3f} K > {target_temp + s['dT_resume']:.3f} K — waiting"))
            return False

        # ----- B=0 characterization -----
        try:
            if abs(s["coil_start_mA"]) < 1e-6:
                _coil_off()
            else:
                _coil_set_current_uA(s["coil_start_mA"]*1000.0, ramp_steps=1, settle_s=s["coil_settle_s"])

            state.progress_q.put(('iv_start',))
            Ic0p_uA, _, ivp = _find_ic_uA_guided(
                v_thresh=s["v_thresh"], start_uA=max(s["ic_start_uA"], s["ic0_min_uA"]),
                growth=s["ic_growth"], max_uA=s["ic_max_uA"], refine_bits=s["refine_bits"],
                pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                sign=+1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=None,
                lower_frac=DEFAULTS["guided_lower_frac"], step_rel=DEFAULTS["guided_step_rel"],
                min_step_uA=DEFAULTS["guided_min_step_uA"], window_frac=DEFAULTS["guided_window_frac"],
                switch_confirm=s["switch_confirm"], min_ic_uA=s["ic0_min_uA"], temp_guard=_linear_temp_guard,
            )

            Ic0n_uA, _, ivn = _find_ic_uA_guided(
                v_thresh=s["v_thresh"], start_uA=max(s["ic_start_uA"], s["ic0_min_uA"]),
                growth=s["ic_growth"], max_uA=s["ic_max_uA"], refine_bits=s["refine_bits"],
                pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                sign=-1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=None,
                lower_frac=DEFAULTS["guided_lower_frac"], step_rel=DEFAULTS["guided_step_rel"],
                min_step_uA=DEFAULTS["guided_min_step_uA"], window_frac=DEFAULTS["guided_window_frac"],
                switch_confirm=s["switch_confirm"], min_ic_uA=s["ic0_min_uA"], temp_guard=_linear_temp_guard,
            )

            for (I,V) in (ivp + ivn):
                if state.stop_event.is_set(): break
                state.progress_q.put(('iv', I*1e-6, V))

            iv0_extra, iv0_band = [], []
            if s["iv0_refine"]:
                iv0_extra += _iv0_refine_around_ic(+1, abs(Ic0p_uA), s["iv0_pts_per_branch"],
                                                   s["iv0_above_frac"], s["iv0_window_frac"],
                                                   s["ic_start_uA"], s["ic_max_uA"],
                                                   s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                   collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)
                iv0_extra += _iv0_refine_around_ic(-1, abs(Ic0n_uA), s["iv0_pts_per_branch"],
                                                   s["iv0_above_frac"], s["iv0_window_frac"],
                                                   s["ic_start_uA"], s["ic_max_uA"],
                                                   s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                   collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)
            if s["band_refine_enable"]:
                iv0_band += _refine_band_below_ic(+1, abs(Ic0p_uA),
                                                  s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                  s["ic_start_uA"], s["ic_max_uA"],
                                                  s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                  collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)
                iv0_band += _refine_band_below_ic(-1, abs(Ic0n_uA),
                                                  s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                  s["ic_start_uA"], s["ic_max_uA"],
                                                  s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                  collect_iv=True, stop_event=state.stop_event, temp_guard=_linear_temp_guard)

            state.progress_q.put(('iv_start',))
            for (I, V) in (ivp + ivn + iv0_extra + iv0_band):
                if state.stop_event.is_set(): break
                state.progress_q.put(('iv', I*1e-6, V))

            T_b0 = _get_T(); ts_b0 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for (I, V) in (ivp + ivn + iv0_extra + iv0_band):
                branch = '+' if I >= 0 else '-'
                state.iv_search_rows.append([ts_b0, T_b0, s["coil_start_mA"]*1e-3, branch, I*1e-6, V])

        finally:
            _coil_off()

        # Save IV_B0
        try:
            Tnow = _get_T(); nowstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            iv_b0_all = (ivp + ivn + iv0_extra + iv0_band)
            iv_arr = np.array([(nowstr, s["coil_start_mA"]*1e-3, I*1e-6, V, Tnow) for (I,V) in iv_b0_all], dtype=object)
            _save_dual(state.base, "_IV_B0", 'timestamp,I_coil[A],I_junc[A],V[V],T[K]', iv_arr)
        except Exception as ex:
            state.progress_q.put(('status', f"Warning: failed saving IV_B0: {ex}"))

        use_zero_ic_seed = abs(s["coil_start_mA"]) < 1e-6
        prev_ic_pos_uA = max(abs(Ic0p_uA), s["ic_start_uA"]) if use_zero_ic_seed else None
        prev_ic_neg_uA = max(abs(Ic0n_uA), s["ic_start_uA"]) if use_zero_ic_seed else None

        # ----- Linear coil sweep (dynamisch) -----
        plan_uA = _build_signed_pair_plan_from(
            current_uA=s["coil_start_mA"]*1000.0,
            max_mA=s["coil_max_mA"],
            step_mA=s["coil_step_mA"],
            include_negative=True
        )
        idx = 0
        total = len(plan_uA)
        state.current_coil_uA = None

        i = 0
        while idx < len(plan_uA):
            if state.stop_event.is_set():
                break

            coil_uA = float(plan_uA[idx]); idx += 1; i += 1

            # Temperatur-Guard
            T = _get_T()
            if T > target_temp + s["dT_stop"]:
                t0 = time.time()
                use_temp_control = _use_temperature_control(target_temp)
                if use_temp_control:
                    _start_temperature_control(target_temp, ramp=0.2)
                    state.progress_q.put(('status', f"PAUSE (linear): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — TC hold"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (linear): T={T:.3f} K | TC hold {target_temp:.3f} K | "
                               f"cooldown ~{eta_total/60:.1f} min")
                        state.progress_q.put(('status', txt))
                        if T <= target_temp + s["dT_resume"]:
                            break
                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_results(state)
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return
                else:
                    adr_sent = False
                    state.progress_q.put(('status', f"PAUSE (linear): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — waiting"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_adr = max(0, s["adr_trigger_s"] - elapsed)
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (linear): T={T:.3f} K | ADR in ≥{eta_adr/60:.1f} min | "
                               f"cooldown ~{eta_total/60:.1f} min | ADR events: {state.adr_events}/"
                               f"{('∞' if s['adr_max_resets']==0 else s['adr_max_resets'])}")
                        state.progress_q.put(('status', txt))

                        if (not adr_sent) and (elapsed >= s["adr_trigger_s"]):
                            state.adr_events += 1
                            unlimited = (s["adr_max_resets"] == 0)
                            final_now = (not unlimited) and (state.adr_events >= s["adr_max_resets"])
                            try:
                                state.progress_q.put(('status', f"{'Final ' if final_now else ''}ADR… (event {state.adr_events})"))
                                try:
                                    adr_control.start_adr(setpoint=target_temp, ramp=0.2,
                                                          adr_mode=None, operation_mode='cadr',
                                                          auto_regenerate=True, pre_regenerate=True)
                                except Exception as ex:
                                    state.progress_q.put(('status', f"ADR call failed/absent: {ex}"))
                                adr_sent = True
                            except Exception as ex:
                                state.progress_q.put(('status', f"ADR error: {ex}"))
                            if final_now:
                                state.progress_q.put(('status', "Final ADR sent — stopping and saving results…"))
                                _save_results(state)
                                state.progress_q.put(('done', "Stopped after final ADR. Results saved."))
                                state.stop_event.set()
                                return

                        if adr_sent and abs(T - target_temp) <= s["adr_settle_K"]:
                            state.progress_q.put(('status', "ADR reached setpoint; extra settling…"))
                            time.sleep(max(0.0, s["post_adr_settle_s"]))
                            break

                        if (not adr_sent) and (T <= target_temp + s["dT_resume"]):
                            break

                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_results(state)
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return

                if state.stop_event.is_set():
                    _save_results(state)
                    state.progress_q.put(('done', "Stopped by user, results saved."))
                    return

            # Coil setzen, Ic+ messen
            _coil_set_current_uA(coil_uA, ramp_steps=s["coil_ramp_steps"], settle_s=s["coil_settle_s"])
            state.current_coil_uA = coil_uA

            lowfield = (abs(coil_uA) <= s["band_refine_coil_uA_max"])
            ref_bits = s["refine_bits"] if lowfield else int(s.get("refine_bits_highfield", 6))

            Ic_pos_uA, lim_p, ivp_list = _find_ic_uA_guided(
                v_thresh=s["v_thresh"], start_uA=s["ic_start_uA"], growth=s["ic_growth"], max_uA=s["ic_max_uA"],
                refine_bits=ref_bits,
                pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                sign=+1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=prev_ic_pos_uA,
                lower_frac=s["guided_lower_frac"], step_rel=s["guided_step_rel"], min_step_uA=s["guided_min_step_uA"],
                window_frac=s["guided_window_frac"], switch_confirm=s["switch_confirm"],
            )

            ivp_band = []
            if lowfield and s["band_refine_enable"]:
                ivp_band = _refine_band_below_ic(+1, abs(Ic_pos_uA),
                                                 s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                 s["ic_start_uA"], s["ic_max_uA"],
                                                 s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                 collect_iv=True, stop_event=state.stop_event)

            T_now = _get_T(); ts_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for (I, V) in ivp_list:
                state.iv_search_rows.append([ts_now, T_now, coil_uA * 1e-6, '+', I*1e-6, V])
            for (I, V) in ivp_band:
                state.iv_search_rows.append([ts_now, T_now, coil_uA * 1e-6, '+', I*1e-6, V])

            state.rows_pos.append([ts_now, T_now, coil_uA * 1e-6, Ic_pos_uA * 1e-6, int(lim_p)])
            prev_ic_pos_uA = max(abs(Ic_pos_uA), s["ic_start_uA"])

            icn_A = None
            if s["measure_negative"]:
                Ic_neg_uA, lim_n, ivn_list = _find_ic_uA_guided(
                    v_thresh=s["v_thresh"], start_uA=s["ic_start_uA"], growth=s["ic_growth"], max_uA=s["ic_max_uA"],
                    refine_bits=ref_bits,
                    pulse_width_s=s["pulse_width_s"], pulse_settle_s=s["pulse_settle_s"], cooldown_s=s["cooldown_s"],
                    sign=-1, collect_iv=True, stop_event=state.stop_event, prev_ic_uA=prev_ic_neg_uA,
                    lower_frac=s["guided_lower_frac"], step_rel=s["guided_step_rel"], min_step_uA=s["guided_min_step_uA"],
                    window_frac=s["guided_window_frac"], switch_confirm=s["switch_confirm"],
                )

                ivn_band = []
                if lowfield and s["band_refine_enable"]:
                    ivn_band = _refine_band_below_ic(-1, abs(Ic_neg_uA),
                                                     s["band_refine_frac_low"], s["band_refine_pts"], s["band_refine_top_margin"],
                                                     s["ic_start_uA"], s["ic_max_uA"],
                                                     s["pulse_width_s"], s["pulse_settle_s"], s["cooldown_s"],
                                                     collect_iv=True, stop_event=state.stop_event)

                T_now_n = _get_T(); ts_now_n = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                for (I, V) in ivn_list:
                    state.iv_search_rows.append([ts_now_n, T_now_n, coil_uA * 1e-6, '-', I*1e-6, V])
                for (I, V) in ivn_band:
                    state.iv_search_rows.append([ts_now_n, T_now_n, coil_uA * 1e-6, '-', I*1e-6, V])

                state.rows_neg.append([ts_now_n, T_now_n, coil_uA * 1e-6, Ic_neg_uA * 1e-6, int(lim_n)])
                prev_ic_neg_uA = max(abs(Ic_neg_uA), s["ic_start_uA"])
                icn_A = Ic_neg_uA * 1e-6

            _coil_off()

            # Liveplot & Fortschritt
            state.progress_q.put(('point', coil_uA * 1e-6, Ic_pos_uA * 1e-6, icn_A))
            state.progress_q.put(('progress', i, total))
            if (i % 10) == 0:
                state.progress_q.put(('status', f"[linear] I_coil={coil_uA*1e-6:.6f} A  Ic+={Ic_pos_uA*1e-6:.6e} A"))

        _save_results(state)
        state.progress_q.put(('done', "Done (linear). Results saved."))

    except Exception as ex:
        try: _save_results(state)
        except Exception: pass
        state.progress_q.put(('error', f"Linear measurement error: {ex}"))


def _smooth_moving_avg(y, w):
    w = int(max(1, w))
    if w == 1 or len(y) <= 2: return np.array(y, float)
    k = w // 2
    padL = np.full(k, y[0], dtype=float)
    padR = np.full(k, y[-1], dtype=float)
    arr = np.concatenate([padL, np.asarray(y, float), padR])
    out = np.convolve(arr, np.ones(w)/w, mode='valid')
    return out


def _find_extrema_centered(x, y, min_prom_rel=0.05):
    """Min/Max anhand Vorzeichenwechsel der 1. Ableitung + einfacher Prominenzfilter."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 5: return [], []
    dy = np.diff(y)
    mins = []; maxs = []
    for i in range(1, len(dy)):
        if dy[i-1] < 0 and dy[i] > 0: mins.append(i)
        if dy[i-1] > 0 and dy[i] < 0: maxs.append(i)
    if len(y) > 0:
        yr = max(1e-12, (np.nanmax(y) - np.nanmin(y)))
        thr = float(min_prom_rel) * yr
        maxs = _extrema_prominent(maxs, y, thr, invert=False)
        mins = _extrema_prominent(mins, y, thr, invert=True)
    return mins, maxs


def _estimate_period_from_extrema(x, idxs):
    if len(idxs) < 2: return float('nan'), 0
    xs = np.array([x[i] for i in idxs], float)
    diffs = np.diff(np.sort(xs))
    if len(diffs) == 0: return float('nan'), len(idxs)
    return float(np.nanmedian(diffs)), len(idxs)


def _worker_run_rough(state: 'FFState', ic_line, ic_ax, ic_canvas, status_lbl, bar):
    """Schneller Rough-Scan: nur |Ic| auf +Branch, Min/Max finden, Periode schätzen, speichern."""
    try:
        s = state.settings

        # Setpoints
        max_mA  = min(100.0, abs(s["rough_max_mA"]))
        step_mA = max(0.001, abs(s["rough_step_mA"]))
        raw = _generate_coil_setpoints_linear_center_out_uA(max_mA, step_mA, include_zero=True)
        coil_setpoints_uA = _order_center_out_around_start(raw, s["coil_start_mA"])
        total = len(coil_setpoints_uA)

        # SMUs (ohne waits/bursts)
        try: k.smua.sense  = k.smua.SENSE_REMOTE
        except Exception: pass
        try: k2.smua.sense = k2.smua.SENSE_LOCAL
        except Exception: pass
        _set_measure_speeds(s["nplc"])
        try: k.smua.source.limitv  = 0.01
        except Exception: pass
        try: k2.smua.source.limitv = 10.0
        except Exception: pass
        try: _clear_k2600(k); _clear_k2600(k2)
        except Exception: pass
        _prep_pulse_smua(k, nplc=s["nplc"], limit_v=0.01)

        target_temp = float(s.get("target_temp_K", state.T0))
        xs_A, icA = [], []

        for i, coil_uA in enumerate(coil_setpoints_uA, 1):
            if state.stop_event.is_set():
                _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
                state.progress_q.put(('done', "Stopped by user, results saved."))
                return

            # Temperatur-Guard
            T = _get_T()
            if T > target_temp + s["dT_stop"]:
                t0 = time.time()
                use_temp_control = _use_temperature_control(target_temp)
                if use_temp_control:
                    _start_temperature_control(target_temp, ramp=0.2)
                    state.progress_q.put(('status', f"PAUSE (rough): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — TC hold"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (rough): T={T:.3f} K | TC hold {target_temp:.3f} K | "
                               f"cooldown ~{eta_total/60:.1f} min")
                        state.progress_q.put(('status', txt))
                        if T <= target_temp + s["dT_resume"]:
                            break
                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return
                else:
                    adr_sent = False
                    state.progress_q.put(('status', f"PAUSE (rough): T={T:.3f} K > {target_temp + s['dT_stop']:.3f} K — waiting"))
                    while not state.stop_event.is_set():
                        time.sleep(0.5)
                        T = _get_T()
                        elapsed = time.time() - t0
                        eta_adr = max(0, s["adr_trigger_s"] - elapsed)
                        eta_total = max(0, s["cooldown_total_s"] - elapsed)
                        txt = (f"PAUSE (rough): T={T:.3f} K | ADR in ≥{eta_adr/60:.1f} min | "
                               f"cooldown ~{eta_total/60:.1f} min | ADR events: {state.adr_events}/"
                               f"{('∞' if s['adr_max_resets']==0 else s['adr_max_resets'])}")
                        state.progress_q.put(('status', txt))

                        if (not adr_sent) and (elapsed >= s["adr_trigger_s"]):
                            state.adr_events += 1
                            unlimited = (s["adr_max_resets"] == 0)
                            final_now = (not unlimited) and (state.adr_events >= s["adr_max_resets"])
                            try:
                                state.progress_q.put(('status', f"{'Final ' if final_now else ''}ADR… (event {state.adr_events})"))
                                try:
                                    adr_control.start_adr(setpoint=target_temp, ramp=0.2,
                                                          adr_mode=None, operation_mode='cadr',
                                                          auto_regenerate=True, pre_regenerate=True)
                                except Exception as ex:
                                    state.progress_q.put(('status', f"ADR call failed/absent: {ex}"))
                                adr_sent = True
                            except Exception as ex:
                                state.progress_q.put(('status', f"ADR error: {ex}"))

                            if final_now:
                                state.progress_q.put(('status', "Final ADR sent — stopping and saving results…"))
                                _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
                                state.progress_q.put(('done', "Stopped after final ADR. Results saved."))
                                state.stop_event.set()
                                return

                        if adr_sent and abs(T - target_temp) <= s["adr_settle_K"]:
                            state.progress_q.put(('status', "ADR reached setpoint; extra settling…"))
                            time.sleep(max(0.0, s["post_adr_settle_s"]))
                            break

                        if (not adr_sent) and (T <= target_temp + s["dT_resume"]):
                            break

                        if elapsed >= s["cooldown_total_s"]:
                            state.progress_q.put(('status', "Cooldown window exceeded — stopping and saving results…"))
                            _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
                            state.progress_q.put(('done', "Stopped after cooldown window. Results saved."))
                            state.stop_event.set()
                            return

                if state.stop_event.is_set():
                    _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
                    state.progress_q.put(('done', "Stopped by user, results saved."))
                    return

            # Coil setzen, Ic grob messen
            _coil_set_current_uA(coil_uA, ramp_steps=s["coil_ramp_steps"], settle_s=s["coil_settle_s"])

            Ic_pos_uA, _, _ = _find_ic_uA_adaptive(
                v_thresh=s["v_thresh"],
                start_uA=s["ic_start_uA"],
                growth=s["ic_growth"],
                max_uA=s["ic_max_uA"],
                refine_bits=s["rough_refine_bits"],
                pulse_width_s=s["pulse_width_s"],
                pulse_settle_s=s["pulse_settle_s"],
                cooldown_s=s["cooldown_s"],
                sign=+1, collect_iv=False, stop_event=state.stop_event, switch_confirm=3
            )

            xs_A.append(coil_uA * 1e-6)
            icA.append(Ic_pos_uA * 1e-6)

            # Liveplot
            ic_line.set_data(xs_A, icA)
            ic_ax.relim(); ic_ax.autoscale_view(); ic_canvas.draw_idle()

            state.progress_q.put(('progress', i, total))
            if (i % 20) == 0:
                state.progress_q.put(('status', f"[rough] I_coil={coil_uA*1e-6:.6f} A  Ic≈{Ic_pos_uA*1e-6:.6e} A"))

            _coil_off()

        # Reguläres Ende → speichern + kurze Diagnose
        _save_rough_arrays(state, xs_A, icA, suffix="")
        try:
            ic_smooth = _smooth_moving_avg(icA, max(1, int(s["rough_smooth_pts"])))
            mins_idx, maxs_idx = _find_extrema_centered(xs_A, ic_smooth, min_prom_rel=s["rough_min_prominence"])
            per_min, nmin = _estimate_period_from_extrema(xs_A, mins_idx)
            per_max, nmax = _estimate_period_from_extrema(xs_A, maxs_idx)
            status_lbl.config(text=f"Rough done. Min-period ~ {per_min:.3e} A ({nmin} minima), Max ~ {per_max:.3e} A ({nmax} maxima)")
        except Exception:
            status_lbl.config(text="Rough done. Saved.")

        state.progress_q.put(('done', "Done (rough). Results saved."))

    except Exception as ex:
        try:
            if 'xs_A' in locals() and 'icA' in locals() and len(xs_A) > 0:
                _save_rough_arrays(state, xs_A, icA, suffix="_PARTIAL")
        except Exception:
            pass
        state.progress_q.put(('error', f"Rough-scan error: {ex}"))


def _save_results(state: 'FFState'):
    """Dual-save aller Resulttabellen (Ic_pos, Ic_neg, IV_all_search) mit Datum im Basenamen."""
    base = state.base or "fraunhofer"
    if len(state.rows_pos) > 0:
        pos_arr = np.array(state.rows_pos, dtype=object)
        _save_dual(base, "_Ic_pos", 'timestamp,T[K],I_coil[A],Ic_pos[A],limit', pos_arr)
    if len(state.rows_neg) > 0:
        neg_arr = np.array(state.rows_neg, dtype=object)
        _save_dual(base, "_Ic_neg", 'timestamp,T[K],I_coil[A],Ic_neg[A],limit', neg_arr)
    if len(state.iv_search_rows) > 0:
        iv_all_arr = np.array(state.iv_search_rows, dtype=object)
        _save_dual(base, "_IV_all_search", 'timestamp,T[K],I_coil[A],branch,I_junc[A],V[V]', iv_all_arr)
#%% Cell 4: Simple Tc Measurement
#Tc_Measurement 
def create_tc_frame(parent):
    frame = tk.Frame(parent)
    tk.Label(frame, text="Start Temp[K]").grid(row=0, column=0)
    tk.Entry(frame, width=10).grid(row=1, column=0)  # Start Temp
    tk.Label(frame, text="End Temp[K]").grid(row=0, column=1)
    tk.Entry(frame, width=10).grid(row=1, column=1)  # End Temp
    tk.Label(frame, text="Ramp Speed[K/min]").grid(row=0, column=2)
    tk.Entry(frame, width=10).grid(row=1, column=2)  # Ramp Speed
    tk.Label(frame, text="Ramp Current[µA]").grid(row=0, column=3)
    tk.Entry(frame, width=10).grid(row=1, column=3)  # Ramp Current
    tk.Button(frame, text="Remove", command=frame.destroy).grid(row=1, column=4)
    return frame

@dataclass
class TcMeasurementUI:
    sections_frame: object
    section_frames: list
    measurement_started: object
    tc_window: object

def _tc_add_section(ui: TcMeasurementUI):
    frame = create_tc_frame(ui.sections_frame)
    frame.pack(fill=tk.X)
    ui.section_frames.append(frame)

def _tc_update_r_plot(temp_data, resistance_data, line, ax, fig, _frame):
    if temp_data:
        line.set_data(temp_data, resistance_data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()

def _tc_get_data(ui: TcMeasurementUI):
    tc_data = []
    maximumProgress = 0
    offset = 0
    for frame in ui.section_frames:
        try:
            start_temp = float(frame.winfo_children()[1].get())
            if 0.199 < start_temp < 290:
                None
            else:
                Errormessage("Out of Value Range 0.2-290K")
                if start_temp > 290:
                    start_temp = 290
                else:
                    start_temp = 0.2
            end_temp = float(frame.winfo_children()[3].get())
            if 0.2 < end_temp < 290:
                None
            else:
                Errormessage("Out of Value Range 0.2-290K")
                if end_temp > 290:
                    end_temp = 290
                else:
                    end_temp = 0.2
            ramp_speed = float(frame.winfo_children()[5].get())
            ramp_current = float(frame.winfo_children()[7].get()) / 1000000
            tc_data.append((start_temp, end_temp, ramp_speed, ramp_current))
            maximumProgress = maximumProgress + abs(end_temp - start_temp)
        except ValueError:
            print("Fehler")
            tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e

    print(tc_data)
    progress_window = tk.Toplevel(root)
    progress_window.title("Tc Measurement Progress")
    progress_bar = ttk.Progressbar(progress_window, maximum=maximumProgress, length=300)
    progress_bar.pack(padx=20, pady=20)
    progress_window.update()
    global rt
    rt = ResultTable(
        column_titles=['Time',  'Current [A]','Voltage[V]','Temperature[K]'],
        units=[' ',  'V',"A",'K'],
        params={'recorded': time.asctime(), 'sweep_type': 'tc'},
    )
    user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name:")
    NameWindow = str(user_input)
    plot_window, fig, ax, line = create_plot_window_for_Resistance(NameWindow)

    Temp_data = []
    Resistance_data = []

    ani = FuncAnimation(fig, partial(_tc_update_r_plot, Temp_data, Resistance_data, line, ax, fig),
                        frames=10000, repeat=False, interval=5000)
    ani._start()
    for start_temp, end_temp, ramp_speed, ramp_current in tc_data:
        if start_temp + 5 < temperature_control.kelvin and start_temp > 5:
            pass
        else:
            temperature_control.start((start_temp, ramp_speed))
        if ui.measurement_started.get() == 1:
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)
            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                voltage = k.smua.measure.v()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                temperature = temperature_control.kelvin
                Temp_data.append(temperature)
                rt.append_row([timestamp, ramp_current, voltage, temperature])
                Resistance_data.append(voltage / ramp_current)
                time.sleep(0.2)
        else:
            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                time.sleep(0.2)
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)
        while abs(client.query('T_sample.kelvin') - end_temp) > 0.01:
            voltage = k.smua.measure.v()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            temperature = temperature_control.kelvin
            Temp_data.append(temperature)
            rt.append_row([timestamp, ramp_current, voltage, temperature])
            Resistance_data.append(voltage / ramp_current)
            progress_bar['value'] = abs(temperature - start_temp) + offset
            progress_window.update()
            time.sleep(0.2)
        offset = offset + abs(end_temp - start_temp)
    data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
    safe_file(data_numeric, user_input)
    ui.tc_window.destroy()
    progress_window.destroy()
    k.smua.source.output = 0
    k2.smua.source.output = 0
    global measurement
    measurement = 0
    try:
        ani._stop()
    except AttributeError:
        pass
# New function to create the Tc measurement window
def create_tc_measurement():
    global measurement 
    measurement = 1
    k.smua.sense = k.smua.SENSE_LOCAL
    section_frames = []
    tc_window = tk.Toplevel(root)
    tc_window.title("Tc Measurement")
    tc_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(tc_window))

    sections_frame = tk.Frame(tc_window)
    sections_frame.grid(row=0, column=0, padx=10, pady=10)

    ui = TcMeasurementUI(
        sections_frame=sections_frame,
        section_frames=section_frames,
        measurement_started=measurement_started,
        tc_window=tc_window,
    )

    _tc_add_section(ui)  # Add one section to begin with

    controls_frame = tk.Frame(tc_window)
    controls_frame.grid(row=1, column=0, padx=10, pady=10)

    # Place the Checkbutton and buttons in the controls_frame using grid
    tk.Checkbutton(controls_frame, text="Start Measurement", variable=measurement_started).grid(row=0, column=0, columnspan=2)
    tk.Button(controls_frame, text="Add Section", command=partial(_tc_add_section, ui)).grid(row=0, column=2)
    tk.Button(controls_frame, text="Measurement start", command=partial(_tc_get_data, ui)).grid(row=0, column=3)
    
    # Additional widgets or layout adjustments can be made here if needed
    
    # Optionally, set column and row weights to control resizing behavior
    tc_window.columnconfigure(0, weight=1)
    tc_window.rowconfigure(0, weight=1)

    

#%% Cell 5: Simple IV Curve

#IV sweeps

#%% Cell 5: Simple IV Curve (with ETA)
#%% Cell 5: Simple IV Curve (with ETA and correct V↔I mapping)

def _update_iv_plot(voltage_data, current_data, line, ax, fig, _frame):
    if voltage_data:
        line.set_data(voltage_data, current_data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()

def run_sweep(sweep_points, user_input, user_input2=None):
    global measurement
    measurement = 1

    # 1) 4-point sense mode
    if local_remote_var.get() == 1:
        k.smua.sense  = k.smua.SENSE_REMOTE
        k2.smua.sense = k2.smua.SENSE_REMOTE
    else:
        k.smua.sense  = k.smua.SENSE_LOCAL
        k2.smua.sense = k2.smua.SENSE_LOCAL

    # 2) Count total steps
    total_steps = sum(steps for start, end, steps in sweep_points)

    # 3) Build the progress UI
    progress_window = tk.Toplevel(root)
    progress_window.title("Sweep Progress")
    progress_bar = ttk.Progressbar(progress_window, maximum=total_steps, length=300)
    progress_bar.pack(padx=20, pady=(20,5))
    time_label = tk.Label(progress_window, text="Estimating time remaining…")
    time_label.pack(padx=20, pady=(0,20))
    progress_window.update()

    # 4) Start timer
    start_time = time.time()

    # 5) Prepare ResultTable and plotting for SMU A
    global rt
    rt = ResultTable(
        column_titles=['Time', 'Voltage[V]', 'Current [A]', 'Temperature[K]'],
        units=[' ',            'V',           'A',               'K'],
        params={'recorded': time.asctime(), 'sweep_type': 'iv'},
    )
    current_step = 0

    temp_str = client.query('T_sample.kelvin') or 300
    NameWindow = f"{user_input}{temp_str}"
    plot_window, fig, ax, line = create_plot_window_for_Current_Voltage(NameWindow)

    # 6) Optionally set up SMU 2
    use_smu2 = use_smu2_var.get()
    if use_smu2:
        k2.smua.sense = k2.smua.SENSE_REMOTE
        rt2 = ResultTable(
            column_titles=['Time','Current [A]','Voltage [V]','Temperature [K]'],
            units=[' ',         'A',           'V',            'K'],
            params={'recorded': time.asctime(), 'sweep_type': 'iv_current'},
        )
        NameWindow2 = f"{user_input2}{temp_str}"
        plot_window2, fig2, ax2, line2 = create_plot_window_for_Current_Voltage(NameWindow2)

    # 7) Data buffers
    voltage_data, current_data = [], []
    if use_smu2:
        voltage_data2, current_data2 = [], []

    # 8) Live-plot updaters (voltage on x, current on y)
    ani = FuncAnimation(fig, partial(_update_iv_plot, voltage_data, current_data, line, ax, fig),
                        frames=10000, repeat=False, interval=5000)
    ani._start()
    if use_smu2:
        ani2 = FuncAnimation(fig2, partial(_update_iv_plot, voltage_data2, current_data2, line2, ax2, fig2),
                             frames=10000, repeat=False, interval=5000)
        ani2._start()

    # 9) Main sweep
    for start, end, steps in sweep_points:
        voltages = [round(start + (end-start)*j/steps, 6) for j in range(steps)]

        # Ensure SMUs are in voltage mode
        k.smua.   source.output = 1
        k.smua.   source.func   = 1    # **Voltage mode**
        k.smua.   measure.nplc  = integrationtime

        if use_smu2:
            k2.smua.source.output = 1
            k2.smua.source.func   = 1    # **Voltage mode on SMU2**
            k2.smua.measure.nplc  = integrationtime

        for v in voltages:
            # Apply voltage
            k.smua.source.levelv = v
            if use_smu2:
                k2.smua.source.levelv = v

            time.sleep(0.1)  # let DUT settle
            T = client.query('T_sample.kelvin') or 300
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Measure current
            current = k.smua.measure.i()
            if use_smu2:
                current2 = k2.smua.measure.i()

            # Wait integration
            time.sleep(k.smua.measure.nplc)

            # --- Log with correct ordering: [Time, Voltage, Current, Temp] ---
            rt.append_row([timestamp, v, current, T])
            if use_smu2:
                rt2.append_row([timestamp, current2, v, T])

            # --- Progress bar + ETA ---
            current_step += 1
            progress_bar['value'] = current_step
            progress_window.update()
            elapsed   = time.time() - start_time
            remaining = elapsed * (total_steps - current_step) / current_step
            hrs, rem  = divmod(int(remaining), 3600)
            mins, secs= divmod(rem, 60)
            time_label.config(text=f"ETA: {hrs}h {mins}m {secs}s")
            progress_window.update_idletasks()

            # --- Update plot buffers ---
            voltage_data.append(v)
            current_data.append(current)
            if use_smu2:
                voltage_data2.append(v)
                current_data2.append(current2)

    # 10) Save & cleanup
    data_numeric = np.array([[ts] + list(map(float, vals)) for ts, *vals in rt.data])
    safe_file(data_numeric, user_input)
    if use_smu2:
        data_numeric2 = np.array([[ts] + list(map(float, vals)) for ts, *vals in rt2.data])
        safe_file(data_numeric2, user_input2)

    progress_window.destroy()
    try: ani._stop()
    except: pass




def add_predefined_section(parent, start, end, steps):
    frame = create_section_frame(parent)
    frame.winfo_children()[1].insert(0, str(start))  # Set start value
    frame.winfo_children()[3].insert(0, str(end))  # Set end value
    frame.winfo_children()[5].insert(0, str(steps))  # Set steps value
    frame.pack(fill=tk.X)
    return frame

def load_sweep():
    global measurement 
    measurement = 1
    print("Before showing dialog")
    filename = filedialog.askopenfilename(initialdir=DEFAULT_PATH, title="Load Sweep",
                                      filetypes=[("Sweep-Dateien", "*.swp"), ("Alle Dateien", "*.*")])
    print(f"After showing dialog, filename: {filename}")

    measurement = 0
    if filename:
        with open(filename, 'r') as file:
            print("file opened")  # Debugging-Information
            sweep_points = json.load(file)
            print("sweep_points loaded")  # Debugging-Information

            # Create new sweep window
            sweep_window, section_frames, sections_frame = create_sweep()  # Catch the returned values

            # Delete the initial empty section
            section_frames[0].destroy()
            section_frames.clear()

            # Add sections for each loaded sweep point
            for start, end, steps in sweep_points:
                frame = add_predefined_section(sections_frame, start, end, steps)
                section_frames.append(frame)
                

    
def save_sweep(sweep_points):
    filename = filedialog.asksaveasfilename(initialdir=DEFAULT_PATH, title="Sweep speichern",
                                            filetypes=[("Sweep-Dateien", "*.swp"), ("Alle Dateien", "*.*")])
    if filename:
        with open(filename, 'w') as file:
            json.dump(sweep_points, file)
    
def create_section_frame(parent):
    frame = tk.Frame(parent)
    tk.Label(frame, text="Start").grid(row=0, column=0)
    tk.Entry(frame, width=5).grid(row=1, column=0)  # Startwert
    tk.Label(frame, text="End").grid(row=0, column=1)
    tk.Entry(frame, width=5).grid(row=1, column=1)  # Endwert
    tk.Label(frame, text="Steps").grid(row=0, column=2)
    tk.Entry(frame, width=5).grid(row=1, column=2)  # Schritte
    tk.Button(frame, text="Remove", command=frame.destroy).grid(row=1, column=3)

    return frame

@dataclass
class SweepUI:
    section_frames: list
    sections_frame: object
    save_var: object
    sweep_window: object

def _sweep_add_section(ui: SweepUI):
    frame = create_section_frame(ui.sections_frame)
    frame.pack(fill=tk.X)
    ui.section_frames.append(frame)

def _sweep_run(ui: SweepUI):
    sweep_points = []
    total_steps = 0
    for frame in ui.section_frames:
        try:
            start_value = frame.winfo_children()[1].get()
            end_value = frame.winfo_children()[3].get()
            steps_value = frame.winfo_children()[5].get()
            print(f"Start: {start_value}, End: {end_value}, Steps: {steps_value}")
            start = float(start_value)
            end = float(end_value)
            steps = int(steps_value)
            sweep_points.append((start, end, steps))
            total_steps += steps
        except ValueError:
            print("Fehler")
            tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e
    if ui.save_var.get():
        save_sweep(sweep_points)
    user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name:")
    use_smu2var = use_smu2_var.get()
    if use_smu2var:
        user_input2 = simpledialog.askstring("Insert Filename", "Please insert the file Name for SMU2:")
        run_sweep(sweep_points, user_input, user_input2)
    else:
        run_sweep(sweep_points, user_input)
    ui.sweep_window.destroy()
    k.smua.source.output = 0
    k2.smua.source.output = 0
    global measurement
    measurement = 0

def create_sweep():
    global measurement 
    measurement = 1
    if local_remote_var.get() == 1:
        k.smua.sense = k.smua.SENSE_REMOTE
    else:
        k.smua.sense = k.smua.SENSE_LOCAL
    section_frames = []
    sweep_window = tk.Toplevel(root)
    sweep_window.title("Make new Voltage Sweep")
    sweep_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(sweep_window))
    sections_frame = tk.Frame(sweep_window)
    sections_frame.pack()

    save_var = tk.IntVar()  # Create an IntVar to hold the state of the checkbox
    ui = SweepUI(
        section_frames=section_frames,
        sections_frame=sections_frame,
        save_var=save_var,
        sweep_window=sweep_window,
    )

    _sweep_add_section(ui)  # Fügen Sie einen Abschnitt hinzu, um zu beginnen

    controls_frame = tk.Frame(sweep_window)
    controls_frame.pack(fill=tk.X)

    save_checkbutton = tk.Checkbutton(controls_frame, text="Save Measurement Settings", variable=save_var)
    save_checkbutton.pack(side=tk.LEFT)  # Adjust the side argument as needed
    # Add the checkbox for Local/Remote measurement here
    local_remote_checkbox = tk.Checkbutton(controls_frame, text="4 Point", variable=local_remote_var)
    local_remote_checkbox.pack(side=tk.LEFT)
    smu2_checkbox = tk.Checkbutton(controls_frame, text="Use SMU2", variable=use_smu2_var)
    smu2_checkbox.pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Add Section", command=partial(_sweep_add_section, ui)).pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Sweep start", command=partial(_sweep_run, ui)).pack(side=tk.RIGHT)
    return sweep_window, section_frames, sections_frame

def start_sweep_thread(action):
    print(f"Starting thread for {action}")  # Debugging-Information
    sweep_thread = threading.Thread(target=action)
    sweep_thread.start()




    
#%% Cell 6: Temperature dependent IV Curve
#IV Sweep at different temperatures
def create_temp_frame(parent):
    frame = tk.Frame(parent)
    tk.Label(frame, text="Temperature[K]").grid(row=0, column=0)
    tk.Entry(frame, width=10).grid(row=1, column=0)  # Temperature
    tk.Label(frame, text="Ramp Speed[K/min]").grid(row=0, column=1)
    # Set the initial value of the Ramp Speed Entry to 1
    ramp_speed_entry = tk.Entry(frame, width=10)
    ramp_speed_entry.grid(row=1, column=1)
    ramp_speed_entry.insert(0, "1")  # Set the default value to 1
    tk.Button(frame, text="Remove", command=frame.destroy).grid(row=1, column=2)
    return frame

@dataclass
class IVTempGridState:
    row: int = 0
    col: int = 0

@dataclass
class IVTempUI:
    iv_section_frames: list
    temp_section_frames: list
    iv_sections_frame: object
    temp_sections_frame: object
    grid_state: IVTempGridState
    iv_temp_window: object

def _iv_temp_add_iv_section(ui: IVTempUI):
    frame = create_section_frame(ui.iv_sections_frame)
    frame.grid(row=len(ui.iv_section_frames), column=0, padx=10, pady=10)
    ui.iv_section_frames.append(frame)

def _iv_temp_add_temp_section(ui: IVTempUI):
    frame = create_temp_frame(ui.temp_sections_frame)
    frame.grid(row=ui.grid_state.row, column=ui.grid_state.col, padx=10, pady=10)
    ui.temp_section_frames.append(frame)
    ui.grid_state.row += 1
    if ui.grid_state.row >= 10:
        ui.grid_state.row = 0
        ui.grid_state.col += 1

def _iv_temp_run(ui: IVTempUI):
    sweep_data = []
    for frame in ui.iv_section_frames:
        try:
            start_value = frame.winfo_children()[1].get()
            end_value = frame.winfo_children()[3].get()
            steps_value = frame.winfo_children()[5].get()
            start = float(start_value)
            end = float(end_value)
            steps = int(steps_value)
            sweep_data.append((start, end, steps))
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid input")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e

    temperature_data = []
    for frame in ui.temp_section_frames:
        try:
            start_temp_value = frame.winfo_children()[1].get()
            start_temp_value = float(start_temp_value)
            if 0.2 < start_temp_value < 290:
                None
            else:
                Errormessage("Out of Value Range 0.2-290K")
                if start_temp_value > 290:
                    start_temp_value = 290
                else:
                    start_temp_value = 0.2
            ramp_speed_value = frame.winfo_children()[3].get()
            start_temp = float(start_temp_value)
            ramp_speed = float(ramp_speed_value)
            temperature_data.append((start_temp, ramp_speed))
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid input")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e
    user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name:")

    for start_temp, ramp_speed in temperature_data:
        temperature_control.start((start_temp, ramp_speed))
        while abs(client.query('T_sample.kelvin') - start_temp) > 0.01:
            time.sleep(0.2)
        time.sleep(60)
        temperature_str = str(client.query('T_sample.kelvin') or 300)
        user_input_temperature = user_input + "_" + temperature_str + "K" + "_"
        run_sweep(sweep_data, user_input_temperature)
    temperature_control.stop()

    ui.iv_temp_window.destroy()
    k.smua.source.output = 0
    k2.smua.source.output = 0
    global measurement
    measurement = 0

def IV_Temp():
    global measurement 
    measurement = 1
    iv_temp_window = tk.Toplevel(root)
    iv_temp_window.title("IV Sweep at Different Temperatures")
    iv_temp_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(iv_temp_window))
    iv_sections_frame = tk.Frame(iv_temp_window)
    iv_sections_frame.pack()
    temp_sections_frame = tk.Frame(iv_temp_window)
    temp_sections_frame.pack()

    ui = IVTempUI(
        iv_section_frames=iv_section_frames,
        temp_section_frames=temp_section_frames,
        iv_sections_frame=iv_sections_frame,
        temp_sections_frame=temp_sections_frame,
        grid_state=IVTempGridState(),
        iv_temp_window=iv_temp_window,
    )

    _iv_temp_add_iv_section(ui)  # Add one section to begin with for IV sweep data
    _iv_temp_add_temp_section(ui)  # Add one section to begin with for temperature data

    controls_frame = tk.Frame(iv_temp_window)
    controls_frame.pack(fill=tk.X)  # Use pack to place controls_frame correctly
    local_remote_checkbox = tk.Checkbutton(controls_frame, text="4 Point", variable=local_remote_var)
    local_remote_checkbox.pack(side=tk.LEFT)
    smu2_checkbox = tk.Checkbutton(controls_frame, text="Use SMU2", variable=use_smu2_var)
    smu2_checkbox.pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Add Section (IV)", command=partial(_iv_temp_add_iv_section, ui)).pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Add Section (Temp)", command=partial(_iv_temp_add_temp_section, ui)).pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Measurement start", command=partial(_iv_temp_run, ui)).pack(side=tk.RIGHT)

    return iv_temp_window  # Return the window to keep it open


iv_section_frames = []  # List to hold the frames for IV sweep data
temp_section_frames = []  # List to hold the frames for temperature data

#%% Cell 6b: Temperature dependent Current Sweep
def _parse_temperature_list(text):
    temps = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        temps.append(float(value))
    return temps

def _generate_temperature_steps(start_temp, end_temp, intermediate_steps):
    if intermediate_steps < 0:
        raise ValueError("Intermediate steps must be >= 0")
    if intermediate_steps == 0:
        if start_temp == end_temp:
            return [start_temp]
        return [start_temp, end_temp]
    return np.linspace(start_temp, end_temp, intermediate_steps + 2).tolist()

def create_current_temp_sweep():
    global measurement
    measurement = 1

    current_temp_window = tk.Toplevel(root)
    current_temp_window.title("Current Sweep at Different Temperatures")
    current_temp_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(current_temp_window))

    sweep_sections_frame = tk.Frame(current_temp_window)
    sweep_sections_frame.pack()

    def add_current_section():
        frame = create_section_frame(sweep_sections_frame)
        frame.pack(fill=tk.X)
        sweep_section_frames.append(frame)

    sweep_section_frames = []
    add_current_section()

    temp_config_frame = tk.Frame(current_temp_window)
    temp_config_frame.pack(pady=10)

    tk.Label(temp_config_frame, text="Temperatures [K] (comma-separated)").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    temp_list_entry = tk.Entry(temp_config_frame, width=35)
    temp_list_entry.grid(row=0, column=1, columnspan=3, padx=5, pady=2, sticky="w")

    tk.Label(temp_config_frame, text="Start [K]").grid(row=1, column=0, padx=5, pady=2, sticky="w")
    start_temp_entry = tk.Entry(temp_config_frame, width=10)
    start_temp_entry.grid(row=1, column=1, padx=5, pady=2, sticky="w")

    tk.Label(temp_config_frame, text="End [K]").grid(row=1, column=2, padx=5, pady=2, sticky="w")
    end_temp_entry = tk.Entry(temp_config_frame, width=10)
    end_temp_entry.grid(row=1, column=3, padx=5, pady=2, sticky="w")

    tk.Label(temp_config_frame, text="Intermediate Steps").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    steps_entry = tk.Entry(temp_config_frame, width=10)
    steps_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
    steps_entry.insert(0, "0")

    tk.Label(temp_config_frame, text="Ramp Speed [K/min]").grid(row=2, column=2, padx=5, pady=2, sticky="w")
    ramp_speed_entry = tk.Entry(temp_config_frame, width=10)
    ramp_speed_entry.grid(row=2, column=3, padx=5, pady=2, sticky="w")
    ramp_speed_entry.insert(0, "1")

    tk.Label(temp_config_frame, text="Thermalization Time [s]").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    settle_entry = tk.Entry(temp_config_frame, width=10)
    settle_entry.grid(row=3, column=1, padx=5, pady=2, sticky="w")
    settle_entry.insert(0, "60")

    def run_current_temp_sweep():
        sweep_points = []
        for frame in sweep_section_frames:
            try:
                start_value = frame.winfo_children()[1].get()
                end_value = frame.winfo_children()[3].get()
                steps_value = frame.winfo_children()[5].get()
                start = float(start_value) * (10 ** -6)
                end = float(end_value) * (10 ** -6)
                steps = int(steps_value)
                sweep_points.append((start, end, steps))
            except ValueError:
                tk.messagebox.showerror("Fehler", "Ungültige Eingabe für den Strom-Sweep.")
                return
            except tk.TclError as e:
                if "bad window path name" not in str(e):
                    raise e

        try:
            ramp_speed = float(ramp_speed_entry.get())
            settle_time = float(settle_entry.get())
        except ValueError:
            tk.messagebox.showerror("Fehler", "Ungültige Eingabe für Ramp Speed oder Thermalization Time.")
            return

        temp_list_text = temp_list_entry.get().strip()
        try:
            if temp_list_text:
                temperatures = _parse_temperature_list(temp_list_text)
            else:
                start_temp = float(start_temp_entry.get())
                end_temp = float(end_temp_entry.get())
                intermediate_steps = int(steps_entry.get())
                temperatures = _generate_temperature_steps(start_temp, end_temp, intermediate_steps)
        except ValueError:
            tk.messagebox.showerror("Fehler", "Ungültige Temperatureingabe.")
            return

        if not temperatures:
            tk.messagebox.showerror("Fehler", "Bitte mindestens eine Temperatur angeben.")
            return

        user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name for SMU1:")
        if not user_input:
            return
        use_smu2 = use_smu2_var.get()
        user_input2 = None
        if use_smu2:
            user_input2 = simpledialog.askstring("Insert Filename", "Please insert the file Name for SMU2:")
            if not user_input2:
                return

        for target_temp in temperatures:
            if target_temp < 0.5:
                if getattr(temperature_control, "is_active", False):
                    temperature_control.stop()
                    time.sleep(0.5)
                try:
                    adr_control.start_adr(setpoint=target_temp, ramp=ramp_speed,
                                          adr_mode=None, operation_mode='cadr',
                                          auto_regenerate=True, pre_regenerate=True)
                except Exception:
                    pass
            else:
                temperature_control.start((target_temp, ramp_speed))

            while abs((client.query('T_sample.kelvin') or 300) - target_temp) > 0.01:
                time.sleep(0.2)
            time.sleep(max(0.0, settle_time))
            temperature_str = f"{client.query('T_sample.kelvin') or 300:.3f}"
            user_input_temperature = f"{user_input}_{temperature_str}K_"
            if use_smu2:
                user_input_temperature2 = f"{user_input2}_{temperature_str}K_"
                run_current_sweep(sweep_points, user_input_temperature, user_input_temperature2)
            else:
                run_current_sweep(sweep_points, user_input_temperature)

        if getattr(temperature_control, "is_active", False):
            temperature_control.stop()
        current_temp_window.destroy()
        k.smua.source.output = 0
        k2.smua.source.output = 0
        global measurement
        measurement = 0

    controls_frame = tk.Frame(current_temp_window)
    controls_frame.pack(fill=tk.X)

    local_remote_checkbox = tk.Checkbutton(controls_frame, text="4 Point", variable=local_remote_var)
    local_remote_checkbox.pack(side=tk.LEFT)
    smu2_checkbox = tk.Checkbutton(controls_frame, text="Use SMU2", variable=use_smu2_var)
    smu2_checkbox.pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Add Section (Current)", command=add_current_section).pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Measurement start", command=run_current_temp_sweep).pack(side=tk.RIGHT)

    return current_temp_window

#%% Cell 7: 4 Point Tc Measurement Force
#4PointTcMeasurement

@dataclass
class Tc4ptUI:
    sections_frame: object
    section_frames: list
    measurement_started: object
    tc_window: object
    temperature_data: list
    resistance_data: list

def _tc4pt_add_section(ui: Tc4ptUI):
    frame = create_tc_frame(ui.sections_frame)
    frame.pack(fill=tk.X)
    ui.section_frames.append(frame)

def _tc4pt_update_plot(temp_data, resistance_data, line, ax, fig, _frame):
    line.set_data(temp_data, resistance_data)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.flush_events()

def _tc4pt_get_data(ui: Tc4ptUI):
    tc_data = []
    maximumProgress = 0
    offset = 0
    for frame in ui.section_frames:
        try:
            start_temp = float(frame.winfo_children()[1].get())
            end_temp = float(frame.winfo_children()[3].get())
            ramp_speed = float(frame.winfo_children()[5].get())
            ramp_current = float(frame.winfo_children()[7].get()) / 1000000
            tc_data.append((start_temp, end_temp, ramp_speed, ramp_current))
            maximumProgress = maximumProgress + abs(end_temp - start_temp)
        except ValueError:
            print("Fehler")
            tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e
    print(tc_data)

    progress_window = tk.Toplevel(root)
    progress_window.title("Tc 4 Point Measurement Progress")
    progress_bar = ttk.Progressbar(progress_window, maximum=maximumProgress, length=300)
    progress_bar.pack(padx=20, pady=20)
    progress_window.update()
    global rt
    rt = ResultTable(
        column_titles=['Time', 'Current [A]', 'Voltage[V]','Temperature[K]'],
        units=[' ', "A", 'V','K'],
        params={'recorded': time.asctime(), 'sweep_type': 'tc'},
    )
    global rt2
    rt2 = ResultTable(
        column_titles=['Time', 'Current [A]', 'Voltage[V]','Temperature[K]'],
        units=[' ', "A", 'V','K'],
        params={'recorded': time.asctime(), 'sweep_type': 'tc'},
    )
    user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name:")
    NameWindow = str(user_input) + str(client.query('T_sample.kelvin') or 300)
    plot_window, fig, ax, line = create_plot_window_for_Resistance(NameWindow)
    ani = FuncAnimation(fig, partial(_tc4pt_update_plot, ui.temperature_data, ui.resistance_data, line, ax, fig),
                        frames=10000, repeat=False, interval=5000)
    ani._start()
    for start_temp, end_temp, ramp_speed, ramp_current in tc_data:
        if start_temp + 5 < temperature_control.kelvin and start_temp > 5:
            temperature_control.stop()
        else:
            temperature_control.start((start_temp, ramp_speed))
        if ui.measurement_started.get() == 1:
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k2.smua.source.output = 1
            k2.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)
            k2.apply_current(k2.smua, 0)
            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                voltage = k2.smua.measure.v()
                voltage2 = k.smua.measure.v()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                temperature = temperature_control.kelvin
                ui.temperature_data.append(temperature)
                ui.resistance_data.append(voltage / ramp_current)
                rt.append_row([timestamp, ramp_current, voltage, temperature])
                rt2.append_row([timestamp, ramp_current, voltage2, temperature])
                time.sleep(0.2)
        else:
            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                time.sleep(0.2)
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k2.smua.source.output = 1
            k2.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)
            k2.apply_current(k2.smua, 0)

        while abs(client.query('T_sample.kelvin') - end_temp) > 0.01:
            voltage = k2.smua.measure.v()
            voltage2 = k.smua.measure.v()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            temperature = temperature_control.kelvin
            ui.temperature_data.append(temperature)
            ui.resistance_data.append(voltage / ramp_current)
            rt.append_row([timestamp, ramp_current, voltage, temperature])
            rt2.append_row([timestamp, ramp_current, voltage2, temperature])
            progress_bar['value'] = abs(temperature - start_temp) + offset
            progress_window.update()
            time.sleep(0.2)
        offset = offset + abs(end_temp - start_temp)

    data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
    safe_file(data_numeric, user_input)
    data_numeric2 = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt2.data])
    user_input2 = user_input + "CurrentSMUVoltage"
    safe_file(data_numeric2, user_input2)
    ui.tc_window.destroy()
    progress_window.destroy()
    k.smua.source.output = 0
    k2.smua.source.output = 0
    global measurement
    measurement = 0
    try:
        ani._stop()
    except AttributeError:
        pass

# New function to create the 4p Tc measurement window
def create_4ptc_measurement():
    global measurement 
    measurement = 1
    k.smua.sense = k.smua.SENSE_LOCAL
    k2.smua.sense = k2.smua.SENSE_LOCAL
    section_frames = []

    global temperature_data
    global resistance_data
    temperature_data = []
    resistance_data = []

    tc_window = tk.Toplevel(root)
    tc_window.title("Tc Measurement")
    tc_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(tc_window))

    sections_frame = tk.Frame(tc_window)
    sections_frame.grid(row=0, column=0, padx=10, pady=10)

    ui = Tc4ptUI(
        sections_frame=sections_frame,
        section_frames=section_frames,
        measurement_started=measurement_started,
        tc_window=tc_window,
        temperature_data=temperature_data,
        resistance_data=resistance_data,
    )

    _tc4pt_add_section(ui)  # Add one section to begin with

    controls_frame = tk.Frame(tc_window)
    controls_frame.grid(row=1, column=0, padx=10, pady=10)

    # Place the Checkbutton and buttons in the controls_frame using grid
    tk.Checkbutton(controls_frame, text="Start Measurement", variable=measurement_started).grid(row=0, column=0, columnspan=2)
    tk.Button(controls_frame, text="Add Section", command=partial(_tc4pt_add_section, ui)).grid(row=0, column=2)
    tk.Button(controls_frame, text="Measurement start", command=partial(_tc4pt_get_data, ui)).grid(row=0, column=3)
    
    # Additional widgets or layout adjustments can be made here if needed
    
    # Optionally, set column and row weights to control resizing behavior
    tc_window.columnconfigure(0, weight=1)
    tc_window.rowconfigure(0, weight=1)


#%% Cell 8: 4 Point Tc Measurement with Sense
#4PT Sense

@dataclass
class TcSenseUI:
    sections_frame: object
    section_frames: list
    measurement_started: object
    tc_window: object

def _tc_sense_add_section(ui: TcSenseUI):
    frame = create_tc_frame(ui.sections_frame)
    frame.pack(fill=tk.X)
    ui.section_frames.append(frame)

def _tc_sense_update_plot(temp_data, resistance_data, line, ax, fig, _frame):
    if temp_data:
        line.set_data(temp_data, resistance_data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.flush_events()

def _tc_sense_get_data(ui: TcSenseUI):
    tc_data = []
    maximumProgress = 0
    offset = 0
    for frame in ui.section_frames:
        try:
            start_temp = float(frame.winfo_children()[1].get())
            if 0.08 < start_temp < 290:
                None
            else:
                Errormessage("Out of Value Range 0.2-290K")
                if start_temp > 290:
                    start_temp = 290
                else:
                    start_temp = 0.2
            end_temp = float(frame.winfo_children()[3].get())
            if 0.08 < end_temp < 290:
                None
            else:
                Errormessage("Out of Value Range 0.2-290K")
                if end_temp > 290:
                    end_temp = 290
                else:
                    end_temp = 0.2
            ramp_speed = float(frame.winfo_children()[5].get())
            ramp_current = float(frame.winfo_children()[7].get()) / 1000000
            tc_data.append((start_temp, end_temp, ramp_speed, ramp_current))
            maximumProgress = maximumProgress + abs(end_temp - start_temp)
        except ValueError:
            print("Fehler")
            tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
            return
        except tk.TclError as e:
            if "bad window path name" not in str(e):
                raise e

    print(tc_data)
    progress_window = tk.Toplevel(root)
    progress_window.title("4P Sense Measurement Progress")
    progress_bar = ttk.Progressbar(progress_window, maximum=maximumProgress, length=300)
    progress_bar.pack(padx=20, pady=20)
    progress_window.update()
    global rt
    rt = ResultTable(
        column_titles=['Time',  'Current [A]','Voltage[V]','Temperature[K]'],
        units=[' ',  'V',"A",'K'],
        params={'recorded': time.asctime(), 'sweep_type': 'tc'},
    )
    user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name:")
    NameWindow = str(user_input)
    plot_window, fig, ax, line = create_plot_window_for_Resistance(NameWindow)

    Temp_data = []
    Resistance_data = []

    ani = FuncAnimation(fig, partial(_tc_sense_update_plot, Temp_data, Resistance_data, line, ax, fig),
                        frames=10000, repeat=False, interval=5000)
    ani._start()
    for start_temp, end_temp, ramp_speed, ramp_current in tc_data:
        if start_temp + 5 < temperature_control.kelvin and start_temp > 5:
            pass
        else:
            temperature_control.start((start_temp, ramp_speed))
        if ui.measurement_started.get() == 1:
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)

            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                current, voltage = k.smua.measure.iv()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                temperature = temperature_control.kelvin
                Temp_data.append(temperature)
                rt.append_row([timestamp, current, voltage, temperature])
                Resistance_data.append(voltage / current)
                progress_bar['value'] = abs(temperature - start_temp) + offset
                progress_window.update()
                time.sleep(0.2)
        else:
            while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                time.sleep(0.2)
            temperature_control.start((end_temp, ramp_speed))
            k.smua.source.output = 1
            k.smua.source.func = 2
            k.apply_current(k.smua, ramp_current)

        while abs(client.query('T_sample.kelvin') - end_temp) > 0.01:
            current, voltage = k.smua.measure.iv()
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            temperature = temperature_control.kelvin
            Temp_data.append(temperature)
            rt.append_row([timestamp, current, voltage, temperature])
            Resistance_data.append(voltage / current)
            progress_bar['value'] = abs(temperature - start_temp) + offset
            progress_window.update()
            time.sleep(0.2)
        offset = offset + abs(end_temp - start_temp)

    data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
    safe_file(data_numeric, user_input)
    ui.tc_window.destroy()
    progress_window.destroy()
    k.smua.source.output = 0
    k2.smua.source.output = 0
    global measurement
    measurement = 0
    try:
        ani._stop()
    except AttributeError:
        pass


# New function to create the Tc measurement window
def create_4pt_measurement_Sense():
    global measurement 
    measurement = 1
    k.smua.sense = k.smua.SENSE_REMOTE
    section_frames = []
    tc_window = tk.Toplevel(root)
    tc_window.title("Tc Measurement")
    tc_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(tc_window))

    sections_frame = tk.Frame(tc_window)
    sections_frame.grid(row=0, column=0, padx=10, pady=10)

    ui = TcSenseUI(
        sections_frame=sections_frame,
        section_frames=section_frames,
        measurement_started=measurement_started,
        tc_window=tc_window,
    )

    _tc_sense_add_section(ui)  # Add one section to begin with

    controls_frame = tk.Frame(tc_window)
    controls_frame.grid(row=1, column=0, padx=10, pady=10)

    tk.Checkbutton(controls_frame, text="Start Measurement", variable=measurement_started).grid(row=0, column=0, columnspan=2)
    tk.Button(controls_frame, text="Add Section", command=partial(_tc_sense_add_section, ui)).grid(row=0, column=2)
    tk.Button(controls_frame, text="Measurement start", command=partial(_tc_sense_get_data, ui)).grid(row=0, column=3)

    tc_window.columnconfigure(0, weight=1)
    tc_window.rowconfigure(0, weight=1)

    
   

#%% Cell 9: Paralell Tc Measurement
#Paralell Tc
# New function to create the Tc measurement window
def create_parallel_tc_measurement():
    global measurement 
    measurement = 1    
    k.smua.sense = k.smua.SENSE_LOCAL
    k2.smua.sense = k2.smua.SENSE_LOCAL
    def add_tc_section():
        frame = create_tc_frame(sections_frame)
        frame.pack(fill=tk.X)
        section_frames.append(frame)
    
    def get_tc_data():
        tc_data = []
        maximumProgress = 0
        offset=0
        for frame in section_frames:
            try:              
                start_temp = float(frame.winfo_children()[1].get())
                if 0.199<start_temp <290:
                    None
                else:
                    Errormessage("Out of Value Range 0.2-290K")
                    if start_temp >290:
                        start_temp = 290
                    else:
                        start_temp = 0.2
                end_temp = float(frame.winfo_children()[3].get())
                if 0.2<end_temp <290:
                    None
                else:
                    Errormessage("Out of Value Range 0.2-290K")
                    if end_temp >290:
                        end_temp = 290
                    else:
                        end_temp = 0.2
                ramp_speed = float(frame.winfo_children()[5].get())
                ramp_current = float(frame.winfo_children()[7].get())/1000000
                tc_data.append((start_temp, end_temp, ramp_speed, ramp_current))
                maximumProgress = maximumProgress + abs(end_temp - start_temp)
            except ValueError:
                print("Fehler")
                tk.messagebox.showerror("Fehler", "Ungültige Eingabe")  # Fehlermeldung anzeigen
                return
            except tk.TclError as e:
                 #Handle the case where the frame was removed, and its widgets no longer exist
                 if "bad window path name" not in str(e):
                     raise e
        print(tc_data)  # For debugging, print the collected data
        
        progress_window = tk.Toplevel(root)  # New window for the progress bar
        progress_window.title("Tc Paralell Measurement Progress")
        progress_bar = ttk.Progressbar(progress_window, maximum=maximumProgress, length=300)
        progress_bar.pack(padx=20, pady=20)
        progress_window.update()  # Update the window to show the progress bar
        global rt
        rt = ResultTable(
            column_titles=['Time', 'Current [A]', 'Voltage[V]','Temperature[K]'],
            units=[' ', "A", 'V','K'],
            params={'recorded': time.asctime(), 'sweep_type': 'tc'},
        )
        global rt2
        rt2 = ResultTable(
            column_titles=['Time', 'Current [A]', 'Voltage[V]','Temperature[K]'],
            units=[' ', "A", 'V','K'],
            params={'recorded': time.asctime(), 'sweep_type': 'tc'},
        )
        user_input = simpledialog.askstring("Insert Filename for SMU 1", "Please insert the file Name:")
        user_input2 = simpledialog.askstring("Insert Filename for SMU 2", "Please insert the file Name:")
        
        #Live Plotting
        NameWindow = str(user_input)
        plot_window, fig, ax, line = create_plot_window_for_Resistance(NameWindow)
        NameWindow2 = str(user_input2)
        plot_window2, fig2, ax2, line2 = create_plot_window_for_Resistance(NameWindow2)
        
        Temp_data=[]
        Resistance_data=[]
        Resistance_data2=[]
        def update_R_plot(_):
            nonlocal Temp_data
            nonlocal Resistance_data
            if Temp_data:
                Temp_values, Resistance_values = Temp_data,Resistance_data
                line.set_data(Temp_values, Resistance_values)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.flush_events()
        ani = FuncAnimation(fig, update_R_plot, frames=10000, repeat=False, interval=5000)
        ani._start()
        def update_R_plot2(_):
            nonlocal Temp_data
            nonlocal Resistance_data2
            if Temp_data:
                Temp_values, Resistance_values = Temp_data,Resistance_data2
                line2.set_data(Temp_values, Resistance_values)
                ax2.relim()
                ax2.autoscale_view()
                fig2.canvas.flush_events()
        ani2 = FuncAnimation(fig2, update_R_plot2, frames=10000, repeat=False, interval=5000)
        ani2._start()
        for start_temp, end_temp, ramp_speed, ramp_current in tc_data:
            # Set initial temperature
            if start_temp+5<temperature_control.kelvin and start_temp >5:
                temperature_control.stop()
            else:    
                temperature_control.start((start_temp, ramp_speed))
            # Wait until the target start temperature is reached within 0.01K certainty
            temperature = temperature_control.kelvin
            if measurement_started.get() == 1:
                k.smua.source.output = 1
                k.smua.source.func = 2
                k.apply_current(k.smua, ramp_current)  # Apply the current
                k2.smua.source.output = 1
                k2.smua.source.func = 2
                k2.apply_current(k2.smua, ramp_current)  # Apply the current

                while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                    voltage = k.smua.measure.v()  # Measure the voltage
                    voltage2 = k2.smua.measure.v()  # Measure the voltage at SMU2
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current time
                    temperature = temperature_control.kelvin  # Get the current temperature
                    rt.append_row([timestamp, ramp_current, voltage, temperature])  # Append data to the ResultTable
                    rt2.append_row([timestamp, ramp_current, voltage2, temperature])  # Append data to the ResultTable
                    progress_bar['value'] = abs(temperature - start_temp) + offset  # Update the progress bar
                    progress_window.update()  # Update the window to show the progress
                    Temp_data.append(temperature)
                    Resistance_data.append(voltage/ramp_current)
                    Resistance_data2.append(voltage2/ramp_current)
                    time.sleep(0.2)  # Wait for 0.1 seconds
            else:
                k.smua.source.output = 1
                k.smua.source.func = 2
                k.apply_current(k.smua, ramp_current)  # Apply the current
                k2.smua.source.output = 1
                k2.smua.source.func = 2
                k2.apply_current(k2.smua, ramp_current)  # Apply the current

                while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                    voltage = k.smua.measure.v()  # Measure the voltage
                    voltage2 = k2.smua.measure.v()  # Measure the voltage at SMU2
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current time
                    temperature = temperature_control.kelvin  # Get the current temperature
                    rt.append_row([timestamp, ramp_current, voltage, temperature])  # Append data to the ResultTable
                    rt2.append_row([timestamp, ramp_current, voltage2, temperature])  # Append data to the ResultTable
                    progress_bar['value'] = abs(temperature - start_temp) + offset  # Update the progress bar
                    progress_window.update()  # Update the window to show the progress
                    Temp_data.append(temperature)
                    Resistance_data.append(voltage/ramp_current)
                    Resistance_data2.append(voltage2/ramp_current)
                    time.sleep(0.2)  # Wait for 0.1 seconds
                """while abs(client.query('T_sample.kelvin') - start_temp) > 0.1:
                    temperature = temperature_control.kelvin
                    progress_bar['value'] = abs(temperature - start_temp) + offset  # Update the progress bar
                    progress_window.update()  # Update the window to show the progress
                    time.sleep(0.2)  # Wait for 0.1 seconds"""
                k.smua.source.output = 1
                k.smua.source.func = 2
                k.apply_current(k.smua, ramp_current)  # Apply the current
                k2.smua.source.output = 1
                k2.smua.source.func = 2
                k2.apply_current(k2.smua, ramp_current)  # Apply the current
            # Set the next temperature
            temperature_control.start((end_temp, ramp_speed))# Apply current continuously and measure voltage until target end temperature is reached within 0.01K certainty

            last_execution_time = time.time()

            while abs(client.query('T_sample.kelvin') - end_temp) > 0.02:
                current_time = time.time()

                # Check if at least 0.2 seconds have passed since the last execution
                if current_time - last_execution_time >= 0.2:
                    last_execution_time = current_time  # Update the last execution time
                    voltage = k.smua.measure.v()  # Measure the voltage
                    voltage2 = k2.smua.measure.v()  # Measure the voltage at SMU2
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Get the current time
                    temperature = temperature_control.kelvin  # Get the current temperature
                    rt.append_row([timestamp, ramp_current, voltage, temperature])  # Append data to the ResultTable
                    rt2.append_row([timestamp, ramp_current, voltage2, temperature])  # Append data to the ResultTable
                    progress_bar['value'] = abs(temperature - start_temp) + offset  # Update the progress bar
                    progress_window.update()  # Update the window to show the progress
                    Temp_data.append(temperature)
                    Resistance_data.append(voltage/ramp_current)
                    Resistance_data2.append(voltage2/ramp_current)
                else:
                    # Sleep for a short duration (e.g., 0.01 seconds) to avoid busy-waiting
                    time.sleep(0.01)

            offset=offset+abs(end_temp - start_temp)
        
        # Format the data for saving
        data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
        data_numeric2 = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt2.data])
        safe_file(data_numeric,user_input)
        safe_file(data_numeric2,user_input2)
        tc_window.destroy()
        progress_window.destroy()  # Close the progress window
        k.smua.source.output = 0
        k2.smua.source.output = 0
        global measurement 
        measurement = 0
        try:
            ani._stop()
        except AttributeError:
            pass  # Ignore the AttributeError if animation has already run through
        try:
            ani2._stop()
        except AttributeError:
            pass  # Ignore the AttributeError if animation has already run through
    section_frames = []
    tc_window = tk.Toplevel(root)
    tc_window.title("Tc Measurement")
    tc_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(tc_window))

    sections_frame = tk.Frame(tc_window)
    sections_frame.grid(row=0, column=0, padx=10, pady=10)

    add_tc_section()  # Add one section to begin with

    controls_frame = tk.Frame(tc_window)
    controls_frame.grid(row=1, column=0, padx=10, pady=10)

    # Place the Checkbutton and buttons in the controls_frame using grid
    tk.Checkbutton(controls_frame, text="Start Measurement", variable=measurement_started).grid(row=0, column=0, columnspan=2)
    tk.Button(controls_frame, text="Add Section", command=add_tc_section).grid(row=0, column=2)
    tk.Button(controls_frame, text="Measurement start", command=get_tc_data).grid(row=0, column=3)
    
    # Additional widgets or layout adjustments can be made here if needed
    
    # Optionally, set column and row weights to control resizing behavior
    tc_window.columnconfigure(0, weight=1)
    tc_window.rowconfigure(0, weight=1)

#%% Cell 10: Parallel Tc Measurement 4-Point Sense (ADV patched with BASIC features)
#%% Cell 10: Parallel Tc Measurement 4-Point Sense (LOW-NOISE + SMU selection)
def create_parallel_4pt_measurement_Sense():
    """
    Parallel Tc measurement with:
      - Checkboxes to select SMU1, SMU2, or both
      - Low-noise configuration: NPLC, digital filter (repeat), autozero, optional fixed V-range
      - Extra averaging per log point (instrument filter + multiple reads)
      - Original ramp/ADR structure preserved
    """
    global measurement
    measurement = 1

    # -------------------- Config helpers --------------------
    
    def _conf_low_noise(dev, nplc=1.0, filter_count=10, autozero_on=True, v_range_V=None):
        """
        Configure Keithley 2635 for low-noise voltage readout while sourcing current:
          - Long integration (NPLC)
          - Digital filter (repeat, moving)
          - Autozero ON (slower but quieter)
          - Optional fixed voltage range (turns OFF autorange for cleaner data)
        """
        try:
            dev.smua.sense = dev.smua.SENSE_REMOTE
        except Exception:
            pass

        # NPLC
        try:
            dev.smua.measure.nplc = float(nplc)
        except Exception:
            pass

        # Autozero
        try:
            dev.smua.measure.autozero = dev.smua.AUTOZERO_ON if autozero_on else dev.smua.AUTOZERO_OFF
        except Exception:
            try:
                dev.smua.autozero = dev.smua.AUTOZERO_ON if autozero_on else dev.smua.AUTOZERO_OFF
            except Exception:
                pass

        # Fixed V-range (turn off autorange if provided)
        try:
            if v_range_V is not None and v_range_V > 0:
                # Some firmwares use measure.autorangev, some use source.autorangev (rare); try both patterns.
                try:
                    dev.smua.measure.autorangev = dev.smua.AUTORANGE_OFF
                except Exception:
                    pass
                try:
                    dev.smua.measure.rangev = float(v_range_V)
                except Exception:
                    pass
            else:
                try:
                    dev.smua.measure.autorangev = dev.smua.AUTORANGE_ON
                except Exception:
                    pass
        except Exception:
            pass

        # Digital filter (repeat, moving)
        try:
            dev.smua.measure.filter.type = dev.smua.FILTER_REPEAT
            dev.smua.measure.filter.count = int(max(1, filter_count))
            dev.smua.measure.filter.moving = 1  # moving average
            dev.smua.measure.filter.enable = 1
        except Exception:
            # Some APIs expose filter under smua.filter
            try:
                dev.smua.filter.type = dev.smua.FILTER_REPEAT
                dev.smua.filter.count = int(max(1, filter_count))
                dev.smua.filter.moving = 1
                dev.smua.filter.enable = 1
            except Exception:
                pass

        # Source mode (current source for Tc)
        try:
            dev.smua.source.output = 1
            dev.smua.source.func = 2  # current
        except Exception:
            pass

    def _meas_iv(dev, extra_reads=3):
        """
        Read (I,V) with additional averaging. The instrument's filter already averages;
        we add a few reads to reduce random noise further at slow ramps.
        """
        try:
            # First reading to settle filter; then average a few.
            i0, v0 = dev.smua.measure.iv()
            if extra_reads <= 1:
                return i0, v0
            vs, is_ = [v0], [i0]
            for _ in range(extra_reads - 1):
                i, v = dev.smua.measure.iv()
                is_.append(i); vs.append(v)
            return (float(np.mean(is_)), float(np.mean(vs)))
        except Exception:
            # Fallback separate reads
            try:
                i_list, v_list = [], []
                for _ in range(max(1, extra_reads)):
                    i_list.append(float(dev.smua.measure.i()))
                    v_list.append(float(dev.smua.measure.v()))
                return float(np.mean(i_list)), float(np.mean(v_list))
            except Exception:
                return float('nan'), float('nan')

    def _safe_output_off(dev):
        try:
            dev.smua.source.output = 0
        except Exception:
            pass

    # -------------------- Selection flags --------------------
    # These checkboxes let you choose SMU1 / SMU2 independently.
    smu1_on_var = tk.IntVar(master=root, value=1)
    smu2_on_var = tk.IntVar(master=root, value=1)

    # Defaults/noise options (you can tweak as you like)
    nplc_var          = tk.DoubleVar(master=root,value=1.0)   # longer = quieter
    filt_count_var    = tk.IntVar(master=root,value=10)       # instrument average count
    autozero_var      = tk.IntVar(master=root,value=1)        # ON = 1 (quieter)
    vrange_mV_var     = tk.StringVar(master=root,value="10")  # fixed V-range in mV (empty = autorange)
    extra_reads_var   = tk.IntVar(master=root,value=3)        # additional host-side averaging per data point

    # -------------------- Existing sense defaults --------------------
    # Set 4-wire remote sense by default; we’ll still apply per-SMU config later.
    try:
        k.smua.sense = k.smua.SENSE_REMOTE
    except Exception:
        pass
    try:
        k2.smua.sense = k2.smua.SENSE_REMOTE
    except Exception:
        pass

    # Flags für manuellen Modus (Advanced-Feature)
    manual_mode = False
    manual_save_pressed = False

    # ---------------- UI: Temperatur-Abschnitte + "Regenerate" pro Abschnitt ----------------
    def add_tc_section():
        frame = create_tc_frame(sections_frame)
        frame.pack(fill=tk.X)
        # ADR-Regeneration pro Abschnitt
        regenerate_var = tk.IntVar(master=root)
        tk.Checkbutton(frame, text="Regenerate", variable=regenerate_var).grid(row=0, column=8)
        frame.regenerate_var = regenerate_var
        section_frames.append(frame)

    # ---------------- Helper ----------------
    def _clamp_T(T):
        try:
            T = float(T)
        except ValueError:
            return None
        return min(290.0, max(0.085, T))

    def apply_currents(c):
        # Only apply on selected devices
        if smu1_on_var.get():
            try:
                k.smua.source.output = 1
                k.smua.source.func = 2
                k.apply_current(k.smua, c)
            except Exception:
                try:
                    k.smua.source.leveli = c
                except Exception:
                    pass
        if smu2_on_var.get():
            try:
                k2.smua.source.output = 1
                k2.smua.source.func = 2
                k2.apply_current(k2.smua, c)
            except Exception:
                try:
                    k2.smua.source.leveli = c
                except Exception:
                    pass

    def disable_currents():
        if smu1_on_var.get():
            _safe_output_off(k)
        if smu2_on_var.get():
            _safe_output_off(k2)

    def _estimate_temp_rate(duration=2.0, period=0.2):
        t0 = time.time()
        try:
            T_prev = float(temperature_control.kelvin)
        except Exception:
            T_prev = float(client.query('T_sample.kelvin') or 300.0)
        t_prev = t0
        rates = []
        while time.time() - t0 < duration:
            time.sleep(period)
            try:
                T = float(temperature_control.kelvin)
            except Exception:
                T = float(client.query('T_sample.kelvin') or 300.0)
            t = time.time()
            dTdt = abs(T - T_prev) / max(t - t_prev, 1e-6)
            rates.append(dTdt)
            T_prev, t_prev = T, t
        return np.median(rates) if rates else 0.0

    # ---------------- Kernfunktion ----------------
    def get_tc_data():
        nonlocal manual_mode, manual_save_pressed
        progress_window = None  # damit wir im finally sicher zerstören können

        # Validate SMU selection
        if (smu1_on_var.get() == 0) and (smu2_on_var.get() == 0):
            messagebox.showerror("Selection", "Please select at least one SMU (SMU1 and/or SMU2).")
            return

        try:
            # Buttons umschalten (Advanced)
            add_section_button.grid_forget()
            start_button.grid_forget()
            manual_mode_button.config(state=tk.NORMAL)
            manual_save_button.config(state=tk.NORMAL)

            # Low-noise config values
            nplc_val        = float(nplc_var.get())
            filt_count      = int(max(1, filt_count_var.get()))
            autozero_on     = bool(autozero_var.get())
            vr_txt          = vrange_mV_var.get().strip()
            v_range_V       = float(vr_txt) * 1e-3 if vr_txt else None
            extra_reads     = int(max(1, extra_reads_var.get()))

            # Configure selected SMUs for low noise
            if smu1_on_var.get():
                _conf_low_noise(k, nplc=nplc_val, filter_count=filt_count, autozero_on=autozero_on, v_range_V=v_range_V)
            if smu2_on_var.get():
                _conf_low_noise(k2, nplc=nplc_val, filter_count=filt_count, autozero_on=autozero_on, v_range_V=v_range_V)

            # Abschnitte einsammeln
            tc_data = []
            maximumProgress = 0.0
            offset = 0.0

            for frame in section_frames:
                try:
                    w = frame.winfo_children()
                    start_temp = _clamp_T(w[1].get())
                    end_temp   = _clamp_T(w[3].get())
                    ramp_speed = float(w[5].get())
                    ramp_current = float(w[7].get()) / 1e6  # µA -> A
                    regenerate = int(frame.regenerate_var.get())
                    if start_temp is None or end_temp is None:
                        tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
                        return
                    tc_data.append((start_temp, end_temp, ramp_speed, ramp_current, regenerate))
                    maximumProgress += abs(end_temp - start_temp)
                except Exception:
                    tk.messagebox.showerror("Fehler", "Ungültige Eingabe")
                    return

            if maximumProgress == 0 and not manual_mode:
                tk.messagebox.showerror("Fehler", "Keine gültigen Temperaturabschnitte gefunden.")
                return

            # Progress/ETA UI
            progress_window = tk.Toplevel(root)
            progress_window.title("Tc Parallel Measurement Progress")
            progress_bar = ttk.Progressbar(progress_window, maximum=maximumProgress or 1, length=320)
            progress_bar.pack(padx=20, pady=(10, 5))
            time_label = tk.Label(progress_window, text="ETA: calculating...")
            time_label.pack(padx=20, pady=(0, 10))
            progress_window.update()

            start_time = time.time()

            # ResultTables (only for selected SMUs)
            global rt, rt2
            rt = rt2 = None
            if smu1_on_var.get():
                rt  = ResultTable(column_titles=['Time','Current [A]','Voltage[V]','Temperature[K]'],
                                  units=[' ','A','V','K'],
                                  params={'recorded': time.asctime(), 'sweep_type': 'tc_parallel'})
            if smu2_on_var.get():
                rt2 = ResultTable(column_titles=['Time','Current [A]','Voltage[V]','Temperature[K]'],
                                  units=[' ','A','V','K'],
                                  params={'recorded': time.asctime(), 'sweep_type': 'tc_parallel'})

            # Dateinamen (ask only for the selected ones)
            user_input1 = user_input2 = None
            if smu1_on_var.get():
                user_input1 = simpledialog.askstring("Insert Filename for SMU 1", "Please insert the file name:")
            if smu2_on_var.get():
                user_input2 = simpledialog.askstring("Insert Filename for SMU 2", "Please insert the file name:")

            # Live-Plots R(T) for the selected ones
            Temp_data = []
            Resistance_data1, Resistance_data2 = [], []
            ani1 = ani2 = None
            if smu1_on_var.get():
                plot_window1, fig1, ax1, line1 = create_plot_window_for_Resistance(str(user_input1))
                def update_plot1(_):
                    if Temp_data:
                        line1.set_data(Temp_data, Resistance_data1)
                        ax1.relim(); ax1.autoscale_view(); fig1.canvas.flush_events()
                ani1 = FuncAnimation(fig1, update_plot1, frames=10000, repeat=False, interval=3000); ani1._start()

            if smu2_on_var.get():
                plot_window2, fig2, ax2, line2 = create_plot_window_for_Resistance(str(user_input2))
                def update_plot2(_):
                    if Temp_data:
                        line2.set_data(Temp_data, Resistance_data2)
                        ax2.relim(); ax2.autoscale_view(); fig2.canvas.flush_events()
                ani2 = FuncAnimation(fig2, update_plot2, frames=10000, repeat=False, interval=3000); ani2._start()

            total_target = maximumProgress
            # ---------------- Manueller Modus ----------------
            if manual_mode:
                if tc_data:
                    apply_currents(tc_data[0][3])  # ersten Segmentstrom nutzen
                while not manual_save_pressed:
                    temp = float(temperature_control.kelvin) if hasattr(temperature_control, 'kelvin') else float(client.query('T_sample.kelvin') or 300.0)
                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    if smu1_on_var.get():
                        i1, v1 = _meas_iv(k, extra_reads=extra_reads)
                        rt.append_row([ts, i1, v1, temp])
                        Resistance_data1.append(v1/i1 if (i1 and abs(i1) > 0) else float('nan'))
                    if smu2_on_var.get():
                        i2, v2 = _meas_iv(k2, extra_reads=extra_reads)
                        rt2.append_row([ts, i2, v2, temp])
                        Resistance_data2.append(v2/i2 if (i2 and abs(i2) > 0) else float('nan'))

                    Temp_data.append(temp)

                    elapsed = time.time() - start_time
                    hrs, rem = divmod(int(elapsed), 3600); mins, secs = divmod(rem, 60)
                    time_label.config(text=f"Elapsed: {hrs}h {mins}m {secs}s")
                    progress_window.update_idletasks(); progress_window.update()

                    # We’re slow-ramping. 0.2 s cadence is fine and low-noise friendly.
                    time.sleep(0.2)

            # ---------------- Strukturierte Messung ----------------
            else:
                for start_temp, end_temp, ramp_speed, ramp_current, regenerate in tc_data:
                    if manual_save_pressed:
                        break

                    # Segmentbeginn: aktuelle Temperatur & Annäherung bestimmen
                    T_entry = float(client.query('T_sample.kelvin') or 300.0)
                    approach_len = abs(T_entry - start_temp)
                    total_target += approach_len
                    progress_bar['maximum'] = max(progress_bar['maximum'], total_target)
                    segment_offset_base = 0.0 if 'offset' not in locals() else offset

                    # Steuervariante wählen
                    if (start_temp > 4.0) and (T_entry > 10.0):
                        if getattr(temperature_control, "is_active", False):
                            temperature_control.stop(); time.sleep(0.5)
                    elif start_temp < 0.3:
                        if getattr(temperature_control, "is_active", False):
                            temperature_control.stop(); time.sleep(0.5)
                        adr_control.start_adr(setpoint=start_temp, ramp=ramp_speed,
                                              adr_mode=None, operation_mode='cadr',
                                              auto_regenerate=True, pre_regenerate=bool(regenerate))
                    else:
                        temperature_control.start((start_temp, ramp_speed))

                    # Messung während Annäherung?
                    if measurement_started.get() == 1:
                        apply_currents(ramp_current)
                    else:
                        disable_currents()

                    # Timeout Annäherung
                    SAFETY = 20.0
                    eff_rate = max(_estimate_temp_rate(duration=2.0, period=0.2), 1e-4)  # K/s
                    max_timeout_approach = SAFETY * (approach_len / eff_rate)
                    t_approach0 = time.time()

                    # Annäherungsschleife
                    while abs(float(client.query('T_sample.kelvin') or 300.0) - start_temp) > 0.1:
                        if manual_save_pressed:
                            break

                        T_now = float(temperature_control.kelvin) if hasattr(temperature_control, 'kelvin') else float(client.query('T_sample.kelvin') or 300.0)
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        if measurement_started.get() == 1:
                            if smu1_on_var.get():
                                i1, v1 = _meas_iv(k, extra_reads=extra_reads)
                                rt.append_row([ts, i1, v1, T_now])
                                Resistance_data1.append(v1/i1 if (i1 and abs(i1) > 0) else float('nan'))
                            if smu2_on_var.get():
                                i2, v2 = _meas_iv(k2, extra_reads=extra_reads)
                                rt2.append_row([ts, i2, v2, T_now])
                                Resistance_data2.append(v2/i2 if (i2 and abs(i2) > 0) else float('nan'))
                            Temp_data.append(T_now)

                        # Fortschritt: Annäherung
                        current_progress = segment_offset_base + abs(T_entry - T_now)
                        progress_bar['value'] = min(current_progress, total_target)

                        # ETA
                        elapsed = time.time() - start_time
                        if current_progress > 0:
                            rem_time = elapsed * (total_target - current_progress) / max(current_progress, 1e-12)
                            hrs, rem = divmod(int(rem_time), 3600); mins, secs = divmod(rem, 60)
                            time_label.config(text=f"ETA: {hrs}h {mins}m {secs}s")

                        progress_window.update_idletasks(); progress_window.update()
                        time.sleep(0.2)

                        if (time.time() - t_approach0) > max_timeout_approach:
                            print("Maximum timeout reached while approaching start T")
                            break

                    # Annäherung abgeschlossen
                    offset = segment_offset_base + approach_len

                    # Strom an, falls bisher nicht aktiv
                    if measurement_started.get() == 0:
                        apply_currents(ramp_current)

                    # Endtemperatur anfahren
                    if end_temp < 0.3:
                        if getattr(temperature_control, "is_active", False):
                            temperature_control.stop(); time.sleep(0.5)
                        adr_control.start_adr(setpoint=end_temp, ramp=ramp_speed,
                                              adr_mode=None, operation_mode='cadr',
                                              auto_regenerate=True, pre_regenerate=bool(regenerate))
                    else:
                        temperature_control.start((end_temp, ramp_speed))

                    # Timeout für Rampe
                    ramp_len = abs(end_temp - start_temp)
                    eff_rate_ramp = max(_estimate_temp_rate(duration=2.0, period=0.2), 1e-4)
                    max_timeout_ramp = SAFETY * (ramp_len / eff_rate_ramp)
                    t_ramp0 = time.time()

                    # Rampenschleife
                    last_exec = time.time()
                    while abs(float(client.query('T_sample.kelvin') or 300.0) - end_temp) > 0.02:
                        if manual_save_pressed:
                            break
                        now = time.time()
                        if now - last_exec >= 0.2:
                            last_exec = now

                            T_now = float(temperature_control.kelvin) if hasattr(temperature_control, 'kelvin') else float(client.query('T_sample.kelvin') or 300.0)
                            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                            if smu1_on_var.get():
                                i1, v1 = _meas_iv(k, extra_reads=extra_reads)
                                rt.append_row([ts, i1, v1, T_now])
                                Resistance_data1.append(v1/i1 if (i1 and abs(i1) > 0) else float('nan'))
                            if smu2_on_var.get():
                                i2, v2 = _meas_iv(k2, extra_reads=extra_reads)
                                rt2.append_row([ts, i2, v2, T_now])
                                Resistance_data2.append(v2/i2 if (i2 and abs(i2) > 0) else float('nan'))
                            Temp_data.append(T_now)

                            # Fortschritt: Rampe
                            ramp_done = abs(T_now - start_temp)
                            current_progress = segment_offset_base + approach_len + ramp_done
                            progress_bar['value'] = min(current_progress, total_target)

                            # ETA
                            elapsed = time.time() - start_time
                            if current_progress > 0:
                                rem_time = elapsed * (total_target - current_progress) / max(current_progress, 1e-12)
                                hrs, rem = divmod(int(rem_time), 3600); mins, secs = divmod(rem, 60)
                                time_label.config(text=f"ETA: {hrs}h {mins}m {secs}s")

                            progress_window.update_idletasks(); progress_window.update()
                        else:
                            time.sleep(0.01)

                        if (time.time() - t_ramp0) > max_timeout_ramp:
                            print("Maximum timeout reached during ramp")
                            break

                    # Segment vollständig
                    offset = segment_offset_base + approach_len + ramp_len
                    if manual_save_pressed:
                        break

            # Daten speichern (DDMMYYYY ans Ende des Namens)
            date_tag = datetime.datetime.now().strftime('%d%m%Y')
            if smu1_on_var.get() and rt is not None and user_input1:
                data1 = np.array([[ts] + list(map(float, vals)) for ts, *vals in rt.data], dtype=object)
                safe_file(data1, f"{user_input1}_{date_tag}")
            if smu2_on_var.get() and rt2 is not None and user_input2:
                data2 = np.array([[ts] + list(map(float, vals)) for ts, *vals in rt2.data], dtype=object)
                safe_file(data2, f"{user_input2}_{date_tag}")

        finally:
            # Aufräumen
            try: disable_currents()
            except: pass
            try:
                if progress_window and progress_window.winfo_exists():
                    progress_window.destroy()
            except: pass
            try:
                if tc_window and tc_window.winfo_exists():
                    tc_window.destroy()
            except: pass
            try:
                if 'ani1' in locals() and ani1: ani1._stop()
            except: pass
            try:
                if 'ani2' in locals() and ani2: ani2._stop()
            except: pass
            global measurement
            measurement = 0

    # ---------------- Handlers manueller Modus ----------------
    def manual_mode_toggle():
        nonlocal manual_mode
        manual_mode = True
        manual_mode_button.config(state=tk.DISABLED)

    def manual_save():
        nonlocal manual_save_pressed
        manual_save_pressed = True

    # ---------------- GUI bauen ----------------
    section_frames = []
    tc_window = tk.Toplevel(root)
    tc_window.title("Parallel Tc Measurement (Low-Noise)")
    tc_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(tc_window))

    # SMU selection & noise options
    top_opts = ttk.LabelFrame(tc_window, text="Devices & Noise Options")
    top_opts.pack(padx=10, pady=(10, 0), fill='x')

    tk.Checkbutton(top_opts, text="Use SMU1", variable=smu1_on_var).grid(row=0, column=0, padx=6, pady=4, sticky="w")
    tk.Checkbutton(top_opts, text="Use SMU2", variable=smu2_on_var).grid(row=0, column=1, padx=6, pady=4, sticky="w")

    tk.Label(top_opts, text="NPLC").grid(row=1, column=0, sticky="e")
    tk.Entry(top_opts, textvariable=nplc_var, width=8).grid(row=1, column=1, sticky="w", padx=4)

    tk.Label(top_opts, text="Filter count").grid(row=1, column=2, sticky="e")
    tk.Entry(top_opts, textvariable=filt_count_var, width=8).grid(row=1, column=3, sticky="w", padx=4)

    tk.Checkbutton(top_opts, text="Autozero ON", variable=autozero_var).grid(row=1, column=4, padx=6, pady=4, sticky="w")

    tk.Label(top_opts, text="Fixed V-range (mV, empty=auto)").grid(row=2, column=0, sticky="e")
    tk.Entry(top_opts, textvariable=vrange_mV_var, width=10).grid(row=2, column=1, sticky="w", padx=4)

    tk.Label(top_opts, text="Extra reads / point").grid(row=2, column=2, sticky="e")
    tk.Entry(top_opts, textvariable=extra_reads_var, width=8).grid(row=2, column=3, sticky="w", padx=4)

    for c in range(5):
        top_opts.grid_columnconfigure(c, weight=0)

    sections_frame = tk.Frame(tc_window)
    sections_frame.pack(padx=10, pady=10, fill='x')
    add_tc_section()

    controls = tk.Frame(tc_window)
    controls.pack(padx=10, pady=10)

    add_section_button = tk.Button(controls, text="Add Section", command=add_tc_section)
    add_section_button.grid(row=0, column=0, padx=5)

    start_button = tk.Button(controls, text="Start Measurement", command=get_tc_data)
    start_button.grid(row=0, column=1, padx=5)

    manual_mode_button = tk.Button(controls, text="Manual Mode", command=manual_mode_toggle, state=tk.DISABLED)
    manual_mode_button.grid(row=0, column=2, padx=5)

    manual_save_button = tk.Button(controls, text="Manual Save", command=manual_save, state=tk.DISABLED)
    manual_save_button.grid(row=0, column=3, padx=5)

    tk.Checkbutton(controls, text="Start Measurement (measure during approach)", variable=measurement_started)\
        .grid(row=1, column=0, columnspan=4, pady=(8,0))

    return tc_window


    

#%% Cell 11: Live Plotting
def create_plot_window_for_Resistance(Name):
    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title(Name)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-o', label='Resistance vs. Temperature')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Resistance (Ohms)')
    ax.set_title('ResistancePlot')
    ax.legend(loc='upper right')

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack()
    
    return plot_window, fig, ax, line
def create_plot_window_for_Current_Voltage(Name):
    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title(Name)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-o', label='Voltage vs. Current')
    ax.set_xlabel('Voltage(V)')
    ax.set_ylabel('Current(A)')
    ax.set_title('IV Plot')
    ax.legend(loc='upper right')

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack()
    
    return plot_window, fig, ax, line

def create_plot_window_for_Voltage_Current(Name):
    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title(Name)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-o', label='Current vs. Voltage')
    ax.set_xlabel('Current(A)')
    ax.set_ylabel('Voltage(V)')
    ax.set_title('IV Plot')
    ax.legend(loc='upper right')

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack()
    
    return plot_window, fig, ax, line
def create_plot_window_for_Temperature():
    # Create a new window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title("Temperature Sample ")

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '-o', label='Time vs Temperature')
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('Temperature[K]')
    ax.set_title('Temperature')
    ax.legend(loc='upper right')

    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack()
    
    return plot_window, fig, ax, line
#%% Cell 12: IV Current Curve4PT

#IV sweeps
def load_sweep_current():
    global measurement 
    measurement = 1
    print("Before showing dialog")
    filename = filedialog.askopenfilename(initialdir=DEFAULT_PATH, title="Load Sweep",
                                      filetypes=[("Sweep-Dateien", "*.swp"), ("Alle Dateien", "*.*")])
    print(f"After showing dialog, filename: {filename}")

    measurement = 0
    if filename:
        with open(filename, 'r') as file:
            print("file opened")  # Debugging-Information
            sweep_points = json.load(file)
            print("sweep_points loaded")  # Debugging-Information

            # Create new sweep window
            sweep_window, section_frames, sections_frame = create_sweep_current()  # Catch the returned values

            # Delete the initial empty section
            section_frames[0].destroy()
            section_frames.clear()

            # Add sections for each loaded sweep point
            for start, end, steps in sweep_points:
                frame = add_predefined_section(sections_frame, start, end, steps)
                section_frames.append(frame)
                

def run_current_sweep(sweep_points, user_input,user_input2=None):
    global measurement
    measurement = 1
    if local_remote_var.get() == 1:
        k.smua.sense = k.smua.SENSE_REMOTE
        k2.smua.sense = k.smua.SENSE_REMOTE
    else:
        k.smua.sense = k.smua.SENSE_LOCAL
        k2.smua.sense = k.smua.SENSE_LOCAL
    total_steps = sum(steps for start, end, steps in sweep_points)  # Total number of steps across all sections

    progress_window = tk.Toplevel(root)  # New window for the progress bar
    progress_window.title("Sweep Progress")
    progress_bar = ttk.Progressbar(progress_window, maximum=total_steps, length=300)
    progress_bar.pack(padx=20, pady=20)
    progress_window.update()  # Update the window to show the progress bar
    global rt
    rt = ResultTable(
        column_titles=['Time',  'Current [A]','Voltage [V]', 'Temperature [K]'],
        units=[' ',  'A','V', 'K'],
        params={'recorded': time.asctime(), 'sweep_type': 'iv_current'},
    )

    current_step = 0
    NameWindow = str(user_input) + str(client.query('T_sample.kelvin') or 300)
    plot_window, fig, ax, line = create_plot_window_for_Voltage_Current(NameWindow)
    
    use_smu2 = use_smu2_var.get()
    if use_smu2:
        NameWindow2 = str(user_input2) + str(client.query('T_sample.kelvin') or 300)
        plot_window2, fig2, ax2, line2 = create_plot_window_for_Voltage_Current(NameWindow2)
        k2.smua.sense = k2.smua.SENSE_REMOTE
        rt2 = ResultTable(
            column_titles=['Time',  'Current [A]','Voltage [V]', 'Temperature [K]'],
            units=[' ',  'A','V', 'K'],
            params={'recorded': time.asctime(), 'sweep_type': 'iv_current'},
        )
    current_data = []
    voltage_data = []
    if use_smu2:
        voltage_data2 = []
        current_data2 =[]
    def update_iv_plot(_):
        nonlocal voltage_data
        nonlocal current_data
        if voltage_data:
            voltage_values, current_values = voltage_data, current_data
            line.set_data(current_values, voltage_values)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.flush_events()
    def update_iv_plot2(_):
        nonlocal voltage_data
        nonlocal current_data
        if voltage_data:
            voltage_values, current_values = voltage_data2, current_data2
            line2.set_data(current_values, voltage_values)
            ax2.relim()
            ax2.autoscale_view()
            fig2.canvas.flush_events()
    ani = FuncAnimation(fig, update_iv_plot, frames=10000, repeat=False, interval=5000)
    ani._start()
    if use_smu2:
        ani2 = FuncAnimation(fig2, update_iv_plot2, frames=10000, repeat=False, interval=5000)
        ani2._start()
    for start, end, steps in sweep_points:
        currents = [round(start + (end - start) * j / steps, 14) for j in range(steps)]
        k.smua.source.output = 1
        k.smua.source.func = 2 # Set source function to current
        k.smua.measure.nplc = integrationtime
        if use_smu2:
            k2.smua.source.output = 1
            k2.smua.source.func = 2  # Set source function to current
            k2.smua.measure.nplc = integrationtime
        for current in currents:
            k.apply_current(k.smua, current)
            if use_smu2:
                k2.apply_current(k2.smua, current)
            time.sleep(integrationtime)  # 0.5 seconds for the DUT to stabilize
            T = client.query('T_sample.kelvin') or 300 # T_sample should be defined and connected
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            voltage = k.smua.measure.v() # Measure voltage at smuA
            if use_smu2:
                voltage2 = k2.smua.measure.v()
                
            time.sleep(k.smua.measure.nplc)
            rt.append_row([timestamp,  current,voltage, T])
            if use_smu2:
                rt2.append_row([timestamp, current, voltage2, T])
                voltage_data2.append(voltage2)
                current_data2.append(current)
            current_step += 1
            progress_bar['value'] = current_step  # Update the progress bar
            progress_window.update()  # Update the window to show the progress
            current_data.append(current)
            voltage_data.append(voltage)
    
    # Save the data
    data_numeric = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt.data])
    if use_smu2:
        data_numeric2 = np.array([[timestamp] + list(map(float, values)) for timestamp, *values in rt2.data])
        safe_file(data_numeric2, user_input2)
    safe_file(data_numeric, user_input)
    
    progress_window.destroy()
    try:
        ani._stop()
    except AttributeError:
        pass  # Ignore the AttributeError if animation has already run through
def create_sweep_current():
    global measurement
    measurement = 1
    if local_remote_var.get() == 1:
        k.smua.sense = k.smua.SENSE_REMOTE
    else:
        k.smua.sense = k.smua.SENSE_LOCAL

    def add_section():
        frame = create_section_frame(sections_frame)
        frame.pack(fill=tk.X)
        section_frames.append(frame)

    def run_current():
        sweep_points = []
        if save_var.get():
            sweep_save_points =[]
        total_steps = 0
        for frame in section_frames:
            try:
                start_value = frame.winfo_children()[1].get()
                end_value = frame.winfo_children()[3].get()
                steps_value = frame.winfo_children()[5].get()
                print(f"Start: {start_value}, End: {end_value}, Steps: {steps_value}")  # Debugging-Information
                start = float(start_value)*(10**-6)
                end = float(end_value)*(10**-6)
                steps = int(steps_value)
                if save_var.get():
                    sweep_save_points.append((start*(10**6), end*(10**6), steps))
                sweep_points.append((start, end, steps))
                total_steps += steps
            except ValueError:
                print("Fehler")
                tk.messagebox.showerror("Fehler", "Ungültige Eingabe")  # Fehlermeldung anzeigen
                return  # Frühzeitig zurückkehren, um die Ausführung zu stoppen
            except tk.TclError as e:
                # Handle the case where the frame was removed, and its widgets no longer exist
                if "bad window path name" not in str(e):
                    raise e
        if save_var.get():  # Only save the data if the checkbox is checked
            save_sweep(sweep_save_points)
        user_input = simpledialog.askstring("Insert Filename", "Please insert the file Name for SMU1:")
        
        use_smu2var = use_smu2_var.get()
        if use_smu2var:  # Use get() to obtain the boolean value
            user_input2 = simpledialog.askstring("Insert Filename", "Please insert the file Name for SMU2:")
            run_current_sweep(sweep_points, user_input, user_input2)
        else:
            run_current_sweep(sweep_points, user_input)
        sweep_window.destroy()  # Planen Sie die Zerstörung des Fensters im Haupt-Thread
        k.smua.source.output = 0
        k2.smua.source.output = 0
        global measurement
        measurement = 0

    section_frames = []
    sweep_window = tk.Toplevel(root)
    sweep_window.title("Make new Current Sweep")
    sweep_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(sweep_window))
    sections_frame = tk.Frame(sweep_window)
    sections_frame.pack()

    add_section()  # Fügen Sie einen Abschnitt hinzu, um zu beginnen

    controls_frame = tk.Frame(sweep_window)
    controls_frame.pack(fill=tk.X)

    save_var = tk.IntVar()  # Create an IntVar to hold the state of the checkbox
    save_checkbutton = tk.Checkbutton(controls_frame, text="Save Measurement Settings", variable=save_var)
    save_checkbutton.pack(side=tk.LEFT)  # Adjust the side argument as needed
    # Add the checkbox for Local/Remote measurement here
    local_remote_checkbox = tk.Checkbutton(controls_frame, text="4 Point", variable=local_remote_var)
    local_remote_checkbox.pack(side=tk.LEFT)
    smu2_checkbox = tk.Checkbutton(controls_frame, text="Use SMU2", variable=use_smu2_var)
    smu2_checkbox.pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Add Section", command=add_section).pack(side=tk.LEFT)
    tk.Button(controls_frame, text="Sweep start", command=run_current).pack(side=tk.RIGHT)
    return sweep_window, section_frames, sections_frame


#%% FIND JJ — thread-safe UI (after), temp guard, SMU selectable, Ic & Ir annotated, dual-save
#%% FIND JJ — soft search + pretty sweep (dense around Ic), thread-safe UI, dual-save
#%% FIND JJ — soft search + pretty sweep (dense around Ic), thread-safe UI, dual-save
#%% FIND JJ — soft search + pretty sweep (dense around Ic), thread-safe UI, dual-save (+txt), reset plot, color by direction, multi-cycles hysteresis

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np, time, datetime, threading, os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def Find_JJ():
    # ---------- small helpers ----------
    def _tempK():
        try:
            return float(client.query('T_sample.kelvin') or 300.0)
        except Exception:
            return 300.0

    def _dated(base):
        return f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_save_paths():
        """Return (primary_path, secondary_path_or_None). Never raises."""
        try:
            p1 = SAVE_PATH
        except Exception:
            p1 = os.getcwd()
        p2 = None
        try:
            p2 = SAVE_PATH_2
        except Exception:
            p2 = None
        return p1, p2

    def _dual_save_csv_and_txt(base, suffix, header_cols, rows):
        """
        Always writes:
          - CSV: comma-separated, header line without '#'
          - TXT: tab-separated, header line without '#'
        to SAVE_PATH and (if set) SAVE_PATH_2.

        base: filename base WITHOUT extension
        suffix: e.g. "_SMU1_JJfind_search"
        header_cols: iterable of column names (strings)
        """
        csv_name = f"{base}{suffix}.csv"
        txt_name = f"{base}{suffix}.txt"

        header_csv = ",".join(header_cols)
        header_txt = "\t".join(header_cols)

        if rows:
            arr = np.asarray(rows, dtype=object)
        else:
            arr = np.empty((0, len(header_cols)), dtype=object)

        p1, p2 = _get_save_paths()
        for p in (p1, p2):
            if not p:
                continue
            try:
                os.makedirs(p, exist_ok=True)
            except Exception:
                pass

            # CSV
            try:
                out_csv = os.path.join(p, csv_name)
                np.savetxt(out_csv, arr, fmt="%s", delimiter=",", header=header_csv, comments="")
            except Exception:
                pass

            # TXT (tab)
            try:
                out_txt = os.path.join(p, txt_name)
                np.savetxt(out_txt, arr, fmt="%s", delimiter="\t", header=header_txt, comments="")
            except Exception:
                pass

    def _conf_smu(dev, nplc, vlimit_V):
        try: dev.smua.sense = dev.smua.SENSE_REMOTE
        except Exception: pass
        try: dev.smua.measure.nplc = float(nplc)
        except Exception: pass
        try:
            dev.smua.measure.autozero = dev.smua.AUTOZERO_ON
        except Exception:
            try: dev.smua.autozero = dev.smua.AUTOZERO_ON
            except Exception: pass
        try:
            dev.smua.source.output = 1
            dev.smua.source.func   = 2  # current source
        except Exception: pass
        try: dev.smua.source.limitv = float(vlimit_V)
        except Exception: pass
        # gentle digital filter
        try:
            dev.smua.measure.filter.type = dev.smua.FILTER_REPEAT
            dev.smua.measure.filter.count = 5
            dev.smua.measure.filter.moving = 1
            dev.smua.measure.filter.enable = 1
        except Exception: pass

    def _apply_I(dev, I_A):
        try: k.apply_current(dev.smua, float(I_A))
        except Exception:
            try: dev.smua.source.leveli = float(I_A)
            except Exception: pass

    def _meas_iv(dev):
        try:
            i, v = dev.smua.measure.iv()
            return float(i), float(v)
        except Exception:
            try: return float(dev.smua.measure.i()), float(dev.smua.measure.v())
            except Exception: return float('nan'), float('nan')

    def _output_off(dev):
        try: dev.smua.source.output = 0
        except Exception: pass

    def _soft_ramp_sequence(I_start_uA, I_max_uA, grow=1.3, min_step_uA=0.02):
        I = max(I_start_uA, min_step_uA)
        seq_uA = []
        while I <= I_max_uA * (1 + 1e-9):
            seq_uA.append(I)
            I = max(I + min_step_uA, I * grow)
        return [x * 1e-6 for x in seq_uA]  # -> A

    def _pretty_sweep_points(Ic_A, pts_total=300, dense_factor=4.0, over_factor=1.05):
        """
        Build current grid for a nice IV around Ic:
          - overall range: 0 → over_factor*Ic (up & down)
          - extra dense region near [0.9..1.02]*Ic
          - returns (up_points_A, down_points_A)
        """
        Ic = max(abs(Ic_A), 1e-12)
        Imax = over_factor * Ic

        z1 = (0.0, 0.5 * Ic)
        z2 = (0.5 * Ic, 0.9 * Ic)
        z3 = (0.9 * Ic, min(1.02 * Ic, Imax))
        z4 = (z3[1], Imax)

        base = max(10, int(pts_total / 10))
        n1 = base
        n2 = base * 2
        n3 = max(base, int(base * dense_factor))
        n4 = base

        def lin(a, b, n):
            if n <= 1:
                return [a, b]
            return list(np.linspace(a, b, n, endpoint=True))

        up = []
        for (a, b, n) in ((z1[0], z1[1], n1), (z2[0], z2[1], n2), (z3[0], z3[1], n3), (z4[0], z4[1], n4)):
            seg = lin(a, b, n)
            if up and seg and seg[0] == up[-1]:
                seg = seg[1:]
            up.extend(seg)

        up = [max(0.0, x) for x in up]
        down = list(reversed(up))
        return up, down

    def _apply_jitter(points_A, jitter_frac, keep_ends=True):
        """
        Jitter each point by +/- jitter_frac * max(points) (relative to sweep max current).
        keep_ends keeps first/last exactly unchanged (helps repeatability & return-to-zero).
        """
        if not points_A:
            return points_A
        Imax = max(points_A) if max(points_A) > 0 else 1.0
        pts = np.array(points_A, dtype=float)
        if jitter_frac and jitter_frac > 0:
            r = (np.random.rand(len(pts)) * 2.0 - 1.0) * (jitter_frac * Imax)
            pts = pts + r
            pts = np.clip(pts, 0.0, None)
        if keep_ends and len(pts) >= 2:
            pts[0] = points_A[0]
            pts[-1] = points_A[-1]
        return list(pts)

    # ---------- UI ----------
    win = tk.Toplevel(root)
    win.title("Find JJ (Ic, Ir, pretty sweep)")
    win.geometry("+140+100")

    stop_event = threading.Event()
    ui_closed = {'flag': False}

    def ui_alive():
        return (not ui_closed['flag']) and bool(win.winfo_exists())

    def on_close():
        ui_closed['flag'] = True
        stop_event.set()
        try: win.destroy()
        except Exception: pass

    win.protocol("WM_DELETE_WINDOW", on_close)

    opts = ttk.LabelFrame(win, text="Settings")
    opts.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)

    use1 = tk.IntVar(value=1)
    use2 = tk.IntVar(value=0)
    tk.Checkbutton(opts, text="Use SMU1", variable=use1).grid(row=0, column=0, sticky="w", padx=4, pady=2)
    tk.Checkbutton(opts, text="Use SMU2", variable=use2).grid(row=0, column=1, sticky="w", padx=4, pady=2)

    neg_branch = tk.IntVar(value=1)
    tk.Checkbutton(opts, text="Measure negative branch too", variable=neg_branch)\
        .grid(row=0, column=2, columnspan=2, sticky="w", padx=4)

    ttk.Label(opts, text="V_thresh (µV) for Ic").grid(row=1, column=0, sticky="e")
    e_vth = ttk.Entry(opts, width=10); e_vth.insert(0, "100"); e_vth.grid(row=1, column=1, sticky="w")

    ttk.Label(opts, text="V_retrap (µV)").grid(row=1, column=2, sticky="e")
    e_vrt = ttk.Entry(opts, width=10); e_vrt.insert(0, "50"); e_vrt.grid(row=1, column=3, sticky="w")

    ttk.Label(opts, text="ΔT max (K)").grid(row=2, column=0, sticky="e")
    e_dt = ttk.Entry(opts, width=10); e_dt.insert(0, "0.1"); e_dt.grid(row=2, column=1, sticky="w")

    ttk.Label(opts, text="I_start (µA)").grid(row=3, column=0, sticky="e")
    e_i0 = ttk.Entry(opts, width=10); e_i0.insert(0, "0.05"); e_i0.grid(row=3, column=1, sticky="w")

    ttk.Label(opts, text="I_max (µA)").grid(row=3, column=2, sticky="e")
    e_imax = ttk.Entry(opts, width=10); e_imax.insert(0, "1000"); e_imax.grid(row=3, column=3, sticky="w")

    ttk.Label(opts, text="Grow ×").grid(row=4, column=0, sticky="e")
    e_grow = ttk.Entry(opts, width=10); e_grow.insert(0, "1.3"); e_grow.grid(row=4, column=1, sticky="w")

    ttk.Label(opts, text="Min step (µA)").grid(row=4, column=2, sticky="e")
    e_min = ttk.Entry(opts, width=10); e_min.insert(0, "0.02"); e_min.grid(row=4, column=3, sticky="w")

    ttk.Label(opts, text="NPLC").grid(row=5, column=0, sticky="e")
    e_nplc = ttk.Entry(opts, width=10); e_nplc.insert(0, "0.2"); e_nplc.grid(row=5, column=1, sticky="w")

    ttk.Label(opts, text="V-limit (V)").grid(row=5, column=2, sticky="e")
    e_vlim = ttk.Entry(opts, width=10); e_vlim.insert(0, "0.02"); e_vlim.grid(row=5, column=3, sticky="w")

    ttk.Label(opts, text="Settle (ms)").grid(row=6, column=0, sticky="e")
    e_settle = ttk.Entry(opts, width=10); e_settle.insert(0, "2.0"); e_settle.grid(row=6, column=1, sticky="w")

    # Pretty sweep configuration
    prett = ttk.LabelFrame(win, text="Pretty sweep / Hysteresis cycles")
    prett.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,8))

    ttk.Label(prett, text="Pts total (per up or down)").grid(row=0, column=0, sticky="e")
    e_pts = ttk.Entry(prett, width=10); e_pts.insert(0, "300"); e_pts.grid(row=0, column=1, sticky="w")

    ttk.Label(prett, text="Dense factor near Ic").grid(row=0, column=2, sticky="e")
    e_dense = ttk.Entry(prett, width=10); e_dense.insert(0, "4.0"); e_dense.grid(row=0, column=3, sticky="w")

    ttk.Label(prett, text="Overdrive ×Ic (max)").grid(row=0, column=4, sticky="e")
    e_over = ttk.Entry(prett, width=10); e_over.insert(0, "1.05"); e_over.grid(row=0, column=5, sticky="w")

    ttk.Label(prett, text="Cycles (0→Ic→0 repeats)").grid(row=1, column=0, sticky="e")
    e_cycles = ttk.Entry(prett, width=10); e_cycles.insert(0, "1"); e_cycles.grid(row=1, column=1, sticky="w")

    ttk.Label(prett, text="Jitter (fraction of Imax)").grid(row=1, column=2, sticky="e")
    e_jitter = ttk.Entry(prett, width=10); e_jitter.insert(0, "0.005"); e_jitter.grid(row=1, column=3, sticky="w")

    # Buttons / status
    btns = ttk.Frame(win); btns.grid(row=3, column=0, sticky="ew", padx=10, pady=6)
    start_btn = ttk.Button(btns, text="Start")
    stop_btn  = ttk.Button(btns, text="Stop", state=tk.DISABLED)
    start_btn.pack(side=tk.RIGHT, padx=5); stop_btn.pack(side=tk.RIGHT, padx=5)

    status = ttk.Label(win, text="Ready.")
    status.grid(row=4, column=0, sticky="w", padx=10, pady=(0,8))

    # Plot
    fig = Figure(figsize=(6.8, 4.6), dpi=100)
    ax  = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw(); canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
    win.grid_rowconfigure(2, weight=1); win.grid_columnconfigure(0, weight=1)

    # Lines (will be re-created on reset)
    lines = {}

    def _make_plot_fresh(title="Find JJ"):
        ax.cla()
        ax.set_xlabel("I (A)")
        ax.set_ylabel("V (V)")
        ax.set_title(title)

        # Required behavior:
        # - away from 0: blue
        # - towards 0: orange
        #
        # Keep sign visually by marker:
        #  + branch = 'o'
        #  - branch = 'x'
        lines.clear()
        lines["pos_away"],    = ax.plot([], [], linestyle="None", marker="o", markersize=3, color="blue",   label="+ away (0→)")
        lines["pos_towards"], = ax.plot([], [], linestyle="None", marker="o", markersize=3, color="orange", label="+ towards (→0)")
        lines["neg_away"],    = ax.plot([], [], linestyle="None", marker="x", markersize=3, color="blue",   label="− away (0→)")
        lines["neg_towards"], = ax.plot([], [], linestyle="None", marker="x", markersize=3, color="orange", label="− towards (→0)")

        ax.legend(loc="best")
        canvas.draw_idle()

    _make_plot_fresh("Find JJ (ready)")

    # UI helpers (thread-safe)
    def ui_set_status(text):
        if ui_alive():
            win.after(0, lambda: status.config(text=text))

    def ui_reset_plot(base_title):
        if ui_alive():
            win.after(0, lambda: _make_plot_fresh(base_title))

    def ui_plot_update(which_key, xs, ys):
        """
        which_key in {'pos_away','pos_towards','neg_away','neg_towards'}
        """
        if ui_alive():
            def _upd():
                ln = lines.get(which_key)
                if ln is None:
                    return
                ln.set_data(xs, ys)
                ax.relim()
                ax.autoscale_view()
                canvas.draw_idle()
            win.after(0, _upd)

    def ui_annotate_x(I_A, label, color):
        if ui_alive():
            def _upd():
                ax.axvline(I_A, linestyle='--', color=color, linewidth=1.0, alpha=0.85)
                ytop = ax.get_ylim()[1]
                ax.text(I_A, ytop*0.85, label, rotation=90, va='top', ha='right', color=color, fontsize=9)
                canvas.draw_idle()
            win.after(0, _upd)

    def ui_set_buttons(start_enabled, stop_enabled):
        if ui_alive():
            win.after(0, lambda: (start_btn.config(state=(tk.NORMAL if start_enabled else tk.DISABLED)),
                                  stop_btn.config(state=(tk.NORMAL if stop_enabled else tk.DISABLED))))

    # ----- device worker -----
    def _measure_one_device(dev, dev_name, base_name, params):
        # phase 1: soft search
        results_search = []  # (ts,T,I,V,branch,phase,dir)
        markers = {"Ic_plus": None, "Ir_plus": None, "Ic_minus": None, "Ir_minus": None}

        _conf_smu(dev, nplc=params["nplc"], vlimit_V=params["vlimit"])
        T0, dT_max = params["T0"], params["dTmax"]
        settle_s = params["settle_s"]; Vth = params["Vth"]; Vrt = params["Vrt"]
        grow = params["grow"]; I0 = params["I0"]; Imax = params["Imax"]; min_step = params["min_step"]
        do_neg = params["neg_branch"] == 1

        # plot buffers
        xs_pos_away, ys_pos_away = [], []
        xs_pos_tow,  ys_pos_tow  = [], []
        xs_neg_away, ys_neg_away = [], []
        xs_neg_tow,  ys_neg_tow  = [], []

        def _guard_T():
            return (_tempK() <= T0 + dT_max)

        # + branch up (find Ic+): away from 0
        for I in _soft_ramp_sequence(I0, Imax, grow=grow, min_step_uA=min_step):
            if stop_event.is_set() or not ui_alive(): break
            if not _guard_T():
                ui_set_status(f"{dev_name}: ΔT exceeded. Stopping.")
                break
            _apply_I(dev, +I)
            if settle_s > 0: time.sleep(settle_s)
            i, v = _meas_iv(dev)
            ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
            results_search.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", '+', 'ramp_up', 'away'))
            xs_pos_away.append(i); ys_pos_away.append(v); ui_plot_update("pos_away", xs_pos_away[:], ys_pos_away[:])
            if abs(v) >= Vth:
                markers["Ic_plus"] = i
                ui_annotate_x(i, "Ic_plus", "#cc0000")
                break

        # + branch down (find Ir+): towards 0
        if markers["Ic_plus"] is not None and not stop_event.is_set() and ui_alive():
            for I in reversed(_soft_ramp_sequence(I0, abs(markers["Ic_plus"])*1e6, grow=grow, min_step_uA=min_step)):
                if stop_event.is_set() or not ui_alive(): break
                if not _guard_T(): break
                _apply_I(dev, +I)
                if settle_s > 0: time.sleep(settle_s)
                i, v = _meas_iv(dev)
                ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
                results_search.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", '+', 'ramp_down', 'towards'))
                xs_pos_tow.append(i); ys_pos_tow.append(v); ui_plot_update("pos_towards", xs_pos_tow[:], ys_pos_tow[:])
                if abs(v) <= Vrt:
                    markers["Ir_plus"] = i
                    ui_annotate_x(i, "Ir_plus", "#008000")
                    break

        # − branch up (find Ic−): away from 0
        if do_neg and not stop_event.is_set() and ui_alive():
            for I in _soft_ramp_sequence(I0, Imax, grow=grow, min_step_uA=min_step):
                if stop_event.is_set() or not ui_alive(): break
                if not _guard_T(): break
                _apply_I(dev, -I)
                if settle_s > 0: time.sleep(settle_s)
                i, v = _meas_iv(dev)
                ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
                results_search.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", '-', 'ramp_up', 'away'))
                xs_neg_away.append(i); ys_neg_away.append(v); ui_plot_update("neg_away", xs_neg_away[:], ys_neg_away[:])
                if abs(v) >= Vth:
                    markers["Ic_minus"] = i
                    ui_annotate_x(i, "Ic_minus", "#cc6600")
                    break

            # − branch down (find Ir−): towards 0
            if markers["Ic_minus"] is not None and not stop_event.is_set() and ui_alive():
                for I in reversed(_soft_ramp_sequence(I0, abs(markers["Ic_minus"])*1e6, grow=grow, min_step_uA=min_step)):
                    if stop_event.is_set() or not ui_alive(): break
                    if not _guard_T(): break
                    _apply_I(dev, -I)
                    if settle_s > 0: time.sleep(settle_s)
                    i, v = _meas_iv(dev)
                    ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
                    results_search.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", '-', 'ramp_down', 'towards'))
                    xs_neg_tow.append(i); ys_neg_tow.append(v); ui_plot_update("neg_towards", xs_neg_tow[:], ys_neg_tow[:])
                    if abs(v) <= Vrt:
                        markers["Ir_minus"] = i
                        ui_annotate_x(i, "Ir_minus", "#006699")
                        break

        # Save search (CSV + TXT)
        _dual_save_csv_and_txt(
            base_name, f"_{dev_name}_JJfind_search",
            header_cols=("timestamp", "T_K", "I_A", "V_V", "branch", "phase", "direction"),
            rows=results_search
        )

        # phase 2: pretty sweep(s) using Ic results + multi cycles
        results_sweep = []  # (ts,T,I,V,branch,phase,dir,cycle)

        def _sweep_branch(sign, Ic_val):
            """sign = +1 or -1; Ic_val in A (positive magnitude)."""
            if Ic_val is None or Ic_val <= 0:
                return

            pts_total = params["pretty_pts_total"]
            dense     = params["pretty_dense_factor"]
            over      = params["pretty_over_factor"]
            cycles    = params["pretty_cycles"]
            jitter    = params["pretty_jitter_frac"]

            base_up, base_down = _pretty_sweep_points(Ic_val, pts_total=pts_total, dense_factor=dense, over_factor=over)
            label_sign = '+' if sign > 0 else '-'

            for cyc in range(1, cycles + 1):
                # generate slightly different points each cycle (jitter)
                up   = _apply_jitter(base_up,   jitter_frac=jitter, keep_ends=True)
                down = _apply_jitter(base_down, jitter_frac=jitter, keep_ends=True)

                # Up sweep: away from 0
                for Iabs in up:
                    if stop_event.is_set() or not ui_alive(): break
                    if not _guard_T(): break
                    _apply_I(dev, sign * Iabs)
                    if settle_s > 0: time.sleep(settle_s)
                    i, v = _meas_iv(dev)
                    ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
                    results_sweep.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", label_sign, 'pretty_up', 'away', str(cyc)))

                    if sign > 0:
                        xs_pos_away.append(i); ys_pos_away.append(v); ui_plot_update("pos_away", xs_pos_away[:], ys_pos_away[:])
                    else:
                        xs_neg_away.append(i); ys_neg_away.append(v); ui_plot_update("neg_away", xs_neg_away[:], ys_neg_away[:])

                # Down sweep: towards 0
                for Iabs in down:
                    if stop_event.is_set() or not ui_alive(): break
                    if not _guard_T(): break
                    _apply_I(dev, sign * Iabs)
                    if settle_s > 0: time.sleep(settle_s)
                    i, v = _meas_iv(dev)
                    ts, Tk = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _tempK()
                    results_sweep.append((ts, f"{Tk:.6f}", f"{i:.12e}", f"{v:.12e}", label_sign, 'pretty_down', 'towards', str(cyc)))

                    if sign > 0:
                        xs_pos_tow.append(i); ys_pos_tow.append(v); ui_plot_update("pos_towards", xs_pos_tow[:], ys_pos_tow[:])
                    else:
                        xs_neg_tow.append(i); ys_neg_tow.append(v); ui_plot_update("neg_towards", xs_neg_tow[:], ys_neg_tow[:])

                # ensure return to exact zero between cycles (optional but matches your request)
                if stop_event.is_set() or not ui_alive(): break
                _apply_I(dev, 0.0)
                time.sleep(max(0.0, settle_s))

        # Do pretty sweeps based on found Ic
        if markers["Ic_plus"] is not None:
            _sweep_branch(+1, abs(markers["Ic_plus"]))
        if do_neg and (markers["Ic_minus"] is not None):
            _sweep_branch(-1, abs(markers["Ic_minus"]))

        # back to zero
        _apply_I(dev, 0.0)
        time.sleep(max(0.0, settle_s))
        _output_off(dev)

        # Save pretty sweep (CSV + TXT)
        _dual_save_csv_and_txt(
            base_name, f"_{dev_name}_JJfind_sweep",
            header_cols=("timestamp", "T_K", "I_A", "V_V", "branch", "phase", "direction", "cycle"),
            rows=results_sweep
        )

        return markers

    # ----- main worker -----
    def _worker():
        try:
            ui_set_buttons(False, True)
            ui_set_status("Configuring…")
            use_smua, use_smub = bool(use1.get()), bool(use2.get())
            if not (use_smua or use_smub):
                ui_set_status("No SMU selected.")
                return

            # Read ALL Tk variables/entries once (thread-safe)
            params = dict(
                Vth = float(e_vth.get()) * 1e-6,
                Vrt = float(e_vrt.get()) * 1e-6,
                dTmax = float(e_dt.get()),
                I0 = float(e_i0.get()),
                Imax = float(e_imax.get()),
                grow = float(e_grow.get()),
                min_step = float(e_min.get()),
                nplc = float(e_nplc.get()),
                vlimit = float(e_vlim.get()),
                settle_s = float(e_settle.get()) * 1e-3,
                neg_branch = int(neg_branch.get()),
                T0 = _tempK(),
                pretty_pts_total = max(60, int(float(e_pts.get()))),
                pretty_dense_factor = float(e_dense.get()),
                pretty_over_factor  = float(e_over.get()),
                pretty_cycles = max(1, int(float(e_cycles.get()))),
                pretty_jitter_frac = max(0.0, float(e_jitter.get())),
            )

            # ask filename on UI thread
            base_box = {'val': None}
            def ask_name():
                if ui_alive():
                    base_box['val'] = simpledialog.askstring(
                        "Save name",
                        "Enter base filename (no extension):",
                        parent=win
                    )
            if ui_alive(): win.after(0, ask_name)
            while base_box['val'] is None and ui_alive() and not stop_event.is_set():
                time.sleep(0.05)
            if not ui_alive() or stop_event.is_set():
                return

            base = base_box['val']
            if not base:
                ui_set_status("No filename given — aborted.")
                return

            base = _dated(base)

            # IMPORTANT: reset plot on each Start
            ui_reset_plot(f"Find JJ — {base}")

            # clear error queues (helps with queue full)
            for dev in (k, k2):
                try: dev.errorqueue.clear()
                except Exception: pass

            markers_all = {}
            if use_smua and not stop_event.is_set() and ui_alive():
                ui_set_status("SMU1: soft search + pretty sweep…")
                markers_all["SMU1"] = _measure_one_device(k, "SMU1", base, params)
            if use_smub and not stop_event.is_set() and ui_alive():
                ui_set_status("SMU2: soft search + pretty sweep…")
                markers_all["SMU2"] = _measure_one_device(k2, "SMU2", base, params)

            # subtitle summary
            def fmt_markers(m):
                def f(x): return f"{x:.6e} A" if x is not None else "—"
                return (f"Ic+={f(m.get('Ic_plus'))}, Ir+={f(m.get('Ir_plus'))}, "
                        f"Ic−={f(m.get('Ic_minus'))}, Ir−={f(m.get('Ir_minus'))}")

            subtitle = "  |  ".join([f"{name}: {fmt_markers(m)}" for name, m in markers_all.items()])

            if subtitle and ui_alive():
                def upd():
                    ax.set_title(f"Find JJ — {base}\n{subtitle}", fontsize=10)
                    canvas.draw_idle()
                win.after(0, upd)

            ui_set_status("Done. Saved CSV + TXT.")
        except Exception as ex:
            if ui_alive():
                win.after(0, lambda: messagebox.showerror("Find JJ", f"Error: {ex}"))
        finally:
            for dev in (k, k2):
                _output_off(dev)
            ui_set_buttons(True, False)

    def _stop():
        stop_event.set()
        ui_set_status("Stopping…")

    start_btn.config(command=lambda: threading.Thread(target=_worker, daemon=True).start())
    stop_btn.config(command=_stop)

    return win



#%% NEW IV MODE — unified UI (Current µA / Voltage mV), fast & cyclic, threaded, scatter plot


# ---------- helpers: paths & saving ----------
def _iv2_get_primary_path():
    try:
        return SAVE_PATH  # from your environment
    except NameError:
        return os.getcwd()

def _iv2_get_backup_path():
    try:
        return SAVE_PATH_2
    except NameError:
        return None

def _iv2_dated(name_base: str):
    return f"{name_base}_{datetime.datetime.now().strftime('%d%m%Y')}"

def _iv2_dual_save_csv(base, suffix, header, rows):
    """
    rows: list of tuples/iterables; header: string
    tries safe_file if available, otherwise writes CSV to SAVE_PATH (+ optional SAVE_PATH_2)
    """
    arr = np.array(rows, dtype=object)
    try:
        # prefer your own saver if available
        safe_file(arr, base + suffix)
        # also write backup if path exists:
        bkp = _iv2_get_backup_path()
        if bkp:
            os.makedirs(bkp, exist_ok=True)
            np.savetxt(os.path.join(bkp, base + suffix + ".csv"), arr, fmt='%s', delimiter=',', header=header, comments='')
        return
    except Exception:
        pass
    # fallback: write to SAVE_PATH (+ backup if present)
    p1 = _iv2_get_primary_path()
    os.makedirs(p1, exist_ok=True)
    f1 = os.path.join(p1, base + suffix + ".csv")
    np.savetxt(f1, arr, fmt='%s', delimiter=',', header=header, comments='')
    bkp = _iv2_get_backup_path()
    if bkp:
        try:
            os.makedirs(bkp, exist_ok=True)
            f2 = os.path.join(bkp, base + suffix + ".csv")
            np.savetxt(f2, arr, fmt='%s', delimiter=',', header=header, comments='')
        except Exception:
            pass

# ---------- helpers: instrument control ----------
def _iv2_set_sense_from_var():
    try:
        fourpt = bool(local_remote_var.get())
    except Exception:
        fourpt = True
    try:
        k.smua.sense = k.smua.SENSE_REMOTE if fourpt else k.smua.SENSE_LOCAL
    except Exception:
        pass

def _iv2_set_nplc(nplc):
    try:
        k.smua.measure.nplc = float(nplc)
    except Exception:
        pass

def _iv2_set_autozero(on: bool):
    """True => Autozero on, False => off (faster)."""
    try:
        k.smua.measure.autozero = k.smua.AUTOZERO_ON if on else k.smua.AUTOZERO_OFF
    except Exception:
        # some APIs use smua.autozero
        try:
            k.smua.autozero = k.smua.AUTOZERO_ON if on else k.smua.AUTOZERO_OFF
        except Exception:
            pass

def _iv2_source_mode_current():
    """Configure to source current, measure voltage."""
    try:
        k.smua.source.output = 1
        k.smua.source.func = 2  # current
    except Exception:
        pass

def _iv2_source_mode_voltage():
    """Configure to source voltage, measure current."""
    try:
        k.smua.source.output = 1
        k.smua.source.func = 1  # voltage
    except Exception:
        pass

def _iv2_apply_current_A(i_A):
    """Best-effort set current (A)."""
    try:
        k.apply_current(k.smua, float(i_A))
        return
    except Exception:
        pass
    try:
        k.smua.source.leveli = float(i_A)
    except Exception:
        pass

def _iv2_apply_voltage_V(v_V):
    """Best-effort set voltage (V)."""
    try:
        k.apply_voltage(k.smua, float(v_V))
        return
    except Exception:
        pass
    try:
        k.smua.source.levelv = float(v_V)
    except Exception:
        pass

def _iv2_measure_V():
    try:
        return float(k.smua.measure.v())
    except Exception:
        return float('nan')

def _iv2_measure_I():
    try:
        return float(k.smua.measure.i())
    except Exception:
        return float('nan')

def _iv2_output_off():
    try:
        k.smua.source.output = 0
    except Exception:
        pass

def _iv2_tempK():
    try:
        return float(client.query('T_sample.kelvin') or 300.0)
    except Exception:
        return 300.0

# ---------- helpers: point lists ----------
def _iv2_center_out_levels(max_abs, steps_per_half):
    """
    Build 0, +Δ, −Δ, +2Δ, −2Δ, ..., up to ±max_abs
    where Δ = max_abs / steps_per_half (linear spacing).
    """
    max_abs = float(abs(max_abs))
    n = max(1, int(steps_per_half))
    step = max_abs / n
    mags = [j * step for j in range(1, n + 1)]
    seq = [0.0]
    for m in mags:
        seq.append(+m); seq.append(-m)
    # ensure exact endpoints
    if abs(mags[-1] - max_abs) > 1e-15:
        seq.extend([+max_abs, -max_abs])
    # dedupe keeping order
    out, seen = [], set()
    for v in seq:
        key = round(v, 15)
        if key not in seen:
            out.append(v); seen.add(key)
    return out

def _iv2_make_setpoints(mode, max_abs_user, steps_per_half, cyclic_back_to_zero=True):
    """
    mode: 'I' (µA) or 'V' (mV). max_abs_user given in those user units.
    returns list of source levels in SI units (A for I-mode, V for V-mode).
    """
    if mode == 'I':
        max_abs_SI = float(max_abs_user) * 1e-6  # µA -> A
    else:
        max_abs_SI = float(max_abs_user) * 1e-3  # mV -> V
    co = _iv2_center_out_levels(max_abs_SI, steps_per_half)
    if cyclic_back_to_zero and (len(co) == 0 or abs(co[-1]) > 1e-18):
        co = co + [0.0]
    return co

# ---------- GUI ----------
def IV_Sweeps_NewMode():
    """
    Unified IV sweeps:
      - Toggle: Current (µA) vs Voltage (mV)
      - Cyclic order (0 → +max → −max → 0)
      - Fast: small NPLC, tiny settle, optional Autozero OFF
      - Live scatter plot, threaded, Stop button
    """
    # main window
    win = tk.Toplevel(root)
    win.title("IV Sweeps — New Mode (fast & cyclic)")
    win.geometry("+120+80")

    # --- Controls frame (nice layout) ---
    frm = ttk.LabelFrame(win, text="Settings")
    frm.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    win.grid_rowconfigure(1, weight=1)
    win.grid_columnconfigure(1, weight=1)

    # Mode toggle
    mode_var = tk.StringVar(value='I')  # 'I' current (µA), 'V' voltage (mV)
    ttk.Label(frm, text="Sweep mode").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    ttk.Radiobutton(frm, text="Current (µA)", variable=mode_var, value='I').grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(frm, text="Voltage (mV)", variable=mode_var, value='V').grid(row=0, column=2, sticky="w")

    # Max & steps
    ttk.Label(frm, text="Max amplitude").grid(row=1, column=0, sticky="e", padx=4, pady=4)
    max_entry = ttk.Entry(frm, width=10); max_entry.insert(0, "500.0")  # 500 µA or 500 mV
    max_entry.grid(row=1, column=1, sticky="w")
    unit_lbl = ttk.Label(frm, text="µA (in I-mode) / mV (in V-mode)")
    unit_lbl.grid(row=1, column=2, sticky="w")

    ttk.Label(frm, text="Steps per half").grid(row=2, column=0, sticky="e", padx=4, pady=4)
    steps_entry = ttk.Entry(frm, width=10); steps_entry.insert(0, "50")
    steps_entry.grid(row=2, column=1, sticky="w")

    # Cyclic & return to zero
    cyclic_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text="Cyclic (0 → +max → −max → 0)", variable=cyclic_var).grid(row=3, column=1, columnspan=2, sticky="w")

    # Speed settings
    ttk.Label(frm, text="NPLC").grid(row=4, column=0, sticky="e", padx=4, pady=4)
    nplc_entry = ttk.Entry(frm, width=10); nplc_entry.insert(0, "0.02")
    nplc_entry.grid(row=4, column=1, sticky="w")
    ttk.Label(frm, text="Settle (ms)").grid(row=4, column=2, sticky="e", padx=4, pady=4)
    settle_entry = ttk.Entry(frm, width=10); settle_entry.insert(0, "2.0")
    settle_entry.grid(row=4, column=3, sticky="w")

    autozero_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm, text="Autozero ON (slower)", variable=autozero_var).grid(row=5, column=1, columnspan=2, sticky="w")

    # Limits (optional, safe defaults)
    ttk.Label(frm, text="V-limit (A→V)").grid(row=6, column=0, sticky="e", padx=4, pady=4)
    vlim_entry = ttk.Entry(frm, width=10); vlim_entry.insert(0, "0.02")  # 20 mV limit for current mode
    vlim_entry.grid(row=6, column=1, sticky="w")

    ttk.Label(frm, text="I-limit (V→A)").grid(row=6, column=2, sticky="e", padx=4, pady=4)
    ilim_entry = ttk.Entry(frm, width=10); ilim_entry.insert(0, "0.002")  # 2 mA limit for voltage mode
    ilim_entry.grid(row=6, column=3, sticky="w")

    # Buttons
    btns = ttk.Frame(win)
    btns.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
    start_btn = ttk.Button(btns, text="Start")
    stop_btn  = ttk.Button(btns, text="Stop", state=tk.DISABLED)
    start_btn.pack(side=tk.RIGHT, padx=5)
    stop_btn.pack(side=tk.RIGHT, padx=5)

    # Progress / status
    prog = ttk.Progressbar(win, mode="determinate", maximum=100)
    prog.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
    status = ttk.Label(win, text="Idle.")
    status.grid(row=4, column=0, sticky="w", padx=8, pady=2)

    # Plot (scatter)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    fig = Figure(figsize=(6.2, 4.0), dpi=100)
    ax  = fig.add_subplot(111)
    ax.set_xlabel("Source (A in I-mode / V in V-mode)")
    ax.set_ylabel("Measure (V in I-mode / A in V-mode)")
    sc_line, = ax.plot([], [], linestyle='None', marker='o', markersize=3)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw(); canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

    # Thread state
    stop_event = threading.Event()
    data_pts = []  # (timestamp, T[K], src_SI, meas_SI, mode_char)

    def _worker():
        try:
            start_btn.config(state=tk.DISABLED); stop_btn.config(state=tk.NORMAL)
            status.config(text="Configuring…")
            _iv2_set_sense_from_var()
            mode = mode_var.get()   # 'I' or 'V'
            max_user = float(max_entry.get())
            steps_half = int(steps_entry.get())
            nplc = float(nplc_entry.get())
            settle_s = max(0.0, float(settle_entry.get()) * 1e-3)
            _iv2_set_nplc(nplc)
            _iv2_set_autozero(autozero_var.get())

            # set limits
            try:
                if mode == 'I':
                    # current source ⇒ voltage limit
                    vlim = float(vlim_entry.get())
                    k.smua.source.limitv = vlim
                else:
                    ilim = float(ilim_entry.get())
                    k.smua.source.limiti = ilim
            except Exception:
                pass

            # set source mode
            if mode == 'I':
                _iv2_source_mode_current()
            else:
                _iv2_source_mode_voltage()

            levels = _iv2_make_setpoints(mode, max_user, steps_half, cyclic_back_to_zero=cyclic_var.get())
            total = len(levels)
            prog['maximum'] = max(1, total)

            xs, ys = [], []
            t0K = _iv2_tempK()

            for i, L in enumerate(levels, 1):
                if stop_event.is_set():
                    break
                # apply setpoint (SI)
                if mode == 'I':
                    _iv2_apply_current_A(L)
                else:
                    _iv2_apply_voltage_V(L)

                if settle_s > 0:
                    time.sleep(settle_s)

                # measure complement
                if mode == 'I':
                    meas = _iv2_measure_V()
                    xval, yval = L, meas
                else:
                    meas = _iv2_measure_I()
                    xval, yval = L, meas

                xs.append(xval); ys.append(yval)
                # live scatter
                sc_line.set_data(xs, ys)
                ax.relim(); ax.autoscale_view()
                canvas.draw_idle()

                # log row
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                Tk = _iv2_tempK()
                data_pts.append((ts, f"{Tk:.6f}", f"{xval:.12e}", f"{yval:.12e}", mode))
                # progress
                prog['value'] = i
                if (i % 8) == 0 or i == 1 or i == total:
                    status.config(text=f"{i}/{total}  (T~{Tk:.3f} K)")
                    try: win.update_idletasks()
                    except tk.TclError: pass

            status.config(text="Saving…")
            # ask filename
            base = simpledialog.askstring("Save as", "Enter base filename (no extension):", parent=win)
            if base:
                base = _iv2_dated(base)
                header = "timestamp,T[K],source_SI,measure_SI,mode"
                _iv2_dual_save_csv(base, "_IV_sweep", header, data_pts)
                status.config(text=f"Saved as {base}_IV_sweep.csv")
            else:
                status.config(text="Not saved (no filename).")

        except Exception as ex:
            try: messagebox.showerror("IV Sweeps — New Mode", f"Error: {ex}")
            except Exception: pass
        finally:
            _iv2_output_off()
            stop_btn.config(state=tk.DISABLED); start_btn.config(state=tk.NORMAL)

    def on_start():
        stop_event.clear()
        threading.Thread(target=_worker, daemon=True).start()

    def on_stop():
        stop_event.set()
        status.config(text="Stopping… (finishing current point)")
        # best effort: turn output off soon
        try: _iv2_output_off()
        except Exception:
            pass

    start_btn.config(command=on_start)
    stop_btn.config(command=on_stop)

    # make window resizable & nice
    for r in range(0, 5):
        win.grid_rowconfigure(r, weight=(1 if r == 1 else 0))
    win.grid_columnconfigure(0, weight=1)

    # initial status
    status.config(text="Ready. Choose mode, set Max & Steps, then Start.")

#%% Cell 13: Connect
# Setup:
class _InstrumentAlias:
    def __init__(self, inst, ch_attr="smub"):
        self._inst = inst
        # mappe "smua" auf den gewünschten Kanal des echten Instruments
        self.smua = getattr(inst, ch_attr)

    def __getattr__(self, name):
        # alles andere (Methoden wie apply_current, sweeps, etc.)
        # an das echte Instrument weiterreichen
        return getattr(self._inst, name)

# benutze den Proxy
client, host, temperature_control, sample_control, adr_control = ConnectKiutra()

# Nur noch EIN Instrument (Keithley 2636 an ASRL5)
# → nutzt die neue ConnectKeithley_ASRL5_TSP Funktion aus Connection_Codes.py
k = ConnectKeithley_ASRL5_TSP()

# Alias: k2.smua zeigt auf k.smub  (dadurch funktioniert dein alter Code weiter)
class _ChannelAlias:
    def __init__(self, ch):
        self.smua = ch


k2 = _InstrumentAlias(k, "smub")

print(k)
print("2636A Kanäle:", k.smua, k.smub)

# Autozero initial setzen
k.smua.measure.autozero = k.smua.AUTOZERO_ONCE
k2.smua.measure.autozero = k2.smua.AUTOZERO_ONCE

def AutozeroSMU():
    k.smua.measure.autozero = k.smua.AUTOZERO_ONCE
    k2.smua.measure.autozero = k2.smua.AUTOZERO_ONCE

def AutozeroSMU1():
    k.smua.measure.autozero = k.smua.AUTOZERO_AUTO
    k2.smua.measure.autozero = k2.smua.AUTOZERO_AUTO

# Outputs sicher aus
k.smua.source.output = 0
k2.smua.source.output = 0
k.smua.source.autorangei = 1 
k2.smua.source.autorangei = 1 

#%% Cell 14: Help

def open_detail_window(detail_text):
    detail_window = tk.Toplevel(root)
    detail_window.title("Details")
    
    detail_label = tk.Label(detail_window, text=detail_text, wraplength=300)  # Adjust the wrap length as needed
    detail_label.pack()
def add_button_with_details(root, button_text, button_command, details_text):
    button_frame = tk.Frame(root)
    button_frame.pack()

    # Create the main button
    main_button = tk.Button(button_frame, text=button_text, command=lambda: checkformeasurement(button_command))
    main_button.grid(row=0, column=0)

    # Create a "Details" button next to it
    details_button = tk.Button(button_frame, text="Details", command=lambda text=details_text: open_detail_window(text))
    details_button.grid(row=0, column=1)
def add_button_with_details_and_load(root, button_text, button_command, details_text,loadfunction):
    button_frame = tk.Frame(root)
    button_frame.pack()
    load_button = tk.Button(button_frame, text="Load", command=lambda: checkformeasurement(loadfunction))
    load_button.grid(row=0, column=0)
    # Create the main button
    main_button = tk.Button(button_frame, text=button_text, command=lambda: checkformeasurement(button_command))
    main_button.grid(row=0, column=1)

    # Create a "Details" button next to it
    details_button = tk.Button(button_frame, text="Details", command=lambda text=details_text: open_detail_window(text))
    details_button.grid(row=0, column=2)
#%% Cell 15: MAIN
#Gui For measurements 


root = tk.Tk()
use_smu2_var = tk.IntVar()
local_remote_var = tk.IntVar()
measurement_started = tk.IntVar()
root.title("Kiutra Measurement GUI")

# Create frames to group buttons
iv_frame = tk.LabelFrame(root, text="IV Characteristics")
iv_frame.pack(pady=10)

tc_frame = tk.LabelFrame(root, text="Tc Characteristics")
tc_frame.pack(pady=10)

aux_frame = tk.LabelFrame(root, text="Auxiliary")
aux_frame.pack(pady=10)

# Define a list of functions and their corresponding detail explanations
add_button_with_details_and_load(
    iv_frame,
    "IV Sweeps (Current/Voltage)",
    IV_Sweeps_NewMode,
    "Unified IV sweeps with mode toggle:\n"
    "- Current mode: µA source, measure V\n"
    "- Voltage mode: mV source, measure I\n"
    "Cyclic sweep 0→+max→−max→0, fast NPLC and short settle.\n"
    "Connect SMU1 HI/LO to your device. Safe limits are applied.",
    load_cb := (lambda: None)  # or keep your existing loader if you still want it
)
add_button_with_details_and_load(
    iv_frame,
    "Find JJ",
    Find_JJ,
    "Soft Ic finder with retrapping (optional). Uses SMU1/SMU2 selectable.\n"
    "Starts from tiny currents and grows smoothly. Stops if ΔT exceeds 0.1 K.\n"
    "Saves CSV with timestamp and temperature.",
    load_cb := (lambda: None)
)

add_button_with_details_and_load(iv_frame, "Single IV Curve Current", create_sweep_current, "Start: Start Current in muA, End: End Current in muA, Steps: Amount of Steps (Integer). Checkbox Unchecked: Connect SMU1 HI and SMU1 LO to your device. Checkbox Checked: Connect SMU1 HI and SMU1 LO to your device as outer electrodes and SMU1 Sense Hi and SMU1 Sense Lo as inner electrodes. Range: -0.1A;0.1A",load_sweep_current)
add_button_with_details_and_load(iv_frame, "Single IV Curve Voltage", create_sweep, "Start: Start Voltage, End: End Voltage, Steps: Amount of Steps (Integer). Connect SMU1 HI and SMU1 LO to your device. Range: -20V;20V",load_sweep)
add_button_with_details(tc_frame, "Voltage sweep at several temperatures", IV_Temp, "Connect SMU1 HI and SMU1 LO to your device. Range: -20;20V, 0.2K;290K")
add_button_with_details(tc_frame, "Current sweep at several temperatures", create_current_temp_sweep, "Current sweep in µA at multiple temperatures. Provide either a comma-separated temperature list or start/end with intermediate steps. Thermalization time applies after reaching the temperature. Uses temperature control unless target < 0.5 K.")
add_button_with_details(tc_frame, "2 Point Tc Measurement", create_tc_measurement, "Connect SMU1 HI and SMU1 LO to your device. Range: 1muA;100mA, 0.2K;290K, high power might reduce Temperature Range. Recommended: 100muA")
add_button_with_details(tc_frame, "4-point Tc Measurement 2 SMU Classic", create_4ptc_measurement, "Connect SMU1 HI and SMU1 LO to your device as outer electrodes and SMU2 HI and SMU2 Lo as inner electrodes. Range 1muA;100mA, 0.2K;290K, high power might reduce Temperature Range. Recommended: 100muA")
add_button_with_details(tc_frame, "4-point Tc Measurement 1 SMU Sense", create_4pt_measurement_Sense, "Connect SMU1 HI and SMU1 LO to your device as outer electrodes and SMU1 Sense Hi and SMU1 Sense Lo as inner electrodes. The Hi and Lo should be in Order SMU1 Hi SMU1 Sense Hi SMU1 Sense LO SMU1 LO Range: 1muA;100mA, 0.2K;290K, high power might reduce Temperature Range. Recommended: 100muA")
add_button_with_details(tc_frame, "2 Point Parallel Tc Measurement", create_parallel_tc_measurement, "Connect SMU1 HI and SMU1 LO to your device 1 and SMU2 Hi and SMU2 Lo to your device 2. Range: 1muA;100mA, 0.2K;290K, high power might reduce Temperature Range. Recommended: 100muA")
add_button_with_details(tc_frame, "Parallel 4-point Tc Measurement 2 SMU Sense", create_parallel_4pt_measurement_Sense, "Connect SMU1 HI and SMU1 LO to your device 1 as outer electrodes and SMU1 Sense HI and SMU1 Sense LO as inner electrodes. The HI and LO should be in Order SMU1 HI SMU1 Sense Hi SMU1 Sense LO SMU1 LO. Connect SMU2 HI and SMU2 LO to your device 2 as outer electrodes and SMU2 Sense HI and SMU2 Sense Lo as inner electrodes. The HI and LO should be in Order SMU2 HI SMU2 Sense HI SMU2 Sense LO SMU2 LO Range: 1muA;100mA, 0.2K;290K, high power might reduce Temperature Range. Recommended: 100muA")
add_button_with_details(tc_frame, "SMU1 4PT Current Sweep, SMU2 var Current, Magnetic Measurement,Log", Fraunhofer_Pattern, "Connect SMU1 HI and SMU1 LO to your device 1 as outer electrodes and SMU1 Sense HI and SMU1 Sense LO as inner electrodes. The HI and LO should be in Order SMU1 HI SMU1 Sense Hi SMU1 Sense LO SMU1 LO. Connect SMU2 HI and SMU2 LO to your coil.")
add_button_with_details(tc_frame, "SMU1 4PT Current Sweep, SMU2 var Current, Magnetic Measurement,Rough", RoughScan, "Connect SMU1 HI and SMU1 LO to your device 1 as outer electrodes and SMU1 Sense HI and SMU1 Sense LO as inner electrodes. The HI and LO should be in Order SMU1 HI SMU1 Sense Hi SMU1 Sense LO SMU1 LO. Connect SMU2 HI and SMU2 LO to your coil.")
add_button_with_details(tc_frame, "SMU1 4PT Current Sweep, SMU2 var Current, Magnetic Measurement,linear", MagneticMeasurementLinear, "Connect SMU1 HI and SMU1 LO to your device 1 as outer electrodes and SMU1 Sense HI and SMU1 Sense LO as inner electrodes. The HI and LO should be in Order SMU1 HI SMU1 Sense Hi SMU1 Sense LO SMU1 LO. Connect SMU2 HI and SMU2 LO to your coil.")
add_button_with_details(aux_frame, "Set Integration Time", thread_safe_update, "This button allows you to set the integration time, only applicable to IV Sweeps, default 0.2s.")
add_button_with_details(aux_frame, "Set Default Path", set_default_path, "This button sets the default path for loading IV Sweep scripts.")
add_button_with_details(aux_frame, "Set Save Path", set_save_path, "This button sets the save path for measurements.")
add_button_with_details(aux_frame, "Autozero Once", AutozeroSMU, "Autozero both SMU and disable automatic autozero.")
add_button_with_details(aux_frame, "Autozero Automatic", AutozeroSMU1, "Enable automatic autozero.")
add_button_with_details(aux_frame, "Reset Temperature Plot", whipe_data, "Resets the Temperature data list.")
add_button_with_details(aux_frame,"Save now",lambda: _save_results(_ff_state),"Speichert sofort alle aktuellen Daten (Ic_pos, Ic_neg, IV_all_search).")




temperature_label = tk.Label(root, text="Current Temperature: --- K")
temperature_label.pack()

save_path_label = tk.Label(root, text="Gui and Scripts Programmed by Benedikt Schoof")
save_path_label.pack()
update_temperature_label()
plot_windowT, figT, axT, lineT = create_plot_window_for_Temperature()
aniT = FuncAnimation(figT, update_temp_plot, frames=100000, repeat=False, interval=5000)
aniT._start()
# Update temperature label and plot here

root.mainloop()
#%% Cell 16 Test
import time
import math

def _meas_iv_avg(dev, extra_reads=3, delay_s=0.05):
    """
    Einfache Mittelung von I,V über mehrere Messungen.
    dev: Keithley2600 oder k2-Alias (mit .smua)
    """
    is_, vs = [], []
    for _ in range(max(1, extra_reads)):
        try:
            i, v = dev.smua.measure.iv()   # je nach Wrapper: (i, v)
        except Exception:
            # Fallback getrennt
            i = dev.smua.measure.i()
            v = dev.smua.measure.v()
        is_.append(float(i))
        vs.append(float(v))
        time.sleep(delay_s)
    i_avg = sum(is_) / len(is_)
    v_avg = sum(vs) / len(vs)
    return i_avg, v_avg

def test_smu_basic(max_current=1e-6, extra_reads=5):
    """
    Kleiner Funktionstest für beide Kanäle deines 2636:
      - bis max_current (Standard: 1 µA)
      - gibt für SMU A (k.smua) und SMU B (k2.smua==k.smub) I, V, R aus.
    """
    # ein paar Testströme (0.25, 0.5, 0.75, 1.0) * max_current
    currents = [0.25, 0.5, 0.75, 1.0]
    currents = [c * max_current for c in currents]

    for label, dev in [("SMU1 (A-Kanal)", k), ("SMU2 (B-Kanal)", k2)]:
        if dev is None:
            continue

        print(f"\n=== Test {label} ===")
        smu = dev.smua

        # Grundkonfiguration: Current Source, Autorange an, Output erstmal aus
        try:
            smu.source.output = 0
        except Exception:
            pass

        try:
            smu.source.func = 2          # 2 = Current Source (wie in deinem Code)
        except Exception:
            pass

        try:
            smu.source.autorangei = 1    # Strom-Autorange an
        except Exception:
            pass

        try:
            smu.measure.nplc = 1.0       # moderate Integrationszeit
        except Exception:
            pass

        try:
            smu.measure.autozero = smu.AUTOZERO_ONCE
        except Exception:
            pass

        # Schleife über die Testströme
        for I_set in currents:
            try:
                smu.source.leveli = I_set
                smu.source.output = 1
                time.sleep(0.1)  # kleines Settling

                i_meas, v_meas = _meas_iv_avg(dev, extra_reads=extra_reads)
                R = float('nan')
                if abs(i_meas) > 0:
                    R = v_meas / i_meas

                print(f"I_set = {I_set: .3e} A,  "
                      f"I_meas = {i_meas: .3e} A,  "
                      f"V_meas = {v_meas: .3e} V,  "
                      f"R = {R: .3e} Ω")
            except Exception as e:
                print(f"Fehler bei I_set={I_set: .3e}: {e}")
            finally:
                try:
                    smu.source.output = 0
                except Exception:
                    pass

    print("\nTest fertig. Beide Kanäle wurden bis max "
          f"{max_current:.2e} A geprüft.")
