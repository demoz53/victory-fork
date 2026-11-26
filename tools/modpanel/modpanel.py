#!/usr/bin/env python3
"""VIctory mod panel: a small window for the game-side VICTORY_MOD commands.

It only APPENDS command lines to the mod file (VICTORY_MOD_FILE) and READS the
status file (VICTORY_MOD_STATUS) the game rewrites twice a second. It never
touches the game itself. Python 3 standard library only (tkinter).

    python modpanel.py --mod-file <path> --status-file <path> [--places <json>] [--top]

See README.md beside this file for every command.
"""

import argparse
import json
import math
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "victory-data", "mod")

DEFAULT_PLACES = [
    {"name": "Jason's house bed", "x": -2353.31, "y": -5534.35, "z": 7.0},
    {"name": "Beside the house", "x": -2357.4, "y": -5525.4, "z": 3.0},
    {"name": "The water's edge", "x": -2368.0, "y": -5530.9, "z": 3.3, "heading": 2.17},
    {"name": "The deck", "x": -2355.6, "y": -5526.3, "z": 7.2},
]
# Model names to try (none is confirmed in the game yet: the game's own
# display names are searched, so type what the game shows when you enter a car).
DEFAULT_VEHICLES = []
WEATHERS = ["sunny", "clear", "misty", "fog", "clouds", "overcast", "darkovercast", "drizzle",
            "shower", "rain", "thunder", "thunderstorm", "hurricane", "hail", "sandstorm"]
# A status file older than this means the game is not running (it is rewritten every 0.5 s).
STALE_S = 3.0


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print(f"modpanel: {path} could not be read ({e}); using the defaults", file=sys.stderr)
    return {}


def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def read_status(path):
    """The status file as a dict, or None (missing or unreadable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def is_live(status):
    if not status:
        return False
    try:
        return time.time() - int(status.get("updated_unix_ms", "0")) / 1000.0 < STALE_S
    except ValueError:
        return False


def fmt(v):
    """A number as the game's parser takes it (no exponent)."""
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


class Panel:
    def __init__(self, root, mod_file, status_file, places_file, top):
        self.root = root
        self.mod_file = mod_file
        self.status_file = status_file
        self.places_file = places_file
        self.status = None
        self.live = False
        self.last_seen = None  # (last_frame, last_command, last_result) already logged
        data = load_json(places_file)
        self.places = data.get("places") or [dict(p) for p in DEFAULT_PLACES]
        self.vehicles = data.get("vehicles") or list(DEFAULT_VEHICLES)
        if not os.path.exists(places_file):
            self.save_places()

        root.title("VIctory mod panel")
        root.minsize(560, 520)
        self.top = tk.BooleanVar(value=top)
        root.attributes("-topmost", top)

        outer = ttk.Frame(root, padding=6)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # Status.
        st = ttk.LabelFrame(outer, text="Status", padding=4)
        st.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.state_label = ttk.Label(st, text="", font=("Segoe UI", 10, "bold"))
        self.state_label.pack(anchor="w")
        self.status_text = tk.StringVar(value="")
        ttk.Label(st, textvariable=self.status_text, justify="left", font=("Consolas", 9)).pack(anchor="w")
        row = ttk.Frame(st)
        row.pack(anchor="w")
        ttk.Checkbutton(row, text="Always on top", variable=self.top, command=self.set_top).pack(side="left")
        ttk.Label(row, text=f"  mod file: {mod_file}", foreground="#666").pack(side="left")

        # Places.
        pl = ttk.LabelFrame(outer, text="Places", padding=4)
        pl.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.place_list = tk.Listbox(pl, height=7, exportselection=False)
        self.place_list.pack(fill="both", expand=True)
        self.place_list.bind("<Double-Button-1>", lambda e: self.go_place())
        b = ttk.Frame(pl)
        b.pack(fill="x")
        ttk.Button(b, text="Teleport", command=self.go_place).pack(side="left")
        ttk.Button(b, text="Save current place", command=self.save_current).pack(side="left")
        ttk.Button(b, text="Delete", command=self.delete_place).pack(side="left")
        self.fill_places()
        xyz = ttk.Frame(pl)
        xyz.pack(fill="x", pady=(4, 0))
        self.tx, self.ty, self.tz = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for label, var in (("x", self.tx), ("y", self.ty), ("z", self.tz)):
            ttk.Label(xyz, text=label).pack(side="left")
            ttk.Entry(xyz, textvariable=var, width=9).pack(side="left")
        ttk.Button(xyz, text="Go", command=self.go_xyz).pack(side="left")
        ttk.Button(xyz, text="Waypoint", command=lambda: self.send("tp waypoint")).pack(side="left")
        ttk.Label(pl, text="(leave z empty to land on the ground)", foreground="#666").pack(anchor="w")
        nudge = ttk.Frame(pl)
        nudge.pack(fill="x", pady=(4, 0))
        ttk.Button(nudge, text="Forward 5 m", command=lambda: self.send("tp rel 5 0")).pack(side="left")
        ttk.Button(nudge, text="Back 5 m", command=lambda: self.send("tp rel -5 0")).pack(side="left")
        ttk.Button(nudge, text="Up 5 m", command=lambda: self.send("tp rel 0 5")).pack(side="left")

        # World and player.
        wp = ttk.LabelFrame(outer, text="World and player", padding=4)
        wp.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        ttk.Label(wp, text="Time of day").grid(row=0, column=0, sticky="w")
        self.hour = tk.DoubleVar(value=12.0)
        self.hour_label = ttk.Label(wp, text="12:00", width=6)
        self.hour_label.grid(row=0, column=2, sticky="w")
        sc = ttk.Scale(wp, from_=0, to=23.75, variable=self.hour, command=self.show_hour, length=150)
        sc.grid(row=0, column=1, sticky="ew")
        sc.bind("<ButtonRelease-1>", lambda e: self.set_time())
        self.freeze = tk.BooleanVar(value=False)
        ttk.Checkbutton(wp, text="Freeze time", variable=self.freeze,
                        command=lambda: self.send(f"time freeze {'on' if self.freeze.get() else 'off'}")).grid(row=1, column=1, sticky="w")
        ttk.Label(wp, text="Weather").grid(row=2, column=0, sticky="w")
        self.weather = tk.StringVar(value="clear")
        ttk.Combobox(wp, textvariable=self.weather, values=WEATHERS + ["cycle"], width=14).grid(row=2, column=1, sticky="w")
        ttk.Button(wp, text="Set", command=lambda: self.send(f"weather {self.weather.get().strip()}")).grid(row=2, column=2, sticky="w")
        ttk.Separator(wp).grid(row=3, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Button(wp, text="Heal", command=lambda: self.send("heal")).grid(row=4, column=0, sticky="w")
        self.invincible = tk.BooleanVar(value=False)
        ttk.Checkbutton(wp, text="Invincible", variable=self.invincible,
                        command=lambda: self.send(f"invincible {'on' if self.invincible.get() else 'off'}")).grid(row=4, column=1, sticky="w")
        ttk.Label(wp, text="Wanted").grid(row=5, column=0, sticky="w")
        self.wanted = tk.StringVar(value="0")
        ttk.Spinbox(wp, from_=0, to=6, textvariable=self.wanted, width=4).grid(row=5, column=1, sticky="w")
        ttk.Button(wp, text="Set", command=lambda: self.send(f"wanted {self.wanted.get().strip()}")).grid(row=5, column=2, sticky="w")
        self.never = tk.BooleanVar(value=False)
        ttk.Checkbutton(wp, text="Never wanted", variable=self.never,
                        command=lambda: self.send(f"wanted never {'on' if self.never.get() else 'off'}")).grid(row=6, column=1, sticky="w")
        ttk.Separator(wp).grid(row=7, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(wp, text="Vehicle").grid(row=8, column=0, sticky="w")
        self.vehicle = tk.StringVar(value=self.vehicles[0] if self.vehicles else "")
        ttk.Combobox(wp, textvariable=self.vehicle, values=self.vehicles, width=14).grid(row=8, column=1, sticky="w")
        ttk.Button(wp, text="Spawn", command=self.spawn).grid(row=8, column=2, sticky="w")
        ttk.Button(wp, text="Repair my vehicle", command=lambda: self.send("vehicle repair")).grid(row=9, column=1, sticky="w")

        # Raw line and log.
        raw = ttk.Frame(outer)
        raw.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        raw.columnconfigure(1, weight=1)
        ttk.Label(raw, text="Command").grid(row=0, column=0)
        self.raw = tk.StringVar()
        e = ttk.Entry(raw, textvariable=self.raw)
        e.grid(row=0, column=1, sticky="ew")
        e.bind("<Return>", lambda ev: self.send_raw())
        ttk.Button(raw, text="Send", command=self.send_raw).grid(row=0, column=2)
        lg = ttk.LabelFrame(outer, text="Log", padding=2)
        lg.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(6, 0))
        outer.rowconfigure(3, weight=1)
        self.log_box = tk.Text(lg, height=9, wrap="word", font=("Consolas", 9), state="disabled")
        self.log_box.pack(fill="both", expand=True)

        self.log(f"status file: {status_file}")
        self.poll()

    # ---- sending -----------------------------------------------------------------
    def set_top(self):
        self.root.attributes("-topmost", self.top.get())

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", time.strftime("%H:%M:%S ") + text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def send(self, line):
        line = " ".join(line.split())
        if not line:
            return
        if not self.live:
            self.log(f"not sent (the game is not running, or VICTORY_MOD is off): {line}")
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.mod_file)), exist_ok=True)
            with open(self.mod_file, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        except OSError as e:
            self.log(f"could not write the mod file ({e}): {line}")
            return
        self.log(f"> {line}")

    def send_raw(self):
        self.send(self.raw.get())
        self.raw.set("")

    def show_hour(self, _=None):
        h = self.hour.get()
        hh, mm = int(h), int(round((h - int(h)) * 60 / 15) * 15) % 60
        self.hour_label.configure(text=f"{hh:02d}:{mm:02d}")
        return hh, mm

    def set_time(self):
        hh, mm = self.show_hour()
        self.send(f"time {hh} {mm}")

    def spawn(self):
        name = self.vehicle.get().strip()
        if not name:
            self.log("type a vehicle name (as the game shows it) or a 0x model hash")
            return
        self.send(f"spawn vehicle {name.replace(' ', '')}")

    # ---- places ------------------------------------------------------------------
    def save_places(self):
        try:
            save_json(self.places_file, {"places": self.places, "vehicles": self.vehicles})
        except OSError as e:
            self.log(f"could not save {self.places_file}: {e}")

    def fill_places(self):
        self.place_list.delete(0, "end")
        for p in self.places:
            self.place_list.insert("end", f"{p['name']}  ({fmt(p['x'])}, {fmt(p['y'])}, {fmt(p['z'])})")

    def selected(self):
        sel = self.place_list.curselection()
        return sel[0] if sel else None

    def go_place(self):
        i = self.selected()
        if i is None:
            self.log("pick a place first")
            return
        p = self.places[i]
        line = f"teleport {fmt(p['x'])} {fmt(p['y'])} {fmt(p['z'])}"
        if p.get("heading") is not None:
            line += f" {fmt(p['heading'])}"
        self.send(line)

    def go_xyz(self):
        try:
            x, y = float(self.tx.get()), float(self.ty.get())
            z = self.tz.get().strip()
            line = f"teleport {fmt(x)} {fmt(y)}" + (f" {fmt(float(z))}" if z else "")
        except ValueError:
            self.log("x and y must be numbers (z a number or empty)")
            return
        self.send(line)

    def save_current(self):
        s = self.status
        if not self.live or not s or s.get("player") != "1":
            self.log("no current place: the game is not running or there is no player")
            return
        name = simpledialog.askstring("Save place", "Name of this place:", parent=self.root)
        if not name:
            return
        try:
            p = {"name": name, "x": float(s["x"]), "y": float(s["y"]), "z": float(s["z"]),
                 "heading": float(s["heading"])}
        except (KeyError, ValueError):
            self.log("the status file has no position yet")
            return
        self.places.append(p)
        self.save_places()
        self.fill_places()
        self.log(f"saved {name} at {fmt(p['x'])} {fmt(p['y'])} {fmt(p['z'])}")

    def delete_place(self):
        i = self.selected()
        if i is None:
            self.log("pick a place first")
            return
        if messagebox.askyesno("Delete place", f"Delete {self.places[i]['name']}?", parent=self.root):
            del self.places[i]
            self.save_places()
            self.fill_places()

    # ---- status ------------------------------------------------------------------
    def poll(self):
        try:
            self.status = read_status(self.status_file)
            self.live = is_live(self.status)
            self.show_status()
        except Exception as e:  # the panel keeps running whatever the file holds
            self.state_label.configure(text=f"status unreadable: {e}")
        self.root.after(500, self.poll)

    def show_status(self):
        s = self.status
        if not s:
            self.state_label.configure(text="The game is not running (no status file yet).", foreground="#a00")
            self.status_text.set("")
            return
        if not self.live:
            self.state_label.configure(text="The game is not running (the status file is old).", foreground="#a00")
        elif s.get("player") != "1":
            self.state_label.configure(text=f"Running, no player yet (frame {s.get('frame', '?')}).", foreground="#a60")
        else:
            self.state_label.configure(text=f"Running, frame {s.get('frame', '?')}.", foreground="#070")
        g = s.get
        lines = []
        if g("player") == "1":
            where = "in a vehicle" if g("in_vehicle") == "1" else "on foot"
            lines.append(f"x {g('x')}  y {g('y')}  z {g('z')}  heading {g('heading_deg', '?')} deg  {where}")
            lines.append(f"health {g('health', '?')} / {g('max_health', '?')}   wanted {g('wanted', '?')}"
                         f"   invincible {g('invincible', 'off')}   never wanted {g('never_wanted', 'off')}")
        lines.append(f"clock {g('clock', '?')}{' (frozen)' if g('clock_frozen') == '1' else ''}"
                     f"   weather {g('weather', '?')}   teleport {g('teleport', '?')}")
        for k in ("spawn", "native_result"):
            if g(k):
                lines.append(f"{k} {g(k)}")
        lines.append(f"last: {g('last_command', '-')}  ->  {g('last_result', '-')}")
        self.status_text.set("\n".join(lines))
        seen = (g("last_frame"), g("last_command"), g("last_result"))
        if g("last_command") and seen != self.last_seen:
            if self.last_seen is not None or self.live:
                self.log(f"  {g('last_command')}: {g('last_result')}")
            self.last_seen = seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mod-file", default=os.environ.get("VICTORY_MOD_FILE") or os.path.join(DEFAULT_DIR, "commands.txt"))
    ap.add_argument("--status-file", default=os.environ.get("VICTORY_MOD_STATUS") or os.path.join(DEFAULT_DIR, "status.txt"))
    ap.add_argument("--places", default=os.path.join(HERE, "places.json"))
    ap.add_argument("--top", action="store_true", help="keep the window above the game's")
    a = ap.parse_args()
    root = tk.Tk()
    Panel(root, a.mod_file, a.status_file, a.places, a.top)
    root.mainloop()


if __name__ == "__main__":
    main()
