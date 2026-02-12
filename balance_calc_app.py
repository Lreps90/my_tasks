import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date, datetime
import calendar

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter


# ============================================================
# Persistence
# ============================================================
APP_NAME = "money_growth_forecaster"
SETTINGS_FILENAME = "settings.json"


def _user_data_dir() -> str:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME)
    return os.path.join(os.path.expanduser("~"), f".{APP_NAME}")


def settings_path() -> str:
    d = _user_data_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, SETTINGS_FILENAME)


def safe_load_settings() -> dict:
    path = settings_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def safe_save_settings(data: dict) -> None:
    path = settings_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def iso_to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def date_to_iso(d: date) -> str:
    return d.isoformat()


# ============================================================
# Scrollable frame (Canvas + vertical scrollbar)
# ============================================================
class ScrollableFrame(ttk.Frame):
    """
    A ttk.Frame with a vertical scrollbar and a fixed viewport.
    Put widgets into `.inner`.
    """
    def __init__(self, master, *, width: int = 520, height: int = 720, **kwargs):
        super().__init__(master, **kwargs)

        self.configure(width=width, height=height)
        self.grid_propagate(False)
        self.pack_propagate(False)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mousewheel scrolling when hovered
        self.canvas.bind("<Enter>", self._bind_wheel_local)
        self.canvas.bind("<Leave>", self._unbind_wheel_local)

        self.after_idle(self._refresh_scrollregion)

    def _refresh_scrollregion(self):
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

    def _on_inner_configure(self, _event=None):
        self._refresh_scrollregion()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_wheel_local(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)      # Windows/mac
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)  # Linux down

    def _unbind_wheel_local(self, _event=None):
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# =============================
# Parsing / validation helpers
# =============================
def parse_float_money_str(s: str, field_name: str) -> float:
    s = (s or "").strip().replace("£", "").replace(",", "")
    if not s:
        raise ValueError(f"{field_name} is required.")
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")
    if v < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return v


def parse_float_money(entry: ttk.Entry, field_name: str) -> float:
    return parse_float_money_str(entry.get(), field_name)


def parse_rate_str(s: str, field_name: str) -> float:
    """
    Accepts: 0.08, 8, 8%, 8.0%
    Returns decimal annual rate (e.g. 0.08)
    """
    s = (s or "").strip().replace(" ", "")
    if not s:
        raise ValueError(f"{field_name} is required.")

    if s.endswith("%"):
        s = s[:-1]
        try:
            v = float(s) / 100.0
        except ValueError:
            raise ValueError(f"{field_name} must be like 8 or 8% or 0.08.")
        if v < 0:
            raise ValueError(f"{field_name} cannot be negative.")
        return v

    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{field_name} must be like 8 or 8% or 0.08.")

    if v < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    if v > 1.0:
        return v / 100.0
    return v


def parse_rate(entry: ttk.Entry, field_name: str) -> float:
    return parse_rate_str(entry.get(), field_name)


def parse_int_str(s: str, field_name: str) -> int:
    s = (s or "").strip()
    if not s:
        raise ValueError(f"{field_name} is required.")
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer.")


def parse_int(entry: ttk.Entry, field_name: str) -> int:
    return parse_int_str(entry.get(), field_name)


def parse_date_iso_str(s: str, field_name: str) -> date:
    s = (s or "").strip()
    if not s:
        return date.today()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (e.g. 2026-02-10).")


def parse_date_iso(entry: ttk.Entry, field_name: str) -> date:
    return parse_date_iso_str(entry.get(), field_name)


def parse_dob_str(s: str, field_name: str) -> date:
    """
    Accepts:
      - 03 January 1990
      - 03 Jan 1990
      - 1990-01-03
    """
    s = (s or "").strip()
    if not s:
        raise ValueError(f"{field_name} is required.")

    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        pass

    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    raise ValueError(f"{field_name} must be like '03 January 1990' (or 1990-01-03).")


def parse_dob(entry: ttk.Entry, field_name: str) -> date:
    return parse_dob_str(entry.get(), field_name)


# =============================
# Date helpers
# =============================
def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)
    return date(y, m, day)


def days_in_month(d: date) -> int:
    return calendar.monthrange(d.year, d.month)[1]


def safe_date_years_later(d: date, years: int) -> date:
    y = d.year + years
    m = d.month
    last_day = calendar.monthrange(y, m)[1]
    day = min(d.day, last_day)
    return date(y, m, day)


def months_between(start_date: date, end_date: date) -> int:
    s = date(start_date.year, start_date.month, 1)
    e = date(end_date.year, end_date.month, 1)
    return (e.year - s.year) * 12 + (e.month - s.month)


def make_month_labels(total_months: int, start_date: date) -> list[str]:
    cur = date(start_date.year, start_date.month, 1)
    labels = ["Start"]
    for _ in range(total_months):
        labels.append(cur.strftime("%b %Y"))
        cur = add_months(cur, 1)
    return labels


def normalise_to_month_start(d: date) -> date:
    return date(d.year, d.month, 1)


# =============================
# Contribution schedule helpers
# =============================
def build_monthly_contribution_vector(
    *,
    start_date: date,
    total_months: int,
    base_monthly: float,
    schedule: list[tuple[date, float]],
) -> list[float]:
    if total_months <= 0:
        return []

    start_m = normalise_to_month_start(start_date)
    sched = [(normalise_to_month_start(d), float(v)) for d, v in schedule]
    sched.sort(key=lambda x: x[0])

    contribs: list[float] = []
    cur_amount = float(base_monthly)
    j = 0

    for m in range(total_months):
        month_date = add_months(start_m, m)
        while j < len(sched) and sched[j][0] <= month_date:
            cur_amount = float(sched[j][1])
            j += 1
        contribs.append(cur_amount)

    return contribs


# =============================
# Deterministic series (daily) with variable contributions
# =============================
def build_account_monthly_series_daily_variable(
    *,
    initial_balance: float,
    monthly_contributions: list[float],
    annual_rate: float,
    start_date: date,
):
    total_months = len(monthly_contributions)
    if total_months <= 0:
        raise ValueError("Duration must be at least 1 month.")
    if annual_rate < 0:
        raise ValueError("Annual rate cannot be negative.")

    daily_rate = annual_rate / 365.0
    bal = float(initial_balance)
    invested = float(initial_balance)

    cur = date(start_date.year, start_date.month, 1)

    labels = ["Start"]
    with_interest = [round(bal, 2)]
    invested_series = [round(invested, 2)]

    for m in range(total_months):
        c = float(monthly_contributions[m])
        if c < 0:
            raise ValueError("Monthly contribution cannot be negative.")

        bal += c
        invested += c

        dim = days_in_month(cur)
        for _ in range(dim):
            bal += bal * daily_rate

        labels.append(cur.strftime("%b %Y"))
        with_interest.append(round(bal, 2))
        invested_series.append(round(invested, 2))

        cur = add_months(cur, 1)

    return labels, with_interest, invested_series


def combine_series(*series: list[float]) -> list[float]:
    if not series:
        return []
    n = len(series[0])
    for s in series[1:]:
        if len(s) != n:
            raise ValueError("Series lengths do not match (internal error).")
    out = [0.0] * n
    for s in series:
        out = [round(a + b, 2) for a, b in zip(out, s)]
    return out


# =============================
# Monte Carlo engine (monthly lognormal) with variable contributions
# =============================
def monte_carlo_paths_lognormal_monthly_variable_contrib(
    *,
    initial_balance: float,
    monthly_contributions: list[float],
    annual_return: float,
    annual_vol: float,
    n_sims: int,
    seed: int | None,
) -> np.ndarray:
    months = len(monthly_contributions)
    if months <= 0:
        raise ValueError("Months must be at least 1.")
    if n_sims <= 0:
        raise ValueError("Simulations must be at least 1.")
    if annual_vol < 0:
        raise ValueError("Volatility cannot be negative.")
    if annual_return < -1.0:
        raise ValueError("Annual return looks invalid.")

    rng = np.random.default_rng(seed)

    sigma_m = annual_vol / np.sqrt(12.0)
    drift_m = (annual_return - 0.5 * (annual_vol ** 2)) / 12.0

    z = rng.standard_normal(size=(n_sims, months))
    growth_factors = np.exp(drift_m + sigma_m * z)

    paths = np.empty((n_sims, months + 1), dtype=np.float64)
    paths[:, 0] = float(initial_balance)

    bal = paths[:, 0].copy()
    for t in range(months):
        c = float(monthly_contributions[t])
        if c < 0:
            raise ValueError("Monthly contribution cannot be negative.")
        bal = (bal + c) * growth_factors[:, t]
        paths[:, t + 1] = bal

    return paths


# ============================================================
# Schedule editor widget (Treeview + Add/Edit/Delete)
# ============================================================
class ScheduleEditor(ttk.LabelFrame):
    """
    Stores schedule as list of (date, amount).
    Effective date applies from the 1st of that month onwards.
    """
    def __init__(self, master, title: str, *, default_items: list[tuple[date, float]] | None = None):
        super().__init__(master, text=title, padding=10)
        self._items: list[tuple[date, float]] = list(default_items) if default_items else []

        self.tree = ttk.Treeview(self, columns=("date", "amount"), show="headings", height=5)
        self.tree.heading("date", text="Effective date (YYYY-MM-01)")
        self.tree.heading("amount", text="Monthly amount (£)")
        self.tree.column("date", width=180, anchor="w")
        self.tree.column("amount", width=160, anchor="e")
        self.tree.grid(row=0, column=0, columnspan=3, sticky="ew")

        self.columnconfigure(0, weight=1)

        ttk.Button(self, text="Add", command=self._add).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(self, text="Edit", command=self._edit).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(6, 0))
        ttk.Button(self, text="Delete", command=self._delete).grid(row=1, column=2, sticky="w", pady=(8, 0), padx=(6, 0))

        self._refresh()

    def get_schedule(self) -> list[tuple[date, float]]:
        out = [(normalise_to_month_start(d), float(v)) for d, v in self._items]
        out.sort(key=lambda x: x[0])
        return out

    def set_schedule(self, items: list[tuple[date, float]]):
        self._items = list(items)
        self._refresh()

    def to_jsonable(self) -> list[dict]:
        return [{"date": date_to_iso(normalise_to_month_start(d)), "amount": float(v)} for d, v in self.get_schedule()]

    def from_jsonable(self, data) -> None:
        items: list[tuple[date, float]] = []
        if isinstance(data, list):
            for row in data:
                try:
                    d = iso_to_date(row["date"])
                    amt = float(row["amount"])
                    items.append((normalise_to_month_start(d), amt))
                except Exception:
                    continue
        self.set_schedule(items)

    def _refresh(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        for d, v in self.get_schedule():
            self.tree.insert("", "end", values=(d.isoformat(), f"{v:,.2f}"))

    def _add(self):
        self._open_editor(title="Add change", initial_date="", initial_amount="")

    def _edit(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        self._open_editor(
            title="Edit change",
            initial_date=str(vals[0]),
            initial_amount=str(vals[1]).replace(",", "").replace("£", ""),
        )

    def _delete(self):
        sel = self.tree.selection()
        if not sel:
            return
        vals = self.tree.item(sel[0], "values")
        d = datetime.strptime(vals[0], "%Y-%m-%d").date()
        target_d = normalise_to_month_start(d)

        for i, (dd, _) in enumerate(self._items):
            if normalise_to_month_start(dd) == target_d:
                self._items.pop(i)
                break
        self._refresh()

    def _open_editor(self, *, title: str, initial_date: str, initial_amount: str):
        win = tk.Toplevel(self)
        win.title(title)
        win.resizable(False, False)
        win.transient(self.winfo_toplevel())

        frm = ttk.Frame(win, padding=12)
        frm.grid(row=0, column=0)

        ttk.Label(frm, text="Effective date (YYYY-MM-01):").grid(row=0, column=0, sticky="w")
        e_date = ttk.Entry(frm, width=18)
        e_date.grid(row=0, column=1, sticky="e")
        e_date.insert(0, initial_date)

        ttk.Label(frm, text="Monthly amount (£):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        e_amt = ttk.Entry(frm, width=18)
        e_amt.grid(row=1, column=1, sticky="e", pady=(8, 0))
        e_amt.insert(0, initial_amount)

        def save():
            try:
                d = datetime.strptime(e_date.get().strip(), "%Y-%m-%d").date()
                d = normalise_to_month_start(d)
                amt = float(e_amt.get().strip().replace("£", "").replace(",", ""))
                if amt < 0:
                    raise ValueError("Amount cannot be negative.")

                replaced = False
                for i, (dd, _) in enumerate(self._items):
                    if normalise_to_month_start(dd) == d:
                        self._items[i] = (d, amt)
                        replaced = True
                        break
                if not replaced:
                    self._items.append((d, amt))

                self._refresh()
                win.destroy()
            except Exception as ex:
                messagebox.showerror("Schedule error", str(ex), parent=win)

        ttk.Button(frm, text="Save", command=save).grid(row=2, column=0, pady=(12, 0), sticky="w")
        ttk.Button(frm, text="Cancel", command=win.destroy).grid(row=2, column=1, pady=(12, 0), sticky="e")


# ============================================================
# AI prompt window
# ============================================================
class AIPromptWindow(tk.Toplevel):
    def __init__(self, master, prompt_text: str):
        super().__init__(master)
        self.title("AI Prompt (copy into ChatGPT)")
        self.geometry("920x640")
        self.minsize(920, 640)

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(
            frm,
            text="Copy this prompt into ChatGPT (or your model of choice). It includes your current forecast outputs.",
            wraplength=860,
        ).pack(anchor="w")

        text_frame = ttk.Frame(frm)
        text_frame.pack(fill="both", expand=True, pady=(10, 10))

        yscroll = ttk.Scrollbar(text_frame, orient="vertical")
        yscroll.pack(side="right", fill="y")

        self.text = tk.Text(text_frame, wrap="word", yscrollcommand=yscroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        yscroll.configure(command=self.text.yview)

        self.text.insert("1.0", prompt_text)
        self.text.configure(state="normal")

        btns = ttk.Frame(frm)
        btns.pack(fill="x")

        ttk.Button(btns, text="Copy to clipboard", command=self.copy).pack(side="left")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")

    def copy(self):
        s = self.text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(s)
        self.update()
        messagebox.showinfo("Copied", "Prompt copied to clipboard.")


# ============================================================
# Monte Carlo window
# ============================================================
class MonteCarloWindow(tk.Toplevel):
    def __init__(self, master, params: dict):
        super().__init__(master)
        self.title("Monte Carlo Simulation (HL + T212 + Cash ISA)")
        self.geometry("1280x740")
        self.minsize(1280, 740)

        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=0)
        container.columnconfigure(1, weight=1)

        left_scroll = ScrollableFrame(container, width=560, height=710)
        left_scroll.grid(row=0, column=0, sticky="ns")
        left = left_scroll.inner

        right = ttk.Frame(container)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        # ---- Controls ----
        ctrl = ttk.LabelFrame(left, text="Simulation controls", padding=10)
        ctrl.pack(fill="x")

        self.use_retirement = tk.BooleanVar(value=params.get("use_retirement", False))
        ttk.Checkbutton(
            ctrl, text="Use retirement horizon (from today)", variable=self.use_retirement, command=self._toggle_horizon_mode
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(ctrl, text="Date of birth:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.dob_entry = ttk.Entry(ctrl, width=20)
        self.dob_entry.grid(row=1, column=1, sticky="e", pady=(8, 0))
        self.dob_entry.insert(0, params.get("dob_str", "03 January 1990"))

        ttk.Label(ctrl, text="Retirement age:").grid(row=2, column=0, sticky="w")
        self.ret_age_entry = ttk.Entry(ctrl, width=20)
        self.ret_age_entry.grid(row=2, column=1, sticky="e")
        self.ret_age_entry.insert(0, str(params.get("ret_age", 60)))

        ttk.Separator(ctrl, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(ctrl, text="Years:").grid(row=4, column=0, sticky="w")
        self.years_entry = ttk.Entry(ctrl, width=20)
        self.years_entry.grid(row=4, column=1, sticky="e")
        self.years_entry.insert(0, str(params.get("years", 19)))

        ttk.Label(ctrl, text="Extra months:").grid(row=5, column=0, sticky="w")
        self.extra_months_entry = ttk.Entry(ctrl, width=20)
        self.extra_months_entry.grid(row=5, column=1, sticky="e")
        self.extra_months_entry.insert(0, str(params.get("extra_months", 0)))

        ttk.Label(ctrl, text="Start date (YYYY-MM-DD):").grid(row=6, column=0, sticky="w")
        self.start_entry = ttk.Entry(ctrl, width=20)
        self.start_entry.grid(row=6, column=1, sticky="e")
        self.start_entry.insert(0, params.get("start_str", date.today().isoformat()))

        ttk.Separator(ctrl, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(ctrl, text="Simulations:").grid(row=8, column=0, sticky="w")
        self.n_sims = ttk.Entry(ctrl, width=20)
        self.n_sims.grid(row=8, column=1, sticky="e")
        self.n_sims.insert(0, "500")

        ttk.Label(ctrl, text="Seed (optional):").grid(row=9, column=0, sticky="w")
        self.seed_entry = ttk.Entry(ctrl, width=20)
        self.seed_entry.grid(row=9, column=1, sticky="e")
        self.seed_entry.insert(0, "")

        self.apply_schedules = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Apply contribution schedules", variable=self.apply_schedules).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        # ---- Include accounts ----
        inc = ttk.LabelFrame(left, text="Include accounts", padding=10)
        inc.pack(fill="x", pady=(10, 0))

        self.hl_on = tk.BooleanVar(value=params.get("hl_enabled", True))
        self.t212_on = tk.BooleanVar(value=params.get("t212_enabled", True))
        self.cisa_on = tk.BooleanVar(value=params.get("cisa_enabled", True))

        ttk.Checkbutton(inc, text="Include HL (Hargreaves Lansdown)", variable=self.hl_on).pack(anchor="w")
        ttk.Checkbutton(inc, text="Include T212 (Trading 212)", variable=self.t212_on).pack(anchor="w")
        ttk.Checkbutton(inc, text="Include Cash ISA", variable=self.cisa_on).pack(anchor="w")

        # ---- HL ----
        hl = ttk.LabelFrame(left, text="HL (stochastic)", padding=10)
        hl.pack(fill="x", pady=(10, 0))

        self.hl_initial = self._mini_row(hl, 0, "Initial (£):", str(params.get("hl_initial", 0.0)))
        self.hl_monthly = self._mini_row(hl, 1, "Base monthly (£):", str(params.get("hl_monthly", 0.0)))
        self.hl_return = self._mini_row(hl, 2, "Return / year:", params.get("hl_rate_str", "7%"))
        self.hl_vol = self._mini_row(hl, 3, "Volatility / year:", params.get("hl_vol_str", "15%"))

        self.hl_schedule = ScheduleEditor(left, "HL contribution changes", default_items=[])
        self.hl_schedule.from_jsonable(params.get("hl_schedule_json", []))
        self.hl_schedule.pack(fill="x", pady=(10, 0))

        # ---- T212 ----
        t212 = ttk.LabelFrame(left, text="T212 (stochastic)", padding=10)
        t212.pack(fill="x", pady=(10, 0))

        self.t212_initial = self._mini_row(t212, 0, "Initial (£):", str(params.get("t212_initial", 0.0)))
        self.t212_monthly = self._mini_row(t212, 1, "Base monthly (£):", str(params.get("t212_monthly", 0.0)))
        self.t212_return = self._mini_row(t212, 2, "Return / year:", params.get("t212_rate_str", "7%"))
        self.t212_vol = self._mini_row(t212, 3, "Volatility / year:", params.get("t212_vol_str", "15%"))

        self.t212_schedule = ScheduleEditor(left, "T212 contribution changes", default_items=[])
        self.t212_schedule.from_jsonable(params.get("t212_schedule_json", []))
        self.t212_schedule.pack(fill="x", pady=(10, 0))

        # ---- Cash ISA ----
        cisa = ttk.LabelFrame(left, text="Cash ISA (usually low vol)", padding=10)
        cisa.pack(fill="x", pady=(10, 0))

        self.cisa_initial = self._mini_row(cisa, 0, "Initial (£):", str(params.get("cisa_initial", 0.0)))
        self.cisa_monthly = self._mini_row(cisa, 1, "Base monthly (£):", str(params.get("cisa_monthly", 0.0)))
        self.cisa_return = self._mini_row(cisa, 2, "Return / year:", params.get("cisa_rate_str", "4.5%"))
        self.cisa_vol = self._mini_row(cisa, 3, "Volatility / year:", params.get("cisa_vol_str", "0%"))

        self.cisa_schedule = ScheduleEditor(left, "Cash ISA contribution changes", default_items=[])
        self.cisa_schedule.from_jsonable(params.get("cisa_schedule_json", []))
        self.cisa_schedule.pack(fill="x", pady=(10, 0))

        # Run button
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Run Monte Carlo", command=self.run).pack(side="left")

        # Summary
        out = ttk.LabelFrame(left, text="End value summary (Total pot)", padding=10)
        out.pack(fill="x", pady=(10, 0))

        self.p10_var = tk.StringVar(value="—")
        self.p50_var = tk.StringVar(value="—")
        self.p90_var = tk.StringVar(value="—")

        ttk.Label(out, text="10th percentile:").grid(row=0, column=0, sticky="w")
        ttk.Label(out, textvariable=self.p10_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="e")
        ttk.Label(out, text="Median (50th):").grid(row=1, column=0, sticky="w")
        ttk.Label(out, textvariable=self.p50_var, font=("Segoe UI", 10, "bold")).grid(row=1, column=1, sticky="e")
        ttk.Label(out, text="90th percentile:").grid(row=2, column=0, sticky="w")
        ttk.Label(out, textvariable=self.p90_var, font=("Segoe UI", 10, "bold")).grid(row=2, column=1, sticky="e")

        # Chart
        chart_box = ttk.LabelFrame(right, text="Monte Carlo fan chart (Total pot)", padding=10)
        chart_box.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8.6, 5.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Month")
        self.ax.set_ylabel("Balance (£)")
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._toggle_horizon_mode()

    def _mini_row(self, parent, r, label, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=2)
        e = ttk.Entry(parent, width=20)
        e.grid(row=r, column=1, sticky="e", pady=2)
        e.insert(0, default)
        return e

    def _toggle_horizon_mode(self):
        state_manual = "disabled" if self.use_retirement.get() else "normal"
        for w in (self.years_entry, self.extra_months_entry, self.start_entry):
            w.configure(state=state_manual)

        state_ret = "normal" if self.use_retirement.get() else "disabled"
        for w in (self.dob_entry, self.ret_age_entry):
            w.configure(state=state_ret)

    def _compute_horizon(self) -> tuple[int, date]:
        if self.use_retirement.get():
            dob = parse_dob(self.dob_entry, "Date of birth")
            ret_age = parse_int(self.ret_age_entry, "Retirement age")
            start = date.today()
            retirement_date = safe_date_years_later(dob, ret_age)
            m = months_between(start, retirement_date)
            if m <= 0:
                raise ValueError("Retirement date is this month or in the past.")
            return m, start

        years = parse_int(self.years_entry, "Years")
        extra_m = parse_int(self.extra_months_entry, "Extra months")
        start = parse_date_iso(self.start_entry, "Start date")
        m = years * 12 + extra_m
        if m <= 0:
            raise ValueError("Horizon must be at least 1 month.")
        return m, start

    def _build_contribs(self, *, start: date, months: int, base_entry: ttk.Entry, schedule_editor: ScheduleEditor, apply_sched: bool):
        base = parse_float_money(base_entry, "Base monthly")
        sched = schedule_editor.get_schedule() if apply_sched else []
        return build_monthly_contribution_vector(start_date=start, total_months=months, base_monthly=base, schedule=sched)

    def run(self):
        try:
            months, start = self._compute_horizon()
            n_sims = parse_int_str(self.n_sims.get(), "Simulations")
            if n_sims <= 0:
                raise ValueError("Simulations must be at least 1.")

            seed_txt = self.seed_entry.get().strip()
            seed = int(seed_txt) if seed_txt else None
            apply_sched = bool(self.apply_schedules.get())

            # HL
            if self.hl_on.get():
                hl_initial = parse_float_money(self.hl_initial, "HL initial")
                hl_ret = parse_rate(self.hl_return, "HL return")
                hl_vol = parse_rate(self.hl_vol, "HL volatility")
                hl_contribs = self._build_contribs(start=start, months=months, base_entry=self.hl_monthly, schedule_editor=self.hl_schedule, apply_sched=apply_sched)
                hl_paths = monte_carlo_paths_lognormal_monthly_variable_contrib(
                    initial_balance=hl_initial,
                    monthly_contributions=hl_contribs,
                    annual_return=hl_ret,
                    annual_vol=hl_vol,
                    n_sims=n_sims,
                    seed=seed,
                )
            else:
                hl_paths = np.zeros((n_sims, months + 1), dtype=np.float64)

            # T212
            if self.t212_on.get():
                t212_initial = parse_float_money(self.t212_initial, "T212 initial")
                t212_ret = parse_rate(self.t212_return, "T212 return")
                t212_vol = parse_rate(self.t212_vol, "T212 volatility")
                t212_contribs = self._build_contribs(start=start, months=months, base_entry=self.t212_monthly, schedule_editor=self.t212_schedule, apply_sched=apply_sched)
                t212_paths = monte_carlo_paths_lognormal_monthly_variable_contrib(
                    initial_balance=t212_initial,
                    monthly_contributions=t212_contribs,
                    annual_return=t212_ret,
                    annual_vol=t212_vol,
                    n_sims=n_sims,
                    seed=None if seed is None else seed + 7,
                )
            else:
                t212_paths = np.zeros((n_sims, months + 1), dtype=np.float64)

            # Cash ISA
            if self.cisa_on.get():
                cisa_initial = parse_float_money(self.cisa_initial, "Cash ISA initial")
                cisa_ret = parse_rate(self.cisa_return, "Cash ISA return")
                cisa_vol = parse_rate(self.cisa_vol, "Cash ISA volatility")
                cisa_contribs = self._build_contribs(start=start, months=months, base_entry=self.cisa_monthly, schedule_editor=self.cisa_schedule, apply_sched=apply_sched)
                cisa_paths = monte_carlo_paths_lognormal_monthly_variable_contrib(
                    initial_balance=cisa_initial,
                    monthly_contributions=cisa_contribs,
                    annual_return=cisa_ret,
                    annual_vol=cisa_vol,
                    n_sims=n_sims,
                    seed=None if seed is None else seed + 11,
                )
            else:
                cisa_paths = np.zeros((n_sims, months + 1), dtype=np.float64)

            total_paths = hl_paths + t212_paths + cisa_paths

            p10 = np.percentile(total_paths, 10, axis=0)
            p50 = np.percentile(total_paths, 50, axis=0)
            p90 = np.percentile(total_paths, 90, axis=0)

            end_vals = total_paths[:, -1]
            end_p10, end_p50, end_p90 = np.percentile(end_vals, [10, 50, 90])

            self.p10_var.set(f"£{end_p10:,.0f}")
            self.p50_var.set(f"£{end_p50:,.0f}")
            self.p90_var.set(f"£{end_p90:,.0f}")

            self.ax.clear()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel("Month")
            self.ax.set_ylabel("Balance (£)")
            self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

            x = np.arange(months + 1)
            self.ax.fill_between(x, p10, p90, alpha=0.25, label="10–90% band")
            self.ax.plot(x, p50, label="Median (50%)")
            self.ax.legend()

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Monte Carlo error", str(e))


# ============================================================
# Main deterministic app
# ============================================================
class GrowthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Money Growth Forecaster (HL + T212 + Cash ISA)")
        self.geometry("1340x780")
        self.minsize(1340, 780)

        self.style = ttk.Style()
        for theme in ("vista", "clam", "alt", "default"):
            try:
                self.style.theme_use(theme)
                break
            except tk.TclError:
                pass

        self._settings = self._load_and_migrate_settings()
        self._last_outputs: dict | None = None  # used by the AI prompt button

        main = ttk.Frame(self, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)

        left_scroll = ScrollableFrame(main, width=600, height=750)
        left_scroll.grid(row=0, column=0, sticky="ns")
        left = left_scroll.inner

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(14, 0))

        # --- Retirement settings ---
        retirement = ttk.LabelFrame(left, text="Retirement settings (optional)", padding=12)
        retirement.pack(fill="x")

        self.use_retirement = tk.BooleanVar(value=bool(self._settings.get("use_retirement", False)))
        ttk.Checkbutton(
            retirement,
            text="Use retirement horizon (from today) — overrides manual horizon",
            variable=self.use_retirement,
            command=self._toggle_retirement_mode,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(retirement, text="Date of birth:").grid(row=1, column=0, sticky="w", pady=4)
        self.dob_entry = ttk.Entry(retirement, width=24)
        self.dob_entry.grid(row=1, column=1, sticky="e", pady=4)
        self.dob_entry.insert(0, self._settings.get("dob_str", "03 January 1990"))

        ttk.Label(retirement, text="Retirement age:").grid(row=2, column=0, sticky="w", pady=4)
        self.ret_age_entry = ttk.Entry(retirement, width=24)
        self.ret_age_entry.grid(row=2, column=1, sticky="e", pady=4)
        self.ret_age_entry.insert(0, str(self._settings.get("ret_age", 60)))

        self.months_to_ret_var = tk.StringVar(value="—")
        ttk.Label(retirement, text="Months to retirement:").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(retirement, textvariable=self.months_to_ret_var, font=("Segoe UI", 10, "bold")).grid(
            row=3, column=1, sticky="e", pady=4
        )

        # Patch that actually works: StringVars + trace
        self._dob_var = tk.StringVar(value=self.dob_entry.get())
        self._retage_var = tk.StringVar(value=self.ret_age_entry.get())
        self.dob_entry.configure(textvariable=self._dob_var)
        self.ret_age_entry.configure(textvariable=self._retage_var)
        self._dob_var.trace_add("write", lambda *_: self._update_months_to_retirement())
        self._retage_var.trace_add("write", lambda *_: self._update_months_to_retirement())

        # --- Manual horizon ---
        horizon = ttk.LabelFrame(left, text="Manual horizon", padding=12)
        horizon.pack(fill="x", pady=(10, 0))

        self.years_entry = self._row(horizon, 0, "Years:", str(self._settings.get("years", 19)))
        self.months_entry = self._row(horizon, 1, "Extra months:", str(self._settings.get("extra_months", 0)))
        self.start_entry = self._row(horizon, 2, "Start date (YYYY-MM-DD):", self._settings.get("start_str", ""))

        # --- Stocks & Shares ISAs (tabs) ---
        ssisa_box = ttk.LabelFrame(left, text="Stocks & Shares ISAs", padding=12)
        ssisa_box.pack(fill="x", pady=(10, 0))

        self.ss_tabs = ttk.Notebook(ssisa_box)
        self.ss_tabs.grid(row=0, column=0, sticky="ew")
        ssisa_box.columnconfigure(0, weight=1)

        # HL tab
        self.hl_tab = ttk.Frame(self.ss_tabs)
        self.ss_tabs.add(self.hl_tab, text="HL (Hargreaves Lansdown)")

        self.hl_enabled = tk.BooleanVar(value=bool(self._settings.get("hl_enabled", True)))
        hl_toggle = ttk.Frame(self.hl_tab)
        hl_toggle.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(hl_toggle, text="Include HL", variable=self.hl_enabled, command=self._toggle_hl).pack(anchor="w")

        self.hl_initial_entry = self._row(self.hl_tab, 1, "Initial balance (£):", str(self._settings.get("hl_initial", 3500)))
        self.hl_monthly_entry = self._row(self.hl_tab, 2, "Base monthly contribution (£):", str(self._settings.get("hl_monthly", 200)))
        self.hl_rate_entry = self._row(self.hl_tab, 3, "Annual return (e.g. 7%):", str(self._settings.get("hl_rate_str", "7%")))

        self.hl_schedule = ScheduleEditor(left, "HL contribution changes (optional)")
        self.hl_schedule.from_jsonable(self._settings.get("hl_schedule_json", []))
        self.hl_schedule.pack(fill="x", pady=(10, 0))

        # T212 tab
        self.t212_tab = ttk.Frame(self.ss_tabs)
        self.ss_tabs.add(self.t212_tab, text="T212 (Trading 212)")

        self.t212_enabled = tk.BooleanVar(value=bool(self._settings.get("t212_enabled", True)))
        t212_toggle = ttk.Frame(self.t212_tab)
        t212_toggle.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(t212_toggle, text="Include T212", variable=self.t212_enabled, command=self._toggle_t212).pack(anchor="w")

        self.t212_initial_entry = self._row(self.t212_tab, 1, "Initial balance (£):", str(self._settings.get("t212_initial", 0)))
        self.t212_monthly_entry = self._row(self.t212_tab, 2, "Base monthly contribution (£):", str(self._settings.get("t212_monthly", 0)))
        self.t212_rate_entry = self._row(self.t212_tab, 3, "Annual return (e.g. 7%):", str(self._settings.get("t212_rate_str", "7%")))

        self.t212_schedule = ScheduleEditor(left, "T212 contribution changes (optional)")
        self.t212_schedule.from_jsonable(self._settings.get("t212_schedule_json", []))
        self.t212_schedule.pack(fill="x", pady=(10, 0))

        # --- Cash ISA ---
        cisa = ttk.LabelFrame(left, text="Cash ISA", padding=12)
        cisa.pack(fill="x", pady=(10, 0))

        self.cisa_enabled = tk.BooleanVar(value=bool(self._settings.get("cisa_enabled", True)))
        cisa_toggle_row = ttk.Frame(cisa)
        cisa_toggle_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(cisa_toggle_row, text="Include Cash ISA", variable=self.cisa_enabled, command=self._toggle_cisa).pack(anchor="w")

        self.cisa_initial_entry = self._row(cisa, 1, "Initial balance (£):", str(self._settings.get("cisa_initial", 0)))
        self.cisa_monthly_entry = self._row(cisa, 2, "Base monthly contribution (£):", str(self._settings.get("cisa_monthly", 0)))
        self.cisa_rate_entry = self._row(cisa, 3, "Annual rate (e.g. 4.5%):", str(self._settings.get("cisa_rate_str", "4.5%")))

        self.cisa_schedule = ScheduleEditor(left, "Cash ISA contribution changes (optional)")
        self.cisa_schedule.from_jsonable(self._settings.get("cisa_schedule_json", []))
        self.cisa_schedule.pack(fill="x", pady=(10, 0))

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Calculate", command=self.on_calculate).pack(side="left")
        ttk.Button(btns, text="Reset", command=self.on_reset).pack(side="left", padx=8)
        ttk.Button(btns, text="Monte Carlo…", command=self.open_monte_carlo).pack(side="left", padx=8)

        # AI Prompt button (NEW)
        self.ai_btn = ttk.Button(btns, text="AI Prompt…", command=self.open_ai_prompt)
        self.ai_btn.pack(side="left", padx=8)
        self.ai_btn.state(["disabled"])  # enabled after first successful Calculate

        # Results
        outputs = ttk.LabelFrame(left, text="End of horizon", padding=12)
        outputs.pack(fill="x", pady=(10, 0))

        self.total_with_var = tk.StringVar(value="—")
        self.total_invested_var = tk.StringVar(value="—")
        self.total_growth_var = tk.StringVar(value="—")

        self.hl_with_var = tk.StringVar(value="—")
        self.hl_inv_var = tk.StringVar(value="—")
        self.hl_growth_var = tk.StringVar(value="—")

        self.t212_with_var = tk.StringVar(value="—")
        self.t212_inv_var = tk.StringVar(value="—")
        self.t212_growth_var = tk.StringVar(value="—")

        self.ss_total_with_var = tk.StringVar(value="—")
        self.ss_total_inv_var = tk.StringVar(value="—")
        self.ss_total_growth_var = tk.StringVar(value="—")

        self.cisa_with_var = tk.StringVar(value="—")
        self.cisa_inv_var = tk.StringVar(value="—")
        self.cisa_growth_var = tk.StringVar(value="—")

        self._out_row(outputs, 0, "Total pot (with interest):", self.total_with_var)
        self._out_row(outputs, 1, "Total invested:", self.total_invested_var)
        self._out_row(outputs, 2, "Total growth:", self.total_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 4, "HL amount:", self.hl_with_var)
        self._out_row(outputs, 5, "HL invested:", self.hl_inv_var)
        self._out_row(outputs, 6, "HL growth:", self.hl_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 8, "T212 amount:", self.t212_with_var)
        self._out_row(outputs, 9, "T212 invested:", self.t212_inv_var)
        self._out_row(outputs, 10, "T212 growth:", self.t212_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=11, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 12, "S&S ISA total amount:", self.ss_total_with_var)
        self._out_row(outputs, 13, "S&S ISA total invested:", self.ss_total_inv_var)
        self._out_row(outputs, 14, "S&S ISA total growth:", self.ss_total_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=15, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 16, "Cash ISA amount:", self.cisa_with_var)
        self._out_row(outputs, 17, "Cash ISA invested:", self.cisa_inv_var)
        self._out_row(outputs, 18, "Cash ISA growth:", self.cisa_growth_var)

        # Chart
        chart_box = ttk.LabelFrame(right, text="Deterministic projection (daily compounding)", padding=10)
        chart_box.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(9.2, 6.0), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Month")
        self.ax.set_ylabel("Balance (£)")
        self.ax.grid(True, alpha=0.3)
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._plot(
            labels=["Start"],
            total_with=[0],
            total_invested=[0],
            hl_with=[0],
            t212_with=[0],
            cisa_with=[0],
            hl_on=True,
            t212_on=True,
            cisa_on=True,
        )

        # initial toggles + initial months-to-ret update
        self._toggle_retirement_mode()
        self._update_months_to_retirement()
        self._toggle_hl()
        self._toggle_t212()
        self._toggle_cisa()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Patch method ----------
    def _update_months_to_retirement(self):
        """
        Updates 'Months to retirement' LIVE, regardless of whether retirement mode is enabled.
        (Shows '—' if invalid.)
        """
        try:
            dob = parse_dob_str(self._dob_var.get(), "Date of birth")
            ret_age = parse_int_str(self._retage_var.get(), "Retirement age")
            retirement_date = safe_date_years_later(dob, ret_age)
            m = months_between(date.today(), retirement_date)
            self.months_to_ret_var.set(str(m) if m > 0 else "0")
        except Exception:
            self.months_to_ret_var.set("—")

    # ---------- Settings migration ----------
    def _load_and_migrate_settings(self) -> dict:
        s = safe_load_settings()

        # Backwards-compat: older versions had a single "ssisa_*"
        if "hl_initial" not in s and "ssisa_initial" in s:
            s["hl_enabled"] = bool(s.get("ssisa_enabled", True))
            s["hl_initial"] = s.get("ssisa_initial", "3500")
            s["hl_monthly"] = s.get("ssisa_monthly", "200")
            s["hl_rate_str"] = s.get("ssisa_rate_str", "7%")
            s["hl_schedule_json"] = s.get("ssisa_schedule_json", [])

        s.setdefault("t212_enabled", True)
        s.setdefault("t212_initial", "0")
        s.setdefault("t212_monthly", "0")
        s.setdefault("t212_rate_str", "7%")
        s.setdefault("t212_schedule_json", [])

        s.setdefault("hl_enabled", True)
        s.setdefault("hl_initial", "3500")
        s.setdefault("hl_monthly", "200")
        s.setdefault("hl_rate_str", "7%")
        s.setdefault("hl_schedule_json", [])

        s.setdefault("cisa_enabled", True)
        s.setdefault("cisa_initial", "0")
        s.setdefault("cisa_monthly", "0")
        s.setdefault("cisa_rate_str", "4.5%")
        s.setdefault("cisa_schedule_json", [])

        s.setdefault("use_retirement", False)
        s.setdefault("dob_str", "03 January 1990")
        s.setdefault("ret_age", 60)
        s.setdefault("years", 19)
        s.setdefault("extra_months", 0)
        s.setdefault("start_str", "")

        return s

    # ---------- UI helpers ----------
    def _row(self, parent, r, label, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        e = ttk.Entry(parent, width=24)
        e.grid(row=r, column=1, sticky="e", pady=4)
        e.insert(0, default)
        return e

    def _out_row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Label(parent, textvariable=var, font=("Segoe UI", 10, "bold")).grid(row=r, column=1, sticky="e", pady=4)

    # ---------- Toggles ----------
    def _toggle_retirement_mode(self):
        manual_state = "disabled" if self.use_retirement.get() else "normal"
        for w in (self.years_entry, self.months_entry, self.start_entry):
            w.configure(state=manual_state)

        ret_state = "normal" if self.use_retirement.get() else "disabled"
        for w in (self.dob_entry, self.ret_age_entry):
            w.configure(state=ret_state)

        self._update_months_to_retirement()

    def _toggle_hl(self):
        state = "normal" if self.hl_enabled.get() else "disabled"
        for w in (self.hl_initial_entry, self.hl_monthly_entry, self.hl_rate_entry):
            w.configure(state=state)

    def _toggle_t212(self):
        state = "normal" if self.t212_enabled.get() else "disabled"
        for w in (self.t212_initial_entry, self.t212_monthly_entry, self.t212_rate_entry):
            w.configure(state=state)

    def _toggle_cisa(self):
        state = "normal" if self.cisa_enabled.get() else "disabled"
        for w in (self.cisa_initial_entry, self.cisa_monthly_entry, self.cisa_rate_entry):
            w.configure(state=state)

    # ---------- Horizon ----------
    def _get_horizon(self) -> tuple[int, date, list[str]]:
        if self.use_retirement.get():
            dob = parse_dob_str(self._dob_var.get(), "Date of birth")
            ret_age = parse_int_str(self._retage_var.get(), "Retirement age")
            start = date.today()
            retirement_date = safe_date_years_later(dob, ret_age)
            total_months = months_between(start, retirement_date)
            if total_months <= 0:
                raise ValueError("Retirement date is this month or in the past.")
            labels = make_month_labels(total_months, start)
            return total_months, start, labels

        years = parse_int(self.years_entry, "Years")
        extra_months = parse_int(self.months_entry, "Extra months")
        start = parse_date_iso(self.start_entry, "Start date")

        if years < 0 or extra_months < 0:
            raise ValueError("Years and extra months cannot be negative.")

        total_months = years * 12 + extra_months
        if total_months <= 0:
            raise ValueError("Duration must be at least 1 month.")

        labels = make_month_labels(total_months, start)
        return total_months, start, labels

    # ---------- Plot ----------
    def _plot(
        self,
        *,
        labels,
        total_with,
        total_invested,
        hl_with,
        t212_with,
        cisa_with,
        hl_on: bool,
        t212_on: bool,
        cisa_on: bool,
    ):
        self.ax.clear()

        self.ax.plot(total_with, label="Total pot (with interest)")
        self.ax.plot(total_invested, label="Total invested")

        if hl_on:
            self.ax.plot(hl_with, label="HL (with interest)")
        if t212_on:
            self.ax.plot(t212_with, label="T212 (with interest)")
        if cisa_on:
            self.ax.plot(cisa_with, label="Cash ISA (with interest)")

        self.ax.set_xlabel("Month")
        self.ax.set_ylabel("Balance (£)")
        self.ax.grid(True, alpha=0.3)
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

        n = len(labels)
        if n <= 15:
            ticks = list(range(n))
            tick_labels = labels
        else:
            step = max(1, n // 10)
            ticks = list(range(0, n, step))
            tick_labels = [labels[i] for i in ticks]

        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        self.ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- AI Prompt (NEW) ----------
    def open_ai_prompt(self):
        if not self._last_outputs:
            messagebox.showinfo("AI Prompt", "Run Calculate first so the app has outputs to include in the prompt.")
            return
        prompt = self._build_ai_prompt(self._last_outputs)
        AIPromptWindow(self, prompt)

    def _fmt_money(self, x: float, decimals: int = 2) -> str:
        return f"£{x:,.{decimals}f}"

    def _fmt_rate(self, r: float) -> str:
        return f"{r*100:.2f}%"

    def _next_changes_preview(self, schedule_items: list[tuple[date, float]]) -> str:
        future = [(d, v) for d, v in schedule_items if d >= normalise_to_month_start(date.today())]
        future.sort(key=lambda t: t[0])
        if not future:
            return "No scheduled future contribution changes."
        lines = []
        for d, v in future[:5]:
            lines.append(f"- From {d.isoformat()}: {self._fmt_money(v)} per month")
        if len(future) > 5:
            lines.append(f"- (+{len(future)-5} more changes)")
        return "\n".join(lines)

    def _build_ai_prompt(self, out: dict) -> str:
        # Creative but structured prompt that uses the app outputs + assumptions.
        today = date.today().isoformat()
        months_to_ret = out.get("months_to_retirement_str", "—")
        retirement_mode = out.get("use_retirement", False)

        return f"""You are a UK-focused retirement planning coach and risk manager. Use my forecast outputs below to give me a practical plan for what to do as I approach retirement. Be realistic, highlight risks, and suggest sensible next steps.

IMPORTANT: Ask any clarifying questions you genuinely need, but still provide a best-effort plan with assumptions.

=====================================
MY CURRENT FORECAST (from my app)
Date generated: {today}

Horizon mode: {"Retirement horizon (from today)" if retirement_mode else "Manual horizon"}
Months to retirement (if relevant): {months_to_ret}

Account toggles:
- HL included: {out["hl_on"]}
- T212 included: {out["t212_on"]}
- Cash ISA included: {out["cisa_on"]}

Assumptions used in deterministic projection (daily compounding):
- HL annual return: {out["hl_rate_str"]} (input)
- T212 annual return: {out["t212_rate_str"]} (input)
- Cash ISA annual rate: {out["cisa_rate_str"]} (input)

Contribution plans:
HL base monthly: {self._fmt_money(out["hl_base_monthly"])}  | schedule notes:
{self._next_changes_preview(out["hl_schedule"])}

T212 base monthly: {self._fmt_money(out["t212_base_monthly"])} | schedule notes:
{self._next_changes_preview(out["t212_schedule"])}

Cash ISA base monthly: {self._fmt_money(out["cisa_base_monthly"])} | schedule notes:
{self._next_changes_preview(out["cisa_schedule"])}

END-OF-HORIZON OUTPUTS:
TOTAL POT:
- Total with interest: {self._fmt_money(out["end_total_with"])}
- Total invested:      {self._fmt_money(out["end_total_invested"])}
- Total growth:        {self._fmt_money(out["end_total_growth"])}

HL:
- Amount:   {self._fmt_money(out["end_hl_with"])}  | Invested: {self._fmt_money(out["end_hl_invested"])}  | Growth: {self._fmt_money(out["end_hl_growth"])}

T212:
- Amount:   {self._fmt_money(out["end_t212_with"])} | Invested: {self._fmt_money(out["end_t212_invested"])} | Growth: {self._fmt_money(out["end_t212_growth"])}

S&S ISA (HL + T212 total):
- Amount:   {self._fmt_money(out["end_ss_with"])}  | Invested: {self._fmt_money(out["end_ss_invested"])}  | Growth: {self._fmt_money(out["end_ss_growth"])}

Cash ISA:
- Amount:   {self._fmt_money(out["end_cisa_with"])} | Invested: {self._fmt_money(out["end_cisa_invested"])} | Growth: {self._fmt_money(out["end_cisa_growth"])}

=====================================
WHAT I WANT FROM YOU
1) Give me a “glidepath” plan for the final 10 years to retirement and the first 10 years of retirement:
   - How (roughly) should I adjust risk as I get closer?
   - What should I do about sequence-of-returns risk?
   - Should I build a cash buffer (and how big, conceptually)?

2) Based on the total pot above, give *illustrative* withdrawal ranges:
   - A conservative range, a mid-range, and an aggressive range (explain trade-offs).
   - Explain in plain English what could go wrong and how to mitigate it.

3) Tell me what to monitor each year and each quarter:
   - “If X happens, do Y” style rules (e.g., markets down 25%, contributions change, retirement age shifts).

4) Tell me what questions you’d ask a financial adviser (or what I should research) next.
   Make it UK-aware (ISAs, tax wrapper strategy, drawdown concepts), but do NOT invent personal tax details.

5) Bonus: Suggest an experiment plan:
   - A conservative Monte Carlo setting (return + volatility) to stress-test this,
   - and what outputs I should focus on (p10/p50/p90, failure rates, worst drawdown, etc.).

Use my numbers above heavily. Be concrete, not generic.
"""

    # ---------- Actions ----------
    def on_calculate(self):
        try:
            total_months, start, labels = self._get_horizon()

            # HL
            hl_on = self.hl_enabled.get()
            if hl_on:
                hl_initial = parse_float_money(self.hl_initial_entry, "HL initial balance")
                hl_base = parse_float_money(self.hl_monthly_entry, "HL base monthly contribution")
                hl_rate = parse_rate(self.hl_rate_entry, "HL annual return")
                hl_contribs = build_monthly_contribution_vector(
                    start_date=start,
                    total_months=total_months,
                    base_monthly=hl_base,
                    schedule=self.hl_schedule.get_schedule(),
                )
                labels_hl, hl_with, hl_inv = build_account_monthly_series_daily_variable(
                    initial_balance=hl_initial,
                    monthly_contributions=hl_contribs,
                    annual_rate=hl_rate,
                    start_date=start,
                )
                if labels_hl != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                hl_initial = 0.0
                hl_base = 0.0
                hl_rate = 0.0
                hl_contribs = [0.0] * total_months
                hl_with = [0.0] * len(labels)
                hl_inv = [0.0] * len(labels)

            # T212
            t212_on = self.t212_enabled.get()
            if t212_on:
                t212_initial = parse_float_money(self.t212_initial_entry, "T212 initial balance")
                t212_base = parse_float_money(self.t212_monthly_entry, "T212 base monthly contribution")
                t212_rate = parse_rate(self.t212_rate_entry, "T212 annual return")
                t212_contribs = build_monthly_contribution_vector(
                    start_date=start,
                    total_months=total_months,
                    base_monthly=t212_base,
                    schedule=self.t212_schedule.get_schedule(),
                )
                labels_t, t212_with, t212_inv = build_account_monthly_series_daily_variable(
                    initial_balance=t212_initial,
                    monthly_contributions=t212_contribs,
                    annual_rate=t212_rate,
                    start_date=start,
                )
                if labels_t != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                t212_initial = 0.0
                t212_base = 0.0
                t212_rate = 0.0
                t212_contribs = [0.0] * total_months
                t212_with = [0.0] * len(labels)
                t212_inv = [0.0] * len(labels)

            # Cash ISA
            cisa_on = self.cisa_enabled.get()
            if cisa_on:
                cisa_initial = parse_float_money(self.cisa_initial_entry, "Cash ISA initial balance")
                cisa_base = parse_float_money(self.cisa_monthly_entry, "Cash ISA base monthly contribution")
                cisa_rate = parse_rate(self.cisa_rate_entry, "Cash ISA annual rate")
                cisa_contribs = build_monthly_contribution_vector(
                    start_date=start,
                    total_months=total_months,
                    base_monthly=cisa_base,
                    schedule=self.cisa_schedule.get_schedule(),
                )
                labels_c, cisa_with, cisa_inv = build_account_monthly_series_daily_variable(
                    initial_balance=cisa_initial,
                    monthly_contributions=cisa_contribs,
                    annual_rate=cisa_rate,
                    start_date=start,
                )
                if labels_c != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                cisa_initial = 0.0
                cisa_base = 0.0
                cisa_rate = 0.0
                cisa_contribs = [0.0] * total_months
                cisa_with = [0.0] * len(labels)
                cisa_inv = [0.0] * len(labels)

            # Totals
            ss_with = combine_series(hl_with, t212_with)
            ss_inv = combine_series(hl_inv, t212_inv)
            total_with = combine_series(ss_with, cisa_with)
            total_inv = combine_series(ss_inv, cisa_inv)

            def end_triplet(with_series, inv_series):
                end_with = with_series[-1]
                end_inv = inv_series[-1]
                end_growth = round(end_with - end_inv, 2)
                return end_with, end_inv, end_growth

            end_total_with, end_total_inv, end_total_growth = end_triplet(total_with, total_inv)
            end_hl_with, end_hl_inv, end_hl_growth = end_triplet(hl_with, hl_inv)
            end_t212_with, end_t212_inv, end_t212_growth = end_triplet(t212_with, t212_inv)
            end_ss_with, end_ss_inv, end_ss_growth = end_triplet(ss_with, ss_inv)
            end_cisa_with, end_cisa_inv, end_cisa_growth = end_triplet(cisa_with, cisa_inv)

            # Update UI
            self.total_with_var.set(f"£{end_total_with:,.2f}")
            self.total_invested_var.set(f"£{end_total_inv:,.2f}")
            self.total_growth_var.set(f"£{end_total_growth:,.2f}")

            self.hl_with_var.set(f"£{end_hl_with:,.2f}" if hl_on else "—")
            self.hl_inv_var.set(f"£{end_hl_inv:,.2f}" if hl_on else "—")
            self.hl_growth_var.set(f"£{end_hl_growth:,.2f}" if hl_on else "—")

            self.t212_with_var.set(f"£{end_t212_with:,.2f}" if t212_on else "—")
            self.t212_inv_var.set(f"£{end_t212_inv:,.2f}" if t212_on else "—")
            self.t212_growth_var.set(f"£{end_t212_growth:,.2f}" if t212_on else "—")

            self.ss_total_with_var.set(f"£{end_ss_with:,.2f}")
            self.ss_total_inv_var.set(f"£{end_ss_inv:,.2f}")
            self.ss_total_growth_var.set(f"£{end_ss_growth:,.2f}")

            self.cisa_with_var.set(f"£{end_cisa_with:,.2f}" if cisa_on else "—")
            self.cisa_inv_var.set(f"£{end_cisa_inv:,.2f}" if cisa_on else "—")
            self.cisa_growth_var.set(f"£{end_cisa_growth:,.2f}" if cisa_on else "—")

            self._plot(
                labels=labels,
                total_with=total_with,
                total_invested=total_inv,
                hl_with=hl_with,
                t212_with=t212_with,
                cisa_with=cisa_with,
                hl_on=hl_on,
                t212_on=t212_on,
                cisa_on=cisa_on,
            )

            # Store outputs for AI prompt button (NEW)
            self._last_outputs = {
                "use_retirement": bool(self.use_retirement.get()),
                "months_to_retirement_str": self.months_to_ret_var.get(),
                "hl_on": bool(hl_on),
                "t212_on": bool(t212_on),
                "cisa_on": bool(cisa_on),

                "hl_rate_str": self.hl_rate_entry.get().strip(),
                "t212_rate_str": self.t212_rate_entry.get().strip(),
                "cisa_rate_str": self.cisa_rate_entry.get().strip(),

                "hl_base_monthly": float(hl_base),
                "t212_base_monthly": float(t212_base),
                "cisa_base_monthly": float(cisa_base),

                "hl_schedule": self.hl_schedule.get_schedule(),
                "t212_schedule": self.t212_schedule.get_schedule(),
                "cisa_schedule": self.cisa_schedule.get_schedule(),

                "end_total_with": float(end_total_with),
                "end_total_invested": float(end_total_inv),
                "end_total_growth": float(end_total_growth),

                "end_hl_with": float(end_hl_with),
                "end_hl_invested": float(end_hl_inv),
                "end_hl_growth": float(end_hl_growth),

                "end_t212_with": float(end_t212_with),
                "end_t212_invested": float(end_t212_inv),
                "end_t212_growth": float(end_t212_growth),

                "end_ss_with": float(end_ss_with),
                "end_ss_invested": float(end_ss_inv),
                "end_ss_growth": float(end_ss_growth),

                "end_cisa_with": float(end_cisa_with),
                "end_cisa_invested": float(end_cisa_inv),
                "end_cisa_growth": float(end_cisa_growth),
            }
            self.ai_btn.state(["!disabled"])

            self._save_settings()

        except Exception as e:
            messagebox.showerror("Input error", str(e))

    def on_reset(self):
        self.use_retirement.set(False)
        self._dob_var.set("03 January 1990")
        self._retage_var.set("60")

        self.years_entry.delete(0, tk.END); self.years_entry.insert(0, "19")
        self.months_entry.delete(0, tk.END); self.months_entry.insert(0, "0")
        self.start_entry.delete(0, tk.END)

        self.hl_enabled.set(True)
        self.hl_initial_entry.delete(0, tk.END); self.hl_initial_entry.insert(0, "3500")
        self.hl_monthly_entry.delete(0, tk.END); self.hl_monthly_entry.insert(0, "200")
        self.hl_rate_entry.delete(0, tk.END); self.hl_rate_entry.insert(0, "7%")
        self.hl_schedule.set_schedule([])

        self.t212_enabled.set(True)
        self.t212_initial_entry.delete(0, tk.END); self.t212_initial_entry.insert(0, "0")
        self.t212_monthly_entry.delete(0, tk.END); self.t212_monthly_entry.insert(0, "0")
        self.t212_rate_entry.delete(0, tk.END); self.t212_rate_entry.insert(0, "7%")
        self.t212_schedule.set_schedule([])

        self.cisa_enabled.set(True)
        self.cisa_initial_entry.delete(0, tk.END); self.cisa_initial_entry.insert(0, "0")
        self.cisa_monthly_entry.delete(0, tk.END); self.cisa_monthly_entry.insert(0, "0")
        self.cisa_rate_entry.delete(0, tk.END); self.cisa_rate_entry.insert(0, "4.5%")
        self.cisa_schedule.set_schedule([])

        self._toggle_retirement_mode()
        self._update_months_to_retirement()
        self._toggle_hl()
        self._toggle_t212()
        self._toggle_cisa()

        for var in (
            self.total_with_var, self.total_invested_var, self.total_growth_var,
            self.hl_with_var, self.hl_inv_var, self.hl_growth_var,
            self.t212_with_var, self.t212_inv_var, self.t212_growth_var,
            self.ss_total_with_var, self.ss_total_inv_var, self.ss_total_growth_var,
            self.cisa_with_var, self.cisa_inv_var, self.cisa_growth_var,
        ):
            var.set("—")

        self._plot(
            labels=["Start"],
            total_with=[0],
            total_invested=[0],
            hl_with=[0],
            t212_with=[0],
            cisa_with=[0],
            hl_on=True,
            t212_on=True,
            cisa_on=True,
        )

        self._last_outputs = None
        self.ai_btn.state(["disabled"])

        self._save_settings()

    def open_monte_carlo(self):
        try:
            params = self._snapshot_params_for_mc()
            MonteCarloWindow(self, params)
        except Exception as e:
            messagebox.showerror("Monte Carlo setup error", str(e))

    def _snapshot_params_for_mc(self) -> dict:
        _ = parse_rate(self.hl_rate_entry, "HL annual return")
        _ = parse_rate(self.t212_rate_entry, "T212 annual return")
        _ = parse_rate(self.cisa_rate_entry, "Cash ISA annual rate")

        start_str = self.start_entry.get().strip() or date.today().isoformat()

        return {
            "use_retirement": bool(self.use_retirement.get()),
            "dob_str": self._dob_var.get().strip() or "03 January 1990",
            "ret_age": int(self._retage_var.get().strip() or "60"),
            "years": int(self.years_entry.get().strip() or "19"),
            "extra_months": int(self.months_entry.get().strip() or "0"),
            "start_str": start_str,

            "hl_enabled": bool(self.hl_enabled.get()),
            "t212_enabled": bool(self.t212_enabled.get()),
            "cisa_enabled": bool(self.cisa_enabled.get()),

            "hl_initial": parse_float_money(self.hl_initial_entry, "HL initial balance"),
            "hl_monthly": parse_float_money(self.hl_monthly_entry, "HL base monthly contribution"),
            "hl_rate_str": self.hl_rate_entry.get().strip(),
            "hl_vol_str": "15%",
            "hl_schedule_json": self.hl_schedule.to_jsonable(),

            "t212_initial": parse_float_money(self.t212_initial_entry, "T212 initial balance"),
            "t212_monthly": parse_float_money(self.t212_monthly_entry, "T212 base monthly contribution"),
            "t212_rate_str": self.t212_rate_entry.get().strip(),
            "t212_vol_str": "15%",
            "t212_schedule_json": self.t212_schedule.to_jsonable(),

            "cisa_initial": parse_float_money(self.cisa_initial_entry, "Cash ISA initial balance"),
            "cisa_monthly": parse_float_money(self.cisa_monthly_entry, "Cash ISA base monthly contribution"),
            "cisa_rate_str": self.cisa_rate_entry.get().strip(),
            "cisa_vol_str": "0%",
            "cisa_schedule_json": self.cisa_schedule.to_jsonable(),
        }

    # ---------- Settings save/load ----------
    def _collect_settings(self) -> dict:
        return {
            "use_retirement": bool(self.use_retirement.get()),
            "dob_str": self._dob_var.get(),
            "ret_age": self._retage_var.get(),
            "years": self.years_entry.get().strip(),
            "extra_months": self.months_entry.get().strip(),
            "start_str": self.start_entry.get().strip(),

            "hl_enabled": bool(self.hl_enabled.get()),
            "hl_initial": self.hl_initial_entry.get().strip(),
            "hl_monthly": self.hl_monthly_entry.get().strip(),
            "hl_rate_str": self.hl_rate_entry.get().strip(),
            "hl_schedule_json": self.hl_schedule.to_jsonable(),

            "t212_enabled": bool(self.t212_enabled.get()),
            "t212_initial": self.t212_initial_entry.get().strip(),
            "t212_monthly": self.t212_monthly_entry.get().strip(),
            "t212_rate_str": self.t212_rate_entry.get().strip(),
            "t212_schedule_json": self.t212_schedule.to_jsonable(),

            "cisa_enabled": bool(self.cisa_enabled.get()),
            "cisa_initial": self.cisa_initial_entry.get().strip(),
            "cisa_monthly": self.cisa_monthly_entry.get().strip(),
            "cisa_rate_str": self.cisa_rate_entry.get().strip(),
            "cisa_schedule_json": self.cisa_schedule.to_jsonable(),
        }

    def _save_settings(self) -> None:
        try:
            safe_save_settings(self._collect_settings())
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    app = GrowthApp()
    app.mainloop()
