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
# Scrollable frame (Canvas + vertical scrollbar) ✅ FIXED
# - Fixed viewport height so content can overflow and scroll
# - Scroll wheel bound locally (no bind_all / no global unbind)
# ============================================================
class ScrollableFrame(ttk.Frame):
    """
    A ttk.Frame with a vertical scrollbar and a fixed viewport.
    Put widgets into `.inner`.
    """
    def __init__(self, master, *, width: int = 420, height: int = 650, **kwargs):
        super().__init__(master, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0, width=width, height=height)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Local mousewheel scrolling (only when pointer is over the canvas)
        self.canvas.bind("<Enter>", self._bind_wheel_local)
        self.canvas.bind("<Leave>", self._unbind_wheel_local)

    def _on_inner_configure(self, _event=None):
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

    def _on_canvas_configure(self, event):
        # Keep inner frame width synced to visible viewport width
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_wheel_local(self, _event=None):
        # Windows / macOS
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        # Linux
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)

    def _unbind_wheel_local(self, _event=None):
        self.canvas.unbind("<MouseWheel>")
        self.canvas.unbind("<Button-4>")
        self.canvas.unbind("<Button-5>")

    def _on_mousewheel(self, event):
        # event.delta is typically ±120 on Windows
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# =============================
# Parsing / validation helpers
# =============================
def parse_float_money(entry: ttk.Entry, field_name: str) -> float:
    s = entry.get().strip().replace("£", "").replace(",", "")
    if not s:
        raise ValueError(f"{field_name} is required.")
    try:
        v = float(s)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")
    if v < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return v


def parse_rate(entry: ttk.Entry, field_name: str) -> float:
    """
    Accepts: 0.08, 8, 8%, 8.0%
    Returns decimal annual rate (e.g. 0.08)
    """
    s = entry.get().strip().replace(" ", "")
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

    # If user typed "8" assume 8%
    if v > 1.0:
        return v / 100.0
    return v


def parse_int(entry: ttk.Entry, field_name: str) -> int:
    s = entry.get().strip()
    if not s:
        raise ValueError(f"{field_name} is required.")
    try:
        v = int(s)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer.")
    return v


def parse_date_iso(entry: ttk.Entry, field_name: str) -> date:
    s = entry.get().strip()
    if not s:
        return date.today()
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (e.g. 2026-02-10).")


def parse_dob(entry: ttk.Entry, field_name: str) -> date:
    """
    Accepts:
      - 03 January 1990
      - 03 Jan 1990
      - 1990-01-03
    """
    s = entry.get().strip()
    if not s:
        raise ValueError(f"{field_name} is required.")

    # ISO
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        pass

    # UK month name
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    raise ValueError(f"{field_name} must be like '03 January 1990' (or 1990-01-03).")


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


# =============================
# Deterministic (daily) series
# =============================
def build_account_monthly_series_daily(
    initial_balance: float,
    monthly_contribution: float,
    annual_rate: float,
    total_months: int,
    start_date: date,
):
    if total_months <= 0:
        raise ValueError("Duration must be at least 1 month.")

    daily_rate = annual_rate / 365.0
    bal = initial_balance
    invested = initial_balance

    cur = date(start_date.year, start_date.month, 1)

    labels = ["Start"]
    with_interest = [round(bal, 2)]
    invested_series = [round(invested, 2)]

    for _ in range(total_months):
        bal += monthly_contribution
        invested += monthly_contribution

        dim = days_in_month(cur)
        for _day in range(dim):
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
# Monte Carlo engine (monthly, lognormal)
# =============================
def monte_carlo_paths_lognormal_monthly(
    *,
    initial_balance: float,
    monthly_contribution: float,
    annual_return: float,
    annual_vol: float,
    months: int,
    n_sims: int,
    seed: int | None,
) -> np.ndarray:
    if months <= 0:
        raise ValueError("Months must be at least 1.")
    if n_sims <= 0:
        raise ValueError("Simulations must be at least 1.")
    if annual_vol < 0:
        raise ValueError("Volatility cannot be negative.")

    rng = np.random.default_rng(seed)

    sigma_m = annual_vol / np.sqrt(12.0)
    drift_m = (annual_return - 0.5 * (annual_vol ** 2)) / 12.0

    z = rng.standard_normal(size=(n_sims, months))
    growth_factors = np.exp(drift_m + sigma_m * z)

    paths = np.empty((n_sims, months + 1), dtype=np.float64)
    paths[:, 0] = initial_balance

    bal = paths[:, 0].copy()
    for t in range(months):
        bal = bal + monthly_contribution
        bal = bal * growth_factors[:, t]
        paths[:, t + 1] = bal

    return paths


# =============================
# Monte Carlo window (LEFT is scrollable ✅)
# =============================
class MonteCarloWindow(tk.Toplevel):
    def __init__(self, master, params: dict):
        super().__init__(master)
        self.title("Monte Carlo Simulation")
        self.geometry("1180x690")
        self.minsize(1180, 690)

        self.params = params

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        # LEFT: scrollable parameters column
        left_scroll = ScrollableFrame(container, width=420, height=650)
        left_scroll.pack(side="left", fill="y")
        left = left_scroll.inner

        # RIGHT: chart
        right = ttk.Frame(container)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))

        # ---- Controls ----
        ctrl = ttk.LabelFrame(left, text="Simulation controls", padding=10)
        ctrl.pack(fill="x")

        self.use_retirement = tk.BooleanVar(value=params["use_retirement"])
        ttk.Label(ctrl, text="Horizon mode:").grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            ctrl,
            text="Use retirement horizon",
            variable=self.use_retirement,
            command=self._toggle_horizon_mode
        ).grid(row=0, column=1, sticky="e")

        ttk.Label(ctrl, text="Date of birth:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.dob_entry = ttk.Entry(ctrl, width=18)
        self.dob_entry.grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.dob_entry.insert(0, params["dob_str"])

        ttk.Label(ctrl, text="Retirement age:").grid(row=2, column=0, sticky="w")
        self.ret_age_entry = ttk.Entry(ctrl, width=18)
        self.ret_age_entry.grid(row=2, column=1, sticky="e")
        self.ret_age_entry.insert(0, str(params["ret_age"]))

        ttk.Separator(ctrl, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(ctrl, text="Years:").grid(row=4, column=0, sticky="w")
        self.years_entry = ttk.Entry(ctrl, width=18)
        self.years_entry.grid(row=4, column=1, sticky="e")
        self.years_entry.insert(0, str(params["years"]))

        ttk.Label(ctrl, text="Extra months:").grid(row=5, column=0, sticky="w")
        self.extra_months_entry = ttk.Entry(ctrl, width=18)
        self.extra_months_entry.grid(row=5, column=1, sticky="e")
        self.extra_months_entry.insert(0, str(params["extra_months"]))

        ttk.Label(ctrl, text="Start date (YYYY-MM-DD):").grid(row=6, column=0, sticky="w")
        self.start_entry = ttk.Entry(ctrl, width=18)
        self.start_entry.grid(row=6, column=1, sticky="e")
        self.start_entry.insert(0, params["start_str"])

        ttk.Separator(ctrl, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(ctrl, text="Simulations:").grid(row=8, column=0, sticky="w")
        self.n_sims = ttk.Entry(ctrl, width=18)
        self.n_sims.grid(row=8, column=1, sticky="e")
        self.n_sims.insert(0, "500")

        ttk.Label(ctrl, text="Seed (optional):").grid(row=9, column=0, sticky="w")
        self.seed_entry = ttk.Entry(ctrl, width=18)
        self.seed_entry.grid(row=9, column=1, sticky="e")
        self.seed_entry.insert(0, "")

        # ---- Include accounts ----
        inc = ttk.LabelFrame(left, text="Include accounts", padding=10)
        inc.pack(fill="x", pady=(10, 0))

        self.ssisa_on = tk.BooleanVar(value=params["ssisa_on"])
        self.cisa_on = tk.BooleanVar(value=params["cisa_on"])
        self.cash_on = tk.BooleanVar(value=params["cash_on"])

        ttk.Checkbutton(inc, text="Include Stocks & Shares ISA", variable=self.ssisa_on).pack(anchor="w")
        ttk.Checkbutton(inc, text="Include Cash ISA", variable=self.cisa_on).pack(anchor="w")
        ttk.Checkbutton(inc, text="Include Cash (non-ISA)", variable=self.cash_on).pack(anchor="w")

        # ---- Stocks & Shares ISA ----
        ssisa = ttk.LabelFrame(left, text="Stocks & Shares ISA (stochastic)", padding=10)
        ssisa.pack(fill="x", pady=(10, 0))

        self.ssisa_initial = self._mini_row(ssisa, 0, "Initial (£):", str(params["ssisa_initial"]))
        self.ssisa_monthly = self._mini_row(ssisa, 1, "Monthly (£):", str(params["ssisa_monthly"]))
        self.ssisa_return = self._mini_row(ssisa, 2, "Return / year:", params["ssisa_rate_str"])
        self.ssisa_vol = self._mini_row(ssisa, 3, "Volatility / year:", "15%")

        # ---- Cash ISA ----
        cisa = ttk.LabelFrame(left, text="Cash ISA (usually low vol)", padding=10)
        cisa.pack(fill="x", pady=(10, 0))

        self.cisa_initial = self._mini_row(cisa, 0, "Initial (£):", str(params["cisa_initial"]))
        self.cisa_monthly = self._mini_row(cisa, 1, "Monthly (£):", str(params["cisa_monthly"]))
        self.cisa_return = self._mini_row(cisa, 2, "Return / year:", params["cisa_rate_str"])
        self.cisa_vol = self._mini_row(cisa, 3, "Volatility / year:", "0%")

        # ---- Cash (non-ISA) ----
        cash = ttk.LabelFrame(left, text="Cash (non-ISA)", padding=10)
        cash.pack(fill="x", pady=(10, 0))

        self.cash_initial = self._mini_row(cash, 0, "Initial (£):", str(params["cash_initial"]))
        self.cash_monthly = self._mini_row(cash, 1, "Monthly (£):", str(params["cash_monthly"]))
        self.cash_return = self._mini_row(cash, 2, "Return / year:", params["cash_rate_str"])
        self.cash_vol = self._mini_row(cash, 3, "Volatility / year:", "0%")

        # Run button
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Run Monte Carlo", command=self.run).pack(side="left")

        # ---- Outputs ----
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

        # ---- Chart ----
        chart_box = ttk.LabelFrame(right, text="Monte Carlo fan chart (Total pot)", padding=10)
        chart_box.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8.6, 5.4), dpi=100)
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
        e = ttk.Entry(parent, width=18)
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

    def _compute_months(self) -> tuple[int, date]:
        if self.use_retirement.get():
            dob = parse_dob(self.dob_entry, "Date of birth")
            ret_age = parse_int(self.ret_age_entry, "Retirement age")
            start = date.today()
            retirement_date = safe_date_years_later(dob, ret_age)
            m = months_between(start, retirement_date)
            if m <= 0:
                raise ValueError("Retirement date is this month or in the past.")
            return m, start

        years = int(self.years_entry.get().strip())
        extra_m = int(self.extra_months_entry.get().strip())
        start = parse_date_iso(self.start_entry, "Start date")
        m = years * 12 + extra_m
        if m <= 0:
            raise ValueError("Horizon must be at least 1 month.")
        return m, start

    def run(self):
        try:
            months, _start_date = self._compute_months()

            n_sims = int(self.n_sims.get().strip())
            if n_sims <= 0:
                raise ValueError("Simulations must be at least 1.")

            seed_txt = self.seed_entry.get().strip()
            seed = int(seed_txt) if seed_txt else None

            ssisa_on = bool(self.ssisa_on.get())
            cisa_on = bool(self.cisa_on.get())
            cash_on = bool(self.cash_on.get())

            n = n_sims
            m = months

            if ssisa_on:
                ssisa_paths = monte_carlo_paths_lognormal_monthly(
                    initial_balance=float(self.ssisa_initial.get().strip().replace(",", "")),
                    monthly_contribution=float(self.ssisa_monthly.get().strip().replace(",", "")),
                    annual_return=parse_rate(self.ssisa_return, "S&S ISA return"),
                    annual_vol=parse_rate(self.ssisa_vol, "S&S ISA volatility"),
                    months=m,
                    n_sims=n,
                    seed=seed,
                )
            else:
                ssisa_paths = np.zeros((n, m + 1), dtype=np.float64)

            if cisa_on:
                cisa_paths = monte_carlo_paths_lognormal_monthly(
                    initial_balance=float(self.cisa_initial.get().strip().replace(",", "")),
                    monthly_contribution=float(self.cisa_monthly.get().strip().replace(",", "")),
                    annual_return=parse_rate(self.cisa_return, "Cash ISA return"),
                    annual_vol=parse_rate(self.cisa_vol, "Cash ISA volatility"),
                    months=m,
                    n_sims=n,
                    seed=None if seed is None else seed + 11,
                )
            else:
                cisa_paths = np.zeros((n, m + 1), dtype=np.float64)

            if cash_on:
                cash_paths = monte_carlo_paths_lognormal_monthly(
                    initial_balance=float(self.cash_initial.get().strip().replace(",", "")),
                    monthly_contribution=float(self.cash_monthly.get().strip().replace(",", "")),
                    annual_return=parse_rate(self.cash_return, "Cash return"),
                    annual_vol=parse_rate(self.cash_vol, "Cash volatility"),
                    months=m,
                    n_sims=n,
                    seed=None if seed is None else seed + 23,
                )
            else:
                cash_paths = np.zeros((n, m + 1), dtype=np.float64)

            total_paths = ssisa_paths + cisa_paths + cash_paths

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

            x = np.arange(m + 1)
            self.ax.fill_between(x, p10, p90, alpha=0.25, label="10–90% band")
            self.ax.plot(x, p50, label="Median (50%)")
            self.ax.legend()

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Monte Carlo error", str(e))


# =============================
# Main deterministic app
# =============================
class GrowthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Money Growth Forecaster (S&S ISA + Cash ISA + Cash)")
        self.geometry("1280x730")
        self.minsize(1280, 730)

        self.style = ttk.Style()
        for theme in ("vista", "clam", "alt", "default"):
            try:
                self.style.theme_use(theme)
                break
            except tk.TclError:
                pass

        main = ttk.Frame(self, padding=14)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y")

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(14, 0))

        # Retirement settings
        retirement = ttk.LabelFrame(left, text="Retirement settings (optional)", padding=12)
        retirement.pack(fill="x")

        self.use_retirement = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            retirement,
            text="Use retirement horizon (overrides Years/Months/Start Date)",
            variable=self.use_retirement,
            command=self._toggle_retirement_mode,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(retirement, text="Date of birth:").grid(row=1, column=0, sticky="w", pady=4)
        self.dob_entry = ttk.Entry(retirement, width=24)
        self.dob_entry.grid(row=1, column=1, sticky="e", pady=4)
        self.dob_entry.insert(0, "03 January 1990")

        ttk.Label(retirement, text="Retirement age:").grid(row=2, column=0, sticky="w", pady=4)
        self.ret_age_entry = ttk.Entry(retirement, width=24)
        self.ret_age_entry.grid(row=2, column=1, sticky="e", pady=4)
        self.ret_age_entry.insert(0, "60")

        self.months_to_ret_var = tk.StringVar(value="—")
        ttk.Label(retirement, text="Months to retirement (from today):").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(retirement, textvariable=self.months_to_ret_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=1, sticky="e", pady=4)

        # Manual horizon
        horizon = ttk.LabelFrame(left, text="Manual horizon", padding=12)
        horizon.pack(fill="x", pady=(10, 0))

        self.years_entry = self._row(horizon, 0, "Years:", "19")
        self.months_entry = self._row(horizon, 1, "Extra months:", "0")
        self.start_entry = self._row(horizon, 2, "Start date (YYYY-MM-DD):", "")

        # Stocks & Shares ISA
        ssisa = ttk.LabelFrame(left, text="Stocks & Shares ISA", padding=12)
        ssisa.pack(fill="x", pady=(10, 0))

        self.ssisa_enabled = tk.BooleanVar(value=True)
        row0 = ttk.Frame(ssisa)
        row0.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(row0, text="Include Stocks & Shares ISA", variable=self.ssisa_enabled, command=self._toggle_ssisa).pack(anchor="w")

        self.ssisa_initial_entry = self._row(ssisa, 1, "Initial balance (£):", "3500")
        self.ssisa_monthly_entry = self._row(ssisa, 2, "Monthly contribution (£):", "200")
        self.ssisa_rate_entry = self._row(ssisa, 3, "Annual return (e.g. 7%):", "7%")

        # Cash ISA
        cisa = ttk.LabelFrame(left, text="Cash ISA", padding=12)
        cisa.pack(fill="x", pady=(10, 0))

        self.cisa_enabled = tk.BooleanVar(value=True)
        row0 = ttk.Frame(cisa)
        row0.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(row0, text="Include Cash ISA", variable=self.cisa_enabled, command=self._toggle_cisa).pack(anchor="w")

        self.cisa_initial_entry = self._row(cisa, 1, "Initial balance (£):", "0")
        self.cisa_monthly_entry = self._row(cisa, 2, "Monthly contribution (£):", "0")
        self.cisa_rate_entry = self._row(cisa, 3, "Annual rate (e.g. 4.5%):", "4.5%")

        # Cash (non-ISA)
        cash = ttk.LabelFrame(left, text="Cash (non-ISA)", padding=12)
        cash.pack(fill="x", pady=(10, 0))

        self.cash_enabled = tk.BooleanVar(value=True)
        row0 = ttk.Frame(cash)
        row0.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Checkbutton(row0, text="Include Cash (non-ISA)", variable=self.cash_enabled, command=self._toggle_cash).pack(anchor="w")

        self.cash_initial_entry = self._row(cash, 1, "Initial cash (£):", "0")
        self.cash_monthly_entry = self._row(cash, 2, "Monthly cash add (£):", "0")
        self.cash_rate_entry = self._row(cash, 3, "Cash annual rate (e.g. 4.5%):", "4.5%")

        # Buttons
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Calculate", command=self.on_calculate).pack(side="left")
        ttk.Button(btns, text="Reset", command=self.on_reset).pack(side="left", padx=8)
        ttk.Button(btns, text="Monte Carlo…", command=self.open_monte_carlo).pack(side="left", padx=8)

        # Results
        outputs = ttk.LabelFrame(left, text="End of horizon", padding=12)
        outputs.pack(fill="x", pady=(10, 0))

        self.total_with_var = tk.StringVar(value="—")
        self.total_invested_var = tk.StringVar(value="—")
        self.total_growth_var = tk.StringVar(value="—")

        self.ssisa_with_var = tk.StringVar(value="—")
        self.ssisa_inv_var = tk.StringVar(value="—")
        self.ssisa_growth_var = tk.StringVar(value="—")

        self.cisa_with_var = tk.StringVar(value="—")
        self.cisa_inv_var = tk.StringVar(value="—")
        self.cisa_growth_var = tk.StringVar(value="—")

        self.cash_with_var = tk.StringVar(value="—")
        self.cash_inv_var = tk.StringVar(value="—")
        self.cash_growth_var = tk.StringVar(value="—")

        self._out_row(outputs, 0, "Total pot (with interest):", self.total_with_var)
        self._out_row(outputs, 1, "Total invested:", self.total_invested_var)
        self._out_row(outputs, 2, "Total growth:", self.total_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 4, "S&S ISA amount:", self.ssisa_with_var)
        self._out_row(outputs, 5, "S&S ISA invested:", self.ssisa_inv_var)
        self._out_row(outputs, 6, "S&S ISA growth:", self.ssisa_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=7, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 8, "Cash ISA amount:", self.cisa_with_var)
        self._out_row(outputs, 9, "Cash ISA invested:", self.cisa_inv_var)
        self._out_row(outputs, 10, "Cash ISA growth:", self.cisa_growth_var)

        ttk.Separator(outputs, orient="horizontal").grid(row=11, column=0, columnspan=2, sticky="ew", pady=8)

        self._out_row(outputs, 12, "Cash (non-ISA) amount:", self.cash_with_var)
        self._out_row(outputs, 13, "Cash (non-ISA) invested:", self.cash_inv_var)
        self._out_row(outputs, 14, "Cash (non-ISA) growth:", self.cash_growth_var)

        # Chart
        chart_box = ttk.LabelFrame(right, text="Deterministic projection (daily compounding)", padding=10)
        chart_box.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(8.7, 5.6), dpi=100)
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
            ssisa_with=[0],
            cisa_with=[0],
            cash_with=[0],
            ssisa_on=True,
            cisa_on=True,
            cash_on=True,
        )

        self._toggle_retirement_mode()
        self._toggle_ssisa()
        self._toggle_cisa()
        self._toggle_cash()

    def _row(self, parent, r, label, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        e = ttk.Entry(parent, width=24)
        e.grid(row=r, column=1, sticky="e", pady=4)
        e.insert(0, default)
        return e

    def _out_row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=4)
        ttk.Label(parent, textvariable=var, font=("Segoe UI", 10, "bold")).grid(row=r, column=1, sticky="e", pady=4)

    def _toggle_retirement_mode(self):
        manual_state = "disabled" if self.use_retirement.get() else "normal"
        for w in (self.years_entry, self.months_entry, self.start_entry):
            w.configure(state=manual_state)

        ret_state = "normal" if self.use_retirement.get() else "disabled"
        for w in (self.dob_entry, self.ret_age_entry):
            w.configure(state=ret_state)

        try:
            dob = parse_dob(self.dob_entry, "Date of birth")
            ret_age = parse_int(self.ret_age_entry, "Retirement age")
            retirement_date = safe_date_years_later(dob, ret_age)
            m = months_between(date.today(), retirement_date)
            self.months_to_ret_var.set(str(m) if m > 0 else "0")
        except Exception:
            self.months_to_ret_var.set("—")

    def _toggle_ssisa(self):
        state = "normal" if self.ssisa_enabled.get() else "disabled"
        for w in (self.ssisa_initial_entry, self.ssisa_monthly_entry, self.ssisa_rate_entry):
            w.configure(state=state)

    def _toggle_cisa(self):
        state = "normal" if self.cisa_enabled.get() else "disabled"
        for w in (self.cisa_initial_entry, self.cisa_monthly_entry, self.cisa_rate_entry):
            w.configure(state=state)

    def _toggle_cash(self):
        state = "normal" if self.cash_enabled.get() else "disabled"
        for w in (self.cash_initial_entry, self.cash_monthly_entry, self.cash_rate_entry):
            w.configure(state=state)

    def _get_horizon(self) -> tuple[int, date, list[str]]:
        if self.use_retirement.get():
            dob = parse_dob(self.dob_entry, "Date of birth")
            ret_age = parse_int(self.ret_age_entry, "Retirement age")
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

    def _plot(
        self,
        *,
        labels,
        total_with,
        total_invested,
        ssisa_with,
        cisa_with,
        cash_with,
        ssisa_on: bool,
        cisa_on: bool,
        cash_on: bool,
    ):
        self.ax.clear()

        self.ax.plot(total_with, label="Total pot (with interest)")
        self.ax.plot(total_invested, label="Total invested")

        if ssisa_on:
            self.ax.plot(ssisa_with, label="S&S ISA (with interest)")
        if cisa_on:
            self.ax.plot(cisa_with, label="Cash ISA (with interest)")
        if cash_on:
            self.ax.plot(cash_with, label="Cash (non-ISA) (with interest)")

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

    def on_calculate(self):
        try:
            total_months, start, labels = self._get_horizon()

            # S&S ISA
            ssisa_on = self.ssisa_enabled.get()
            if ssisa_on:
                ssisa_initial = parse_float_money(self.ssisa_initial_entry, "S&S ISA initial balance")
                ssisa_monthly = parse_float_money(self.ssisa_monthly_entry, "S&S ISA monthly contribution")
                ssisa_rate = parse_rate(self.ssisa_rate_entry, "S&S ISA annual return")
                labels_a, ssisa_with, ssisa_inv = build_account_monthly_series_daily(
                    initial_balance=ssisa_initial,
                    monthly_contribution=ssisa_monthly,
                    annual_rate=ssisa_rate,
                    total_months=total_months,
                    start_date=start,
                )
                if labels_a != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                ssisa_with = [0.0] * len(labels)
                ssisa_inv = [0.0] * len(labels)

            # Cash ISA
            cisa_on = self.cisa_enabled.get()
            if cisa_on:
                cisa_initial = parse_float_money(self.cisa_initial_entry, "Cash ISA initial balance")
                cisa_monthly = parse_float_money(self.cisa_monthly_entry, "Cash ISA monthly contribution")
                cisa_rate = parse_rate(self.cisa_rate_entry, "Cash ISA annual rate")
                labels_b, cisa_with, cisa_inv = build_account_monthly_series_daily(
                    initial_balance=cisa_initial,
                    monthly_contribution=cisa_monthly,
                    annual_rate=cisa_rate,
                    total_months=total_months,
                    start_date=start,
                )
                if labels_b != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                cisa_with = [0.0] * len(labels)
                cisa_inv = [0.0] * len(labels)

            # Cash (non-ISA)
            cash_on = self.cash_enabled.get()
            if cash_on:
                cash_initial = parse_float_money(self.cash_initial_entry, "Cash initial")
                cash_monthly = parse_float_money(self.cash_monthly_entry, "Cash monthly add")
                cash_rate = parse_rate(self.cash_rate_entry, "Cash annual rate")
                labels_c, cash_with, cash_inv = build_account_monthly_series_daily(
                    initial_balance=cash_initial,
                    monthly_contribution=cash_monthly,
                    annual_rate=cash_rate,
                    total_months=total_months,
                    start_date=start,
                )
                if labels_c != labels:
                    raise ValueError("Date labels mismatch (internal error).")
            else:
                cash_with = [0.0] * len(labels)
                cash_inv = [0.0] * len(labels)

            total_with = combine_series(ssisa_with, cisa_with, cash_with)
            total_inv = combine_series(ssisa_inv, cisa_inv, cash_inv)

            def end_triplet(with_series, inv_series):
                end_with = with_series[-1]
                end_inv = inv_series[-1]
                end_growth = round(end_with - end_inv, 2)
                return end_with, end_inv, end_growth

            end_total_with, end_total_inv, end_total_growth = end_triplet(total_with, total_inv)
            end_ssisa_with, end_ssisa_inv, end_ssisa_growth = end_triplet(ssisa_with, ssisa_inv)
            end_cisa_with, end_cisa_inv, end_cisa_growth = end_triplet(cisa_with, cisa_inv)
            end_cash_with, end_cash_inv, end_cash_growth = end_triplet(cash_with, cash_inv)

            self.total_with_var.set(f"£{end_total_with:,.2f}")
            self.total_invested_var.set(f"£{end_total_inv:,.2f}")
            self.total_growth_var.set(f"£{end_total_growth:,.2f}")

            self.ssisa_with_var.set(f"£{end_ssisa_with:,.2f}" if ssisa_on else "—")
            self.ssisa_inv_var.set(f"£{end_ssisa_inv:,.2f}" if ssisa_on else "—")
            self.ssisa_growth_var.set(f"£{end_ssisa_growth:,.2f}" if ssisa_on else "—")

            self.cisa_with_var.set(f"£{end_cisa_with:,.2f}" if cisa_on else "—")
            self.cisa_inv_var.set(f"£{end_cisa_inv:,.2f}" if cisa_on else "—")
            self.cisa_growth_var.set(f"£{end_cisa_growth:,.2f}" if cisa_on else "—")

            self.cash_with_var.set(f"£{end_cash_with:,.2f}" if cash_on else "—")
            self.cash_inv_var.set(f"£{end_cash_inv:,.2f}" if cash_on else "—")
            self.cash_growth_var.set(f"£{end_cash_growth:,.2f}" if cash_on else "—")

            self._plot(
                labels=labels,
                total_with=total_with,
                total_invested=total_inv,
                ssisa_with=ssisa_with,
                cisa_with=cisa_with,
                cash_with=cash_with,
                ssisa_on=ssisa_on,
                cisa_on=cisa_on,
                cash_on=cash_on,
            )

        except Exception as e:
            messagebox.showerror("Input error", str(e))

    def on_reset(self):
        self.use_retirement.set(False)
        self.dob_entry.delete(0, tk.END); self.dob_entry.insert(0, "03 January 1990")
        self.ret_age_entry.delete(0, tk.END); self.ret_age_entry.insert(0, "60")

        self.years_entry.delete(0, tk.END); self.years_entry.insert(0, "19")
        self.months_entry.delete(0, tk.END); self.months_entry.insert(0, "0")
        self.start_entry.delete(0, tk.END)

        self.ssisa_enabled.set(True)
        self.ssisa_initial_entry.delete(0, tk.END); self.ssisa_initial_entry.insert(0, "3500")
        self.ssisa_monthly_entry.delete(0, tk.END); self.ssisa_monthly_entry.insert(0, "200")
        self.ssisa_rate_entry.delete(0, tk.END); self.ssisa_rate_entry.insert(0, "7%")

        self.cisa_enabled.set(True)
        self.cisa_initial_entry.delete(0, tk.END); self.cisa_initial_entry.insert(0, "0")
        self.cisa_monthly_entry.delete(0, tk.END); self.cisa_monthly_entry.insert(0, "0")
        self.cisa_rate_entry.delete(0, tk.END); self.cisa_rate_entry.insert(0, "4.5%")

        self.cash_enabled.set(True)
        self.cash_initial_entry.delete(0, tk.END); self.cash_initial_entry.insert(0, "0")
        self.cash_monthly_entry.delete(0, tk.END); self.cash_monthly_entry.insert(0, "0")
        self.cash_rate_entry.delete(0, tk.END); self.cash_rate_entry.insert(0, "4.5%")

        self._toggle_retirement_mode()
        self._toggle_ssisa()
        self._toggle_cisa()
        self._toggle_cash()

        for var in (
            self.total_with_var, self.total_invested_var, self.total_growth_var,
            self.ssisa_with_var, self.ssisa_inv_var, self.ssisa_growth_var,
            self.cisa_with_var, self.cisa_inv_var, self.cisa_growth_var,
            self.cash_with_var, self.cash_inv_var, self.cash_growth_var,
        ):
            var.set("—")

        self._plot(
            labels=["Start"],
            total_with=[0],
            total_invested=[0],
            ssisa_with=[0],
            cisa_with=[0],
            cash_with=[0],
            ssisa_on=True,
            cisa_on=True,
            cash_on=True,
        )

    def open_monte_carlo(self):
        try:
            params = self._snapshot_params_for_mc()
            MonteCarloWindow(self, params)
        except Exception as e:
            messagebox.showerror("Monte Carlo setup error", str(e))

    def _snapshot_params_for_mc(self) -> dict:
        start_str = self.start_entry.get().strip() or date.today().isoformat()

        _ = parse_date_iso(self.start_entry, "Start date")
        _ = parse_dob(self.dob_entry, "Date of birth")
        _ = parse_int(self.ret_age_entry, "Retirement age")

        _ = parse_rate(self.ssisa_rate_entry, "S&S ISA annual return")
        _ = parse_rate(self.cisa_rate_entry, "Cash ISA annual rate")
        _ = parse_rate(self.cash_rate_entry, "Cash annual rate")

        return {
            "use_retirement": bool(self.use_retirement.get()),
            "dob_str": self.dob_entry.get().strip() or "03 January 1990",
            "ret_age": int(self.ret_age_entry.get().strip() or "60"),
            "years": int(self.years_entry.get().strip() or "0"),
            "extra_months": int(self.months_entry.get().strip() or "0"),
            "start_str": start_str,

            "ssisa_on": bool(self.ssisa_enabled.get()),
            "cisa_on": bool(self.cisa_enabled.get()),
            "cash_on": bool(self.cash_enabled.get()),

            "ssisa_initial": parse_float_money(self.ssisa_initial_entry, "S&S ISA initial balance"),
            "ssisa_monthly": parse_float_money(self.ssisa_monthly_entry, "S&S ISA monthly contribution"),
            "ssisa_rate_str": self.ssisa_rate_entry.get().strip(),

            "cisa_initial": parse_float_money(self.cisa_initial_entry, "Cash ISA initial balance"),
            "cisa_monthly": parse_float_money(self.cisa_monthly_entry, "Cash ISA monthly contribution"),
            "cisa_rate_str": self.cisa_rate_entry.get().strip(),

            "cash_initial": parse_float_money(self.cash_initial_entry, "Cash initial"),
            "cash_monthly": parse_float_money(self.cash_monthly_entry, "Cash monthly add"),
            "cash_rate_str": self.cash_rate_entry.get().strip(),
        }


if __name__ == "__main__":
    app = GrowthApp()
    app.mainloop()
