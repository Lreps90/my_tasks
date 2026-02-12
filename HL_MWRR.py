import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from pathlib import Path
import json

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates


# ============================
# Files
# ============================
DATA_FILE = Path(__file__).with_name("mwrr_cashflows.json")

BACKUP_DIR = Path(__file__).with_name("mwrr_backups")
BACKUP_DIR.mkdir(exist_ok=True)


# ============================
# Finance maths (XIRR / MWRR)
# ============================
def _year_frac(d0: datetime, d1: datetime) -> float:
    return (d1 - d0).days / 365.0

def xnpv(rate: float, dates, cfs) -> float:
    d0 = dates[0]
    total = 0.0
    for d, cf in zip(dates, cfs):
        t = _year_frac(d0, d)
        total += cf / ((1.0 + rate) ** t)
    return total

def xnpv_derivative(rate: float, dates, cfs) -> float:
    d0 = dates[0]
    total = 0.0
    for d, cf in zip(dates, cfs):
        t = _year_frac(d0, d)
        total += -t * cf / ((1.0 + rate) ** (t + 1.0))
    return total

def xirr(dates, cfs, guess=0.1) -> float:
    """
    Annualised money-weighted return (XIRR).
    """
    if len(dates) != len(cfs) or len(dates) < 2:
        raise ValueError("Need at least two cash flows.")
    has_pos = any(cf > 0 for cf in cfs)
    has_neg = any(cf < 0 for cf in cfs)
    if not (has_pos and has_neg):
        raise ValueError("Need at least one positive and one negative cash flow.")

    low = -0.9999
    high = 10.0

    f_low = xnpv(low, dates, cfs)
    f_high = xnpv(high, dates, cfs)

    expand_tries = 0
    while f_low * f_high > 0 and expand_tries < 30:
        high *= 1.5
        f_high = xnpv(high, dates, cfs)
        expand_tries += 1

    if f_low * f_high > 0:
        r = guess
        for _ in range(120):
            f = xnpv(r, dates, cfs)
            df = xnpv_derivative(r, dates, cfs)
            if abs(df) < 1e-12:
                break
            step = f / df
            r -= step
            if r <= -0.9999:
                r = -0.9999 + 1e-6
            if abs(step) < 1e-10:
                return r
        raise ValueError("Could not find a stable XIRR solution (check cash flows).")

    a, b = low, high
    fa, fb = f_low, f_high
    for _ in range(90):
        m = (a + b) / 2.0
        fm = xnpv(m, dates, cfs)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) < 1e-12:
            break

    r = (a + b) / 2.0

    for _ in range(50):
        f = xnpv(r, dates, cfs)
        df = xnpv_derivative(r, dates, cfs)
        if abs(df) < 1e-12:
            break
        step = f / df
        r2 = r - step
        if r2 <= low or r2 >= high:
            break
        r = r2
        if abs(step) < 1e-12:
            break

    return r


# ============================
# Parsing / formatting
# ============================
def parse_date(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%d-%b-%y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError("Date not recognised. Use e.g. 12-Feb-26 or 2026-02-12.")

def parse_money(s: str) -> float:
    s = s.strip().replace("£", "").replace(",", "")
    if not s:
        raise ValueError("Amount is required.")
    return float(s)

def fmt_money(x: float) -> str:
    return f"£{x:,.2f}"

def normalise_display_date(d: datetime) -> str:
    return d.strftime("%d-%b-%y").lstrip("0")


# ============================
# App
# ============================
class MWRRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MWRR (XIRR) Tracker")
        self.geometry("1200x700")

        self._edit_widget = None
        self.history = []  # {"asat": "12-Feb-26", "balance": "3622.00", "mwrr": 0.1088}

        # ---------------- Top row ----------------
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="As at date:").grid(row=0, column=0, sticky="w")
        self.asat_var = tk.StringVar(value="31-Jan-26")
        ttk.Entry(top, textvariable=self.asat_var, width=14).grid(row=0, column=1, padx=(6, 16), sticky="w")

        ttk.Label(top, text="Current balance (£):").grid(row=0, column=2, sticky="w")
        self.balance_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.balance_var, width=16).grid(row=0, column=3, padx=(6, 16), sticky="w")

        ttk.Button(top, text="Calculate + Save Snapshot", command=self.on_calculate).grid(row=0, column=4, padx=(6, 10))

        self.mwrr_var = tk.StringVar(value="MWRR: —")
        ttk.Label(top, textvariable=self.mwrr_var, font=("Segoe UI", 11, "bold")).grid(row=0, column=5, padx=(6, 0), sticky="w")

        # ---------------- Main split ----------------
        main = ttk.Frame(self, padding=(10, 0, 10, 10))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)

        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---------------- Cashflow table (left) ----------------
        table_frame = ttk.LabelFrame(left, text="Cash flows (double-click to edit)", padding=10)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        cols = ("date", "amount")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=14)
        self.tree.heading("date", text="Date")
        self.tree.heading("amount", text="Amount (£)")
        self.tree.column("date", width=160, anchor="w")
        self.tree.column("amount", width=160, anchor="e")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind("<Button-1>", self.on_tree_single_click)

        # Add/delete row controls
        controls = ttk.Frame(left, padding=(0, 10, 0, 0))
        controls.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls, text="New row date:").grid(row=0, column=0, sticky="w")
        self.new_date_var = tk.StringVar(value="12-Feb-26")
        ttk.Entry(controls, textvariable=self.new_date_var, width=14).grid(row=0, column=1, padx=(6, 16), sticky="w")

        ttk.Label(controls, text="New row amount (£):").grid(row=0, column=2, sticky="w")
        self.new_amt_var = tk.StringVar(value="-200.00")
        ttk.Entry(controls, textvariable=self.new_amt_var, width=16).grid(row=0, column=3, padx=(6, 16), sticky="w")

        ttk.Button(controls, text="Add row", command=self.on_add_row).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(controls, text="Delete selected", command=self.on_delete_selected).grid(row=0, column=5)

        tip = ttk.Label(
            left,
            text="Deposits are negative (money in). Withdrawals are positive. The balance is appended automatically as a final positive cash flow on the As at date.",
            foreground="#444",
            wraplength=520
        )
        tip.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        # ---------------- Metrics + history + graphs (right) ----------------
        metrics_frame = ttk.LabelFrame(right, text="Key metrics", padding=10)
        metrics_frame.pack(side=tk.TOP, fill=tk.X)

        self.net_deposits_var = tk.StringVar(value="Net deposits: —")
        self.total_deposits_var = tk.StringVar(value="Total deposits: —")
        self.total_withdrawals_var = tk.StringVar(value="Total withdrawals: —")
        self.profit_var = tk.StringVar(value="Profit (vs cash flows): —")

        ttk.Label(metrics_frame, textvariable=self.net_deposits_var).grid(row=0, column=0, sticky="w")
        ttk.Label(metrics_frame, textvariable=self.total_deposits_var).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(metrics_frame, textvariable=self.total_withdrawals_var).grid(row=2, column=0, sticky="w", pady=(2, 0))
        ttk.Label(metrics_frame, textvariable=self.profit_var, font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="w", pady=(6, 0))

        history_frame = ttk.LabelFrame(right, text="Snapshots (saved history)", padding=10)
        history_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.hist_tree = ttk.Treeview(history_frame, columns=("asat", "balance", "mwrr"), show="headings", height=7)
        self.hist_tree.heading("asat", text="As at")
        self.hist_tree.heading("balance", text="Balance")
        self.hist_tree.heading("mwrr", text="MWRR (p.a.)")
        self.hist_tree.column("asat", width=120, anchor="w")
        self.hist_tree.column("balance", width=120, anchor="e")
        self.hist_tree.column("mwrr", width=120, anchor="e")

        hvsb = ttk.Scrollbar(history_frame, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=hvsb.set)

        self.hist_tree.grid(row=0, column=0, sticky="nsew")
        hvsb.grid(row=0, column=1, sticky="ns")
        history_frame.grid_rowconfigure(0, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)

        hist_buttons = ttk.Frame(history_frame)
        hist_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(hist_buttons, text="Delete snapshot", command=self.on_delete_snapshot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(hist_buttons, text="Clear all snapshots", command=self.on_clear_snapshots).pack(side=tk.LEFT)
        ttk.Button(hist_buttons, text="Restore last backup", command=self.restore_latest_backup).pack(side=tk.LEFT, padx=(8, 0))

        # ---------------- Graphs box with tabs ----------------
        graphs_frame = ttk.LabelFrame(right, text="Graphs", padding=10)
        graphs_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 0))

        self.graph_tabs = ttk.Notebook(graphs_frame)
        self.graph_tabs.pack(fill=tk.BOTH, expand=True)

        # Tab 1: MWRR over time
        tab_mwrr = ttk.Frame(self.graph_tabs)
        self.graph_tabs.add(tab_mwrr, text="MWRR over time")

        self.fig_mwrr = Figure(figsize=(6, 3), dpi=100)
        self.ax_mwrr = self.fig_mwrr.add_subplot(111)
        self.ax_mwrr.set_ylabel("MWRR (p.a.)")
        self.ax_mwrr.set_xlabel("Date")

        self.canvas_mwrr = FigureCanvasTkAgg(self.fig_mwrr, master=tab_mwrr)
        self.canvas_mwrr.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 2: Balance vs Invested baseline
        tab_balance = ttk.Frame(self.graph_tabs)
        self.graph_tabs.add(tab_balance, text="Balance vs invested")

        self.fig_bal = Figure(figsize=(6, 3), dpi=100)
        self.ax_bal = self.fig_bal.add_subplot(111)
        self.ax_bal.set_ylabel("£")
        self.ax_bal.set_xlabel("Date")

        self.canvas_bal = FigureCanvasTkAgg(self.fig_bal, master=tab_balance)
        self.canvas_bal.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Close handler
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load persisted state or preload
        if not self.load_data():
            self.load_preload()
            self.save_data()

        self.refresh_metrics()
        self.refresh_history_view()
        self.refresh_charts()

    # ============================
    # Backups / restore
    # ============================
    def make_backup(self):
        try:
            if not DATA_FILE.exists():
                self.save_data()
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            backup_file = BACKUP_DIR / f"mwrr_cashflows.backup_{stamp}.json"
            backup_file.write_text(DATA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Backup failed", str(e))

    def restore_latest_backup(self):
        backups = sorted(BACKUP_DIR.glob("mwrr_cashflows.backup_*.json"))
        if not backups:
            messagebox.showinfo("No backups", "No backup files found.")
            return

        latest = backups[-1]
        try:
            DATA_FILE.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")
            self.load_data()
            self.refresh_metrics()
            self.refresh_history_view()
            self.refresh_charts()
            messagebox.showinfo("Restored", f"Restored from:\n{latest.name}")
        except Exception as e:
            messagebox.showerror("Restore failed", str(e))

    # ============================
    # Persistence
    # ============================
    def get_cashflow_rows_raw(self):
        rows = []
        for iid in self.tree.get_children():
            d_str, a_str = self.tree.item(iid, "values")
            rows.append((d_str, a_str))
        return rows

    def set_cashflow_rows_raw(self, rows):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        for d_str, a_str in rows:
            self.tree.insert("", "end", values=(d_str, a_str))

    def save_data(self):
        payload = {
            "asat": self.asat_var.get(),
            "balance": self.balance_var.get(),
            "cashflows": self.get_cashflow_rows_raw(),
            "history": self.history,
        }
        try:
            DATA_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save to:\n{DATA_FILE}\n\n{e}")

    def load_data(self) -> bool:
        if not DATA_FILE.exists():
            return False
        try:
            payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            self.asat_var.set(payload.get("asat", "31-Jan-26"))
            self.balance_var.set(payload.get("balance", ""))

            cashflows = payload.get("cashflows", [])
            if cashflows:
                self.set_cashflow_rows_raw(cashflows)

            self.history = payload.get("history", []) or []
            return bool(cashflows)
        except Exception as e:
            messagebox.showerror("Load failed", f"Could not load:\n{DATA_FILE}\n\n{e}")
            return False

    # ============================
    # Defaults
    # ============================
    def load_preload(self):
        preload = [
            ("1-Mar-24", "-200.00"),
            ("1-Apr-24", "-50.00"),
            ("1-May-24", "-146.60"),
            ("1-Jun-24", "-250.00"),
            ("1-Jul-24", "-75.00"),
            ("1-Aug-24", "-126.42"),
            ("1-Sep-24", "-100.00"),
            ("1-Oct-24", "-100.00"),
            ("1-Nov-24", "-100.00"),
            ("1-Dec-24", "-100.00"),
            ("1-Jan-25", "147.04"),
            ("1-Feb-25", "-100.00"),
            ("1-Mar-25", "-100.00"),
            ("1-Apr-25", "-100.00"),
            ("1-May-25", "-200.00"),
            ("1-Jun-25", "-100.00"),
            ("1-Jul-25", "-100.00"),
            ("1-Aug-25", "-200.00"),
            ("1-Sep-25", "-125.00"),
            ("1-Oct-25", "-430.43"),
            ("1-Nov-25", "-175.00"),
            ("1-Dec-25", "-175.00"),
            ("15-Jan-26", "-200.00"),
        ]
        self.set_cashflow_rows_raw(preload)

    # ============================
    # Editing in Treeview
    # ============================
    def on_tree_single_click(self, _event):
        self._commit_edit()

    def on_tree_double_click(self, event):
        self._commit_edit()

        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)
        if not row_id or not col_id:
            return

        col_index = int(col_id.replace("#", "")) - 1
        col_key = ("date", "amount")[col_index]

        x, y, width, height = self.tree.bbox(row_id, col_id)
        value = self.tree.set(row_id, col_key)

        entry = ttk.Entry(self.tree)
        entry.insert(0, value)
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=width, height=height)

        entry.bind("<Return>", lambda _e: self._commit_edit())
        entry.bind("<Escape>", lambda _e: self._cancel_edit())
        entry.bind("<FocusOut>", lambda _e: self._commit_edit())

        self._edit_widget = (entry, row_id, col_key)

    def _cancel_edit(self):
        if self._edit_widget:
            entry, *_ = self._edit_widget
            entry.destroy()
            self._edit_widget = None

    def _commit_edit(self):
        if not self._edit_widget:
            return

        entry, row_id, col_key = self._edit_widget
        new_val = entry.get().strip()
        entry.destroy()
        self._edit_widget = None

        try:
            if col_key == "date":
                d = parse_date(new_val)
                new_val = normalise_display_date(d)
            else:
                a = parse_money(new_val)
                new_val = f"{a:,.2f}"
        except Exception as e:
            messagebox.showerror("Invalid edit", str(e))
            return

        self.tree.set(row_id, col_key, new_val)
        self.save_data()
        self.refresh_metrics()

    # ============================
    # Actions
    # ============================
    def on_add_row(self):
        try:
            d = parse_date(self.new_date_var.get())
            a = parse_money(self.new_amt_var.get())
        except Exception as e:
            messagebox.showerror("Invalid row", str(e))
            return

        self.tree.insert("", "end", values=(normalise_display_date(d), f"{a:,.2f}"))
        self.save_data()
        self.refresh_metrics()

    def on_delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        for iid in sel:
            self.tree.delete(iid)
        self.save_data()
        self.refresh_metrics()

    def on_calculate(self):
        self._commit_edit()

        try:
            asat = parse_date(self.asat_var.get())
            bal = parse_money(self.balance_var.get())
            if bal <= 0:
                raise ValueError("Balance should be a positive number.")
        except Exception as e:
            messagebox.showerror("Invalid inputs", str(e))
            return

        rows = []
        for iid in self.tree.get_children():
            d_str, a_str = self.tree.item(iid, "values")
            try:
                d = parse_date(d_str)
                a = parse_money(a_str)
            except Exception as e:
                messagebox.showerror("Bad cash flow row", f"Row '{d_str} / {a_str}' invalid:\n{e}")
                return
            rows.append((d, a))

        if not rows:
            messagebox.showerror("No data", "Add at least one cash flow row.")
            return

        rows.append((asat, bal))
        rows.sort(key=lambda x: x[0])

        dates = [d for d, _ in rows]
        cfs = [cf for _, cf in rows]

        try:
            r = xirr(dates, cfs, guess=0.1)
        except Exception as e:
            messagebox.showerror("MWRR calculation failed", str(e))
            return

        self.mwrr_var.set(f"MWRR: {r*100:.2f}% p.a.")

        asat_disp = normalise_display_date(asat)
        snapshot = {"asat": asat_disp, "balance": f"{bal:.2f}", "mwrr": r}

        replaced = False
        for i, s in enumerate(self.history):
            if s.get("asat") == asat_disp:
                self.history[i] = snapshot
                replaced = True
                break
        if not replaced:
            self.history.append(snapshot)

        self.history.sort(key=lambda s: parse_date(s["asat"]))

        self.save_data()
        self.refresh_metrics()
        self.refresh_history_view()
        self.refresh_charts()

    # ============================
    # Metrics / History / Charts
    # ============================
    def refresh_metrics(self):
        self._commit_edit()

        deposits = 0.0
        withdrawals = 0.0

        for iid in self.tree.get_children():
            _d_str, a_str = self.tree.item(iid, "values")
            try:
                a = parse_money(a_str)
            except Exception:
                continue
            if a < 0:
                deposits += -a
            elif a > 0:
                withdrawals += a

        net_deposits = deposits - withdrawals

        bal = None
        try:
            bal = parse_money(self.balance_var.get()) if self.balance_var.get().strip() else None
        except Exception:
            bal = None

        if bal is not None:
            profit = bal + withdrawals - deposits
            self.profit_var.set(f"Profit (balance + withdrawals − deposits): {fmt_money(profit)}")
        else:
            self.profit_var.set("Profit (balance + withdrawals − deposits): —")

        self.total_deposits_var.set(f"Total deposits: {fmt_money(deposits)}")
        self.total_withdrawals_var.set(f"Total withdrawals: {fmt_money(withdrawals)}")
        self.net_deposits_var.set(f"Net deposits (deposits − withdrawals): {fmt_money(net_deposits)}")

    def refresh_history_view(self):
        for iid in self.hist_tree.get_children():
            self.hist_tree.delete(iid)

        for s in self.history:
            asat = s.get("asat", "")
            bal = s.get("balance", "")
            mwrr = s.get("mwrr", None)
            mwrr_txt = f"{mwrr*100:.2f}%" if isinstance(mwrr, (int, float)) else ""
            bal_txt = fmt_money(float(bal)) if bal else ""
            self.hist_tree.insert("", "end", values=(asat, bal_txt, mwrr_txt))

    def refresh_charts(self):
        self._refresh_mwrr_chart()
        self._refresh_balance_vs_invested_chart()

    def _refresh_mwrr_chart(self):
        self.ax_mwrr.clear()
        self.ax_mwrr.set_ylabel("MWRR (p.a.)")
        self.ax_mwrr.set_xlabel("Date")

        if not self.history:
            self.ax_mwrr.text(
                0.5, 0.5,
                "No snapshots yet.\nClick 'Calculate + Save Snapshot' to add points.",
                ha="center", va="center", transform=self.ax_mwrr.transAxes
            )
            self.canvas_mwrr.draw()
            return

        xs, ys = [], []
        for s in self.history:
            try:
                xs.append(parse_date(s["asat"]))
                ys.append(float(s["mwrr"]) * 100.0)
            except Exception:
                pass

        if xs and ys:
            self.ax_mwrr.plot(xs, ys, marker="o", linewidth=1.5)
            self.ax_mwrr.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.ax_mwrr.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
            self.fig_mwrr.autofmt_xdate()

        self.canvas_mwrr.draw()

    def _refresh_balance_vs_invested_chart(self):
        self.ax_bal.clear()
        self.ax_bal.set_ylabel("£")
        self.ax_bal.set_xlabel("Date")

        # Need at least 1 snapshot for a balance line
        if not self.history:
            self.ax_bal.text(
                0.5, 0.5,
                "No snapshots yet.\nClick 'Calculate + Save Snapshot' to record balances over time.",
                ha="center", va="center", transform=self.ax_bal.transAxes
            )
            self.canvas_bal.draw()
            return

        # Sort snapshots
        hist = sorted(self.history, key=lambda s: parse_date(s["asat"]))

        # Pre-parse cashflows for invested baseline calculation
        cashflows = []
        for iid in self.tree.get_children():
            d_str, a_str = self.tree.item(iid, "values")
            try:
                d = parse_date(d_str)
                a = parse_money(a_str)
                cashflows.append((d, a))
            except Exception:
                continue
        cashflows.sort(key=lambda x: x[0])

        def invested_up_to(target_date: datetime) -> float:
            """
            'Invested baseline' = net deposits up to that date (deposits - withdrawals),
            where deposits are negative cashflows and withdrawals are positive cashflows.
            Returns a positive £ amount.
            """
            deposits = 0.0
            withdrawals = 0.0
            for d, a in cashflows:
                if d <= target_date:
                    if a < 0:
                        deposits += -a
                    elif a > 0:
                        withdrawals += a
            return deposits - withdrawals

        xs = []
        balance_line = []
        invested_line = []

        for s in hist:
            try:
                d = parse_date(s["asat"])
                b = float(s["balance"])
            except Exception:
                continue
            xs.append(d)
            balance_line.append(b)
            invested_line.append(invested_up_to(d))

        if not xs:
            self.ax_bal.text(
                0.5, 0.5,
                "No valid snapshot balances to plot.",
                ha="center", va="center", transform=self.ax_bal.transAxes
            )
            self.canvas_bal.draw()
            return

        self.ax_bal.plot(xs, balance_line, marker="o", linewidth=1.5, label="Balance")
        self.ax_bal.plot(xs, invested_line, linewidth=1.5, linestyle="--", label="Net deposits (invested baseline)")

        self.ax_bal.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax_bal.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
        self.fig_bal.autofmt_xdate()

        self.ax_bal.legend(loc="best")
        self.canvas_bal.draw()

    # ============================
    # Snapshot actions (with backup)
    # ============================
    def on_delete_snapshot(self):
        sel = self.hist_tree.selection()
        if not sel:
            return

        self.make_backup()

        for iid in sel:
            asat = self.hist_tree.item(iid, "values")[0]
            self.history = [s for s in self.history if s.get("asat") != asat]

        self.save_data()
        self.refresh_history_view()
        self.refresh_charts()

    def on_clear_snapshots(self):
        if not self.history:
            return

        if messagebox.askyesno(
            "Clear all snapshots",
            "Delete all saved snapshots?\n\nYou can restore the latest backup afterwards."
        ):
            self.make_backup()
            self.history = []
            self.save_data()
            self.refresh_history_view()
            self.refresh_charts()

    # ============================
    # Close
    # ============================
    def on_close(self):
        self._commit_edit()
        self.save_data()
        self.destroy()


if __name__ == "__main__":
    # Nicer default ttk scaling on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = MWRRApp()
    app.mainloop()
