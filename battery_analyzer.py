"""
Battery Cycling Data Analyzer
==============================
Author : Ashish Kumar | Senior Process & Manufacturing Engineer
GitHub : github.com/ashishkumar
Dataset: NASA CALCE Battery Aging Dataset (synthetic replica for demo)

Analyzes Li-ion cell cycling data to extract:
  - Charge / Discharge capacity curves
  - Capacity fade & coulombic efficiency over cycles
  - dQ/dV differential analysis (degradation fingerprinting)
  - Internal resistance growth estimation
  - Knee-point detection (rapid degradation onset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STYLE CONFIG  (dark engineering aesthetic)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.titlecolor":   "#f0f6fc",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "text.color":        "#c9d1d9",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
    "font.size":         9,
})

COLORS = {
    "charge":     "#58a6ff",
    "discharge":  "#f78166",
    "efficiency": "#3fb950",
    "resistance": "#d29922",
    "gradient":   "#a371f7",
    "knee":       "#ff7b72",
    "accent":     "#79c0ff",
}


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR
#    Mimics NASA CALCE CS2 dataset behaviour
# ─────────────────────────────────────────────
def generate_calce_like_dataset(
    n_cycles: int = 300,
    nominal_capacity_ah: float = 1.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic cycling data that replicates real CALCE CS2 cell aging:
      - Gradual capacity fade with an accelerated knee after ~cycle 200
      - Coulombic efficiency slightly below 100% (realistic Li-ion)
      - Internal resistance growth over life
      - Cycle-to-cycle noise (manufacturing variation)
    """
    rng = np.random.default_rng(seed)
    cycles = np.arange(1, n_cycles + 1)

    # --- Capacity fade model (empirical power law + knee)
    alpha, beta = 0.00045, 1.12          # power-law degradation
    knee_onset  = 200                    # cycle where degradation accelerates
    knee_factor = np.where(
        cycles > knee_onset,
        0.0008 * (cycles - knee_onset) ** 1.5,
        0.0
    )
    noise = rng.normal(0, 0.003, n_cycles)

    discharge_cap = (
        nominal_capacity_ah
        * np.exp(-alpha * cycles ** beta)
        - knee_factor
        + noise
    ).clip(min=0.3)

    # Charge capacity slightly higher (side reactions, SEI growth)
    charge_cap = discharge_cap * rng.uniform(1.003, 1.010, n_cycles)

    # Coulombic efficiency
    coulombic_eff = (discharge_cap / charge_cap) * 100

    # Internal resistance growth (mΩ)  — linear + noise
    resistance = 40 + 0.18 * cycles + rng.normal(0, 1.5, n_cycles)

    # Discharge energy (Wh) — avg voltage drops with aging
    avg_voltage  = 3.72 - 0.0008 * cycles + rng.normal(0, 0.005, n_cycles)
    discharge_energy = discharge_cap * avg_voltage

    return pd.DataFrame({
        "cycle":            cycles,
        "charge_cap_ah":    charge_cap,
        "discharge_cap_ah": discharge_cap,
        "coulombic_eff":    coulombic_eff,
        "resistance_mohm":  resistance,
        "avg_voltage_v":    avg_voltage,
        "discharge_energy_wh": discharge_energy,
        "soh_pct":          (discharge_cap / nominal_capacity_ah) * 100,
    })


def generate_voltage_curves(
    n_cycles_to_plot: list,
    nominal_capacity: float = 1.1,
) -> dict:
    """
    Generate representative charge/discharge voltage profiles (CCCV protocol)
    for selected cycle numbers. Uses a simplified OCV + overpotential model.
    """
    curves = {}
    for cyc in n_cycles_to_plot:
        fade_factor = np.exp(-0.00045 * cyc ** 1.12)
        cap = nominal_capacity * fade_factor
        q   = np.linspace(0, cap, 500)
        soc = q / cap

        # Discharge: OCV curve (sigmoid-like, typical NMC)
        ocv_d = 3.0 + 0.9 * soc - 0.3 * soc**2 + 0.15 * np.sin(np.pi * soc)
        # Overpotential increases with aging (higher resistance)
        r_int = (40 + 0.18 * cyc) / 1000   # Ω
        v_discharge = ocv_d - r_int * 1.0   # 1C discharge current assumed

        # Charge: reverse + higher overpotential
        ocv_c  = ocv_d[::-1]
        v_charge = ocv_c + r_int * 1.0

        curves[cyc] = {
            "q":          q,
            "v_charge":   v_charge.clip(2.7, 4.25),
            "v_discharge": v_discharge.clip(2.5, 4.2),
        }
    return curves


# ─────────────────────────────────────────────
# 2. ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────
def detect_knee_point(cycles: np.ndarray, soh: np.ndarray) -> int:
    """
    Knee-point detection using the maximum curvature method.
    Fits a line from start→end, finds cycle with max perpendicular distance.
    This is the standard approach used in battery degradation research.
    """
    p1 = np.array([cycles[0],  soh[0]])
    p2 = np.array([cycles[-1], soh[-1]])
    line_vec  = p2 - p1
    line_len  = np.linalg.norm(line_vec)
    distances = []
    for i, (c, s) in enumerate(zip(cycles, soh)):
        point = np.array([c, s])
        dist  = np.abs(np.cross(line_vec, p1 - point)) / line_len
        distances.append(dist)
    return cycles[np.argmax(distances)]


def compute_dqdv(voltage: np.ndarray, capacity: np.ndarray) -> tuple:
    """
    Compute dQ/dV — Incremental Capacity Analysis (ICA).
    Peaks reveal phase transitions in cathode/anode. Shift = degradation.
    """
    dv = np.diff(voltage)
    dq = np.diff(capacity)
    mask = np.abs(dv) > 1e-6
    v_mid  = (voltage[:-1] + voltage[1:]) / 2
    dqdv   = np.where(mask, dq / dv, 0)
    # Smooth with Savitzky-Golay to remove noise
    if len(dqdv) > 21:
        dqdv = savgol_filter(dqdv, window_length=21, polyorder=3)
    return v_mid[mask], dqdv[mask]


# ─────────────────────────────────────────────
# 3. MAIN VISUALIZATION DASHBOARD
# ─────────────────────────────────────────────
def plot_full_dashboard(df: pd.DataFrame, curves: dict, knee_cycle: int):
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    fig.suptitle(
        "🔋  Battery Cycling Analysis Dashboard  ·  Li-ion NMC  ·  CALCE-style Dataset",
        fontsize=14, fontweight="bold", color="#f0f6fc", y=1.01
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])   # Voltage curves
    ax2 = fig.add_subplot(gs[0, 2])    # dQ/dV
    ax3 = fig.add_subplot(gs[1, :])    # Capacity fade (full width)
    ax4 = fig.add_subplot(gs[2, 0])    # Coulombic efficiency
    ax5 = fig.add_subplot(gs[2, 1])    # Resistance growth
    ax6 = fig.add_subplot(gs[2, 2])    # SOH heatmap proxy

    cycle_list = list(curves.keys())
    palette    = plt.cm.plasma(np.linspace(0.2, 0.9, len(cycle_list)))

    # ── Plot 1: Voltage Profiles ──────────────────
    for i, (cyc, color) in enumerate(zip(cycle_list, palette)):
        c = curves[cyc]
        ax1.plot(c["q"], c["v_discharge"], color=color, lw=1.6,
                 label=f"Cycle {cyc}")
    ax1.set_xlabel("Capacity (Ah)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Discharge Voltage Profiles — Aging Evolution")
    ax1.legend(fontsize=7.5, ncol=2)
    ax1.grid(True)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # ── Plot 2: dQ/dV Incremental Capacity ────────
    for i, (cyc, color) in enumerate(zip(cycle_list, palette)):
        c   = curves[cyc]
        v_m, dqdv = compute_dqdv(c["v_discharge"], c["q"])
        if len(v_m) > 5:
            ax2.plot(v_m, dqdv, color=color, lw=1.3, label=f"Cyc {cyc}")
    ax2.set_xlabel("Voltage (V)")
    ax2.set_ylabel("dQ/dV (Ah/V)")
    ax2.set_title("ICA — dQ/dV Analysis")
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True)

    # ── Plot 3: Capacity Fade ─────────────────────
    ax3.fill_between(df.cycle, df.discharge_cap_ah,
                     alpha=0.15, color=COLORS["discharge"])
    ax3.plot(df.cycle, df.discharge_cap_ah,
             color=COLORS["discharge"], lw=1.8, label="Discharge Capacity")
    ax3.plot(df.cycle, df.charge_cap_ah,
             color=COLORS["charge"],    lw=1.2, ls="--", label="Charge Capacity", alpha=0.7)

    # Knee annotation
    knee_soh = df.loc[df.cycle == knee_cycle, "discharge_cap_ah"].values
    if len(knee_soh):
        ax3.axvline(knee_cycle, color=COLORS["knee"], lw=1.5, ls=":", alpha=0.8)
        ax3.annotate(
            f"Knee Point\n(Cycle {knee_cycle})",
            xy=(knee_cycle, knee_soh[0]),
            xytext=(knee_cycle + 15, knee_soh[0] + 0.05),
            arrowprops=dict(arrowstyle="->", color=COLORS["knee"], lw=1.2),
            color=COLORS["knee"], fontsize=8,
        )

    # EOL line at 80% SOH
    eol_cap = 1.1 * 0.80
    ax3.axhline(eol_cap, color="#8b949e", lw=1.0, ls="--", alpha=0.7)
    ax3.text(5, eol_cap + 0.01, "EOL (80% SOH)", color="#8b949e", fontsize=8)

    ax3.set_xlabel("Cycle Number")
    ax3.set_ylabel("Capacity (Ah)")
    ax3.set_title("Capacity Fade — Discharge & Charge over Lifetime")
    ax3.legend()
    ax3.grid(True)

    # ── Plot 4: Coulombic Efficiency ──────────────
    ax4.plot(df.cycle, df.coulombic_eff,
             color=COLORS["efficiency"], lw=1.2, alpha=0.7)
    # Rolling average
    roll = df.coulombic_eff.rolling(10).mean()
    ax4.plot(df.cycle, roll, color="#ffffff", lw=1.8, label="10-cycle avg")
    ax4.set_xlabel("Cycle")
    ax4.set_ylabel("CE (%)")
    ax4.set_title("Coulombic Efficiency")
    ax4.set_ylim(97, 101)
    ax4.legend(fontsize=8)
    ax4.grid(True)

    # ── Plot 5: Resistance Growth ─────────────────
    ax5.scatter(df.cycle, df.resistance_mohm,
                color=COLORS["resistance"], s=3, alpha=0.4)
    roll_r = df.resistance_mohm.rolling(10).mean()
    ax5.plot(df.cycle, roll_r, color="#f0c040", lw=1.8)
    ax5.set_xlabel("Cycle")
    ax5.set_ylabel("Resistance (mΩ)")
    ax5.set_title("Internal Resistance Growth")
    ax5.grid(True)

    # ── Plot 6: SOH heatmap per 30-cycle window ───
    window = 30
    bins   = df.groupby(df.cycle // window)["soh_pct"].mean().reset_index()
    bins.columns = ["bin", "avg_soh"]
    colors_soh = plt.cm.RdYlGn(bins.avg_soh / 100)
    bars = ax6.bar(bins.bin * window, bins.avg_soh,
                   color=colors_soh, width=window * 0.85, edgecolor="#0e1117")
    ax6.set_xlabel("Cycle (bin start)")
    ax6.set_ylabel("Avg SOH (%)")
    ax6.set_title("SOH Degradation Map (30-cycle bins)")
    ax6.axhline(80, color=COLORS["knee"], lw=1.2, ls="--")
    ax6.grid(True, axis="y")

    plt.savefig("/home/claude/battery-cycling-analyzer/outputs/battery_dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0e1117")
    print("✅ Dashboard saved → outputs/battery_dashboard.png")
    return fig


# ─────────────────────────────────────────────
# 4. STATISTICAL SUMMARY
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame, knee_cycle: int):
    first = df.iloc[0]
    last  = df.iloc[-1]
    print("\n" + "═" * 55)
    print("  BATTERY CYCLING ANALYSIS — SUMMARY REPORT")
    print("═" * 55)
    print(f"  Total cycles analysed  : {len(df)}")
    print(f"  Initial capacity (Ah)  : {first.discharge_cap_ah:.4f}")
    print(f"  Final capacity   (Ah)  : {last.discharge_cap_ah:.4f}")
    print(f"  Capacity loss          : {(1 - last.soh_pct/100)*100:.1f}%")
    print(f"  Final SOH              : {last.soh_pct:.1f}%")
    print(f"  Knee-point detected    : Cycle {knee_cycle}")
    print(f"  Initial resistance(mΩ) : {first.resistance_mohm:.1f}")
    print(f"  Final resistance  (mΩ) : {last.resistance_mohm:.1f}")
    print(f"  Resistance increase    : {last.resistance_mohm - first.resistance_mohm:.1f} mΩ")
    print(f"  Avg coulombic eff      : {df.coulombic_eff.mean():.3f}%")
    print("═" * 55 + "\n")


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────
def main():
    print("🔋 Battery Cycling Analyzer — Starting...")

    # Generate dataset
    df = generate_calce_like_dataset(n_cycles=300)
    df.to_csv("/home/claude/battery-cycling-analyzer/data/cycling_data.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} cycles")

    # Detect knee point
    knee = detect_knee_point(df.cycle.values, df.soh_pct.values)

    # Generate voltage curves for key cycles
    cycles_to_plot = [1, 25, 50, 100, 150, 200, 250, 300]
    curves = generate_voltage_curves(cycles_to_plot)

    # Print summary
    print_summary(df, knee)

    # Plot dashboard
    plot_full_dashboard(df, curves, knee)
    print("✅ Analysis complete.")


if __name__ == "__main__":
    main()
