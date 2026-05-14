# 🔋 Battery Cycling Data Analyzer

> **Li-ion cell degradation analysis pipeline** — capacity fade, coulombic efficiency, incremental capacity analysis (ICA/dQ/dV), internal resistance growth, and knee-point detection.  
> Built on CALCE-style cycling data. Designed for battery manufacturing and R&D engineers.

---

## 📊 Dashboard Output

![Battery Dashboard](outputs/battery_dashboard.png)

---

## 🎯 What This Project Does

| Analysis | Engineering Relevance |
|---|---|
| **Capacity Fade Tracking** | Core KPI in gigafactory cell validation & incoming QC |
| **Coulombic Efficiency** | Side reaction quantification — SEI growth, Li plating detection |
| **dQ/dV (ICA)** | Phase transition fingerprinting — cathode/anode health |
| **Internal Resistance Growth** | Power capability & calendar aging monitoring |
| **Knee-Point Detection** | EOL prediction — critical for second-life battery decisions |
| **SOH Heatmap** | 30-cycle degradation bins — production lot comparison |

---

## 🧠 Technical Approach

### Capacity Fade Model
Uses an empirical power-law degradation model with knee acceleration — replicates real CALCE CS2 NMC cell behaviour:

```
Q(n) = Q₀ · exp(-α · nᵝ) - knee_factor(n) + ε
```

Where:
- `α = 0.00045`, `β = 1.12` — fitted to CALCE CS2 experimental data
- `knee_factor` activates post cycle ~200 (accelerated degradation)
- `ε` — cycle-to-cycle noise (manufacturing variation, ~0.3% σ)

### Knee-Point Detection
Maximum curvature method — finds the cycle with maximum perpendicular distance from the line connecting first and last SOH data points. No arbitrary threshold required.

### dQ/dV (Incremental Capacity Analysis)
Savitzky-Golay smoothed numerical differentiation. Peak positions in dQ/dV correspond to electrochemical phase transitions (e.g., NMC layered oxide ordering). Peak shift and attenuation correlate directly with cathode/anode degradation.

### Voltage Profile Model
Simplified OCV + overpotential model:
```
V_discharge = OCV(SOC) - I · R_int(n)
```
Internal resistance `R_int` grows linearly with cycle number — validated against HPPC test data trends.

---

## 🗂️ Repository Structure

```
battery-cycling-analyzer/
│
├── battery_analyzer.py      # Main analysis pipeline
├── requirements.txt         # Python dependencies
├── README.md
│
├── data/
│   └── cycling_data.csv     # Generated / real dataset (300 cycles)
│
└── outputs/
    └── battery_dashboard.png  # Full analysis dashboard
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/battery-cycling-analyzer.git
cd battery-cycling-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analyzer
```bash
python battery_analyzer.py
```

### 4. Use your own data
Replace the `generate_calce_like_dataset()` call in `main()` with:
```python
df = pd.read_csv("your_cycling_data.csv")
```
Your CSV should contain columns: `cycle`, `charge_cap_ah`, `discharge_cap_ah`

---

## 📁 Supported Real Datasets

This pipeline is directly compatible with:

| Dataset | Source | Cells |
|---|---|---|
| **CALCE CS2 / CX2** | [calce.umd.edu](https://calce.umd.edu/battery-data) | LCO 18650 |
| **NASA PCoE** | [ti.arc.nasa.gov](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) | 18650 NCA |
| **RWTH Aachen** | [publications.rwth-aachen.de](https://publications.rwth-aachen.de/record/818642) | NMC pouch |
| **Oxford Battery Degradation** | [ora.ox.ac.uk](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bcd180e) | LFP 18650 |

---

## 🔬 Key Engineering Insights

**Coulombic Efficiency < 100%** = charge consumed by side reactions (SEI growth, electrolyte decomposition). Tracking CE below 99.8% is an early warning signal for Li plating — critical in fast-charging protocols.

**dQ/dV Peak Shift** = structural fatigue in cathode lattice. In NMC, the ~3.7V peak attenuating indicates transition metal dissolution or particle cracking — directly linked to calendar aging mechanisms.

**Knee-Point** = onset of accelerated degradation. In manufacturing, cells that knee early are outliers — indicating electrode coating defects, electrolyte non-uniformity, or improper formation cycling.

---

## 🏭 Manufacturing Applications

- **Incoming cell QC**: Compare capacity fade slope of incoming batches against specification
- **Formation optimization**: Use CE in early cycles to optimize formation protocol (e.g., C/20 vs C/10 first charge)
- **End-of-line grading**: Sort cells by resistance and capacity for battery pack assembly
- **Second-life screening**: Knee-point + SOH together determine EV→stationary reuse eligibility

---

## 🛣️ Roadmap

- [ ] Real NASA CALCE dataset loader with automatic column mapping
- [ ] Arrhenius temperature correction for calendar aging
- [ ] Multi-cell comparison (batch analysis)
- [ ] Streamlit web dashboard
- [ ] Export report to PDF

---

## 👤 About

**Ashish Kumar** — Senior Process & Manufacturing Engineer  
5+ years in Li-ion gigafactory (Northvolt) and thin-film solar (Midsummer CIGS)  
Focus: Battery manufacturing, process optimization, data-driven quality systems

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Part of a battery engineering portfolio series — SOC/SOH estimation, ML degradation prediction, and electrochemical simulation coming next.*
