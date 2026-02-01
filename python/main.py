import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = "/content/optimization"
OUT_DPI = 250
TOL = 1e-6
SMOOTH_WINDOW = 5

def smooth(s, w=SMOOTH_WINDOW):
    if w <= 1: return s
    return s.rolling(w, center=True, min_periods=1).median()

def load_one(p):
    df = pd.read_csv(p)
    needed = ["iter","f","grad_norm","time_ms"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{p} no tiene {c}")
    df = df.dropna(subset=needed).sort_values("iter").reset_index(drop=True)
    return df

paths = sorted(glob.glob(os.path.join(CSV_DIR, "**", "*_strong.csv"), recursive=True))
if not paths:
    raise SystemExit("No hay CSV")

# Agrupar por caso (nombre de carpeta)
by_case = {}
for p in paths:
    method = os.path.splitext(os.path.basename(p))[0]
    case = os.path.basename(os.path.dirname(p))
    by_case.setdefault(case, {})[method] = load_one(p)

# f* global (para gap comparable)
f_star = min(df["f"].min() for case in by_case.values() for df in case.values())

def gap(df):
    g = df["f"] - f_star
    return np.maximum(g, 1e-16)

def plot_case(case, methods_dict):
    # filtra métodos con >=2 puntos (si no, quedan invisibles)
    usable = {m:df for m,df in methods_dict.items() if len(df) >= 2}
    tiny = {m:df for m,df in methods_dict.items() if len(df) < 2}

    print(f"\nCASE={case}: total={len(methods_dict)}, graficables(>=2 puntos)={len(usable)}, cortos(<2)={list(tiny.keys())}")

    if not usable:
        print("⚠️ Nada para graficar en este caso (todos tienen <2 puntos).")
        return

    plt.figure(figsize=(14, 10))

    # 1) gap vs iter
    ax1 = plt.subplot(2,2,1)
    for m, df in usable.items():
        y = smooth(pd.Series(gap(df)))
        ax1.semilogy(df["iter"], y, linewidth=2, alpha=0.9, label=m)
    ax1.set_title(f"{case}: gap = f - f* (log) vs iter")
    ax1.set_xlabel("iter"); ax1.set_ylabel("gap (log)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    # 2) grad_norm vs iter
    ax2 = plt.subplot(2,2,2)
    for m, df in usable.items():
        y = smooth(df["grad_norm"])
        y = np.maximum(y, 1e-16)
        ax2.semilogy(df["iter"], y, linewidth=2, alpha=0.9, label=m)
    ax2.axhline(TOL, linestyle="--", linewidth=1)
    ax2.set_title(f"{case}: ||grad|| (log) vs iter")
    ax2.set_xlabel("iter"); ax2.set_ylabel("||grad|| (log)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=9)

    # 3) gap vs time
    ax3 = plt.subplot(2,2,3)
    for m, df in usable.items():
        y = smooth(pd.Series(gap(df)))
        ax3.semilogy(df["time_ms"], y, linewidth=2, alpha=0.9, label=m)
    ax3.set_title(f"{case}: gap (log) vs tiempo")
    ax3.set_xlabel("ms"); ax3.set_ylabel("gap (log)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=9)

    # 4) tiempo a tolerancia
    ax4 = plt.subplot(2,2,4)
    times = {}
    for m, df in methods_dict.items():
        hit = df[df["grad_norm"] <= TOL]
        times[m] = float(hit["time_ms"].iloc[0]) if len(hit) else np.nan
    rank = pd.Series(times).sort_values()
    ax4.barh(rank.index, rank.values)
    ax4.set_title(f"{case}: tiempo hasta ||grad|| ≤ {TOL}")
    ax4.set_xlabel("ms")
    ax4.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out = f"comparativo_{case}.png"
    plt.savefig(out, dpi=OUT_DPI)
    plt.show()
    print("✅ Guardado:", out)

for case, methods in by_case.items():
    plot_case(case, methods)
