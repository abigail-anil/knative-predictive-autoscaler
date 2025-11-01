# mini.py — minimal predictive vs reactive scaling demo

import numpy as np, math
import matplotlib.pyplot as plt

# --- make synthetic traffic ---
T = 10 * 60                          # 600 seconds
t = np.arange(T)
rps = 15 + 20*(1 + np.sin(2*np.pi*(t%60)/60.0))  # wavy minute pattern
rps[(t>=240) & (t<=270)] += 70                    # spike at 4:00–4:30
rps = np.clip(rps, 0, None)

RPS_PER_POD = 20.0                   # one pod handles ~20 rps

# --- tiny predictor ---
def forecast_next_60(history):
    if len(history) == 0:
        return np.zeros(60)
    window = history[-30:] if len(history) >= 30 else history
    return np.full(60, np.mean(window))

# --- simulator ---
def simulate(mode):
    pods = np.zeros(T)
    for s in range(T):
        if mode == "reactive":
            desired_rps = rps[s]
        else:
            fc = forecast_next_60(rps[:s])
            desired_rps = np.percentile(fc, 95)
        pods[s] = math.ceil(desired_rps / RPS_PER_POD)
    return pods

pods_reactive   = simulate("reactive")
pods_predictive = simulate("predictive")

# --- plot pod counts over time ---
plt.figure(figsize=(10,4))
plt.title("Pods: reactive vs predictive")
plt.step(t, pods_reactive,   where="post", label="reactive")
plt.step(t, pods_predictive, where="post", label="predictive")
plt.xlabel("seconds"); plt.ylabel("pods"); plt.legend(); plt.tight_layout()

# save plot instead of showing
plt.savefig("pods_comparison.png")
print("plot saved as pods_comparison.png")
