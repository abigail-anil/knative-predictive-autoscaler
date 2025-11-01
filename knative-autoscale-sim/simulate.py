import numpy as np                 # numpy: fast arrays + math on arrays
import pandas as pd                # pandas: convenient tables (DataFrame)
import math                        # math.ceil for rounding up pod counts
import matplotlib.pyplot as plt    # plotting

np.random.seed(42)                 # fix randomness so results are repeatable

# ---------------------------------------------
# Config
# ---------------------------------------------
seconds = 10 * 60                  # total simulation time (10 minutes = 600 seconds)

def make_workload(pattern, seconds):
    t = np.arange(seconds)  # time axis: [0, 1, 2, ..., 599]

    if pattern == "periodic+spike":
        # Build requests-per-second (rps) as:
        # base load + periodic wave + short spike + random noise
        base = 15
        periodic = 20 * (1 + np.sin(2*np.pi*(t % 60) / 60.0))
        #   - t % 60 repeats 0..59 each minute
        #   - 2*pi*(...) / 60 turns that into radians, one full sine cycle per minute
        #   - (1 + sin) shifts the curve up to keep it positive
        #   - *20 scales the amplitude
        spike = np.where((t >= 240) & (t <= 270), 80, 0)
        #   - boolean mask selects 240–270 seconds (30s window)
        #   - add 80 rps during that window to simulate a burst
        noise = np.random.normal(0, 3, size=seconds)
        #   - gaussian noise (mean=0, stdev=3) to make it less “perfect”
        rps = base + periodic + spike + noise         #   - combine all parts

    elif pattern == "steady":
        rps = np.full(seconds, 50)  # flat 50 RPS

    elif pattern == "ramp":
        rps = np.linspace(10, 120, seconds)  # slowly rise from 10 → 120 RPS

    else:
        raise ValueError("Unknown pattern")

    return np.clip(rps, 0, None)          #   - clip at 0 so rps never becomes negative

# --------------------------------------------
# Predictor (moving average baseline)
# --------------------------------------------
def predict_next_minute(history_rps, horizon=60):
    """Very simple predictor: average of last 30s, projected flat for next 'horizon' seconds."""
    if len(history_rps) < 30:
        # if we don't have 30 seconds of history yet, average what we do have
        avg = np.mean(history_rps) if len(history_rps) > 0 else 0
    else:
        avg = np.mean(history_rps[-30:])   # mean of last 30 seconds
    return np.full(horizon, max(avg, 0))   # repeat that value horizon times

# --------------------------------------------
# Policy: forecast -> pods
# --------------------------------------------
RPS_PER_POD = 20.0     # assume one warm pod comfortably serves ~20 rps (capacity model)
MAX_PODS    = 8        # don’t exceed this cap
COOLDOWN_UP = 30       # after increasing pods, wait at least 30s before changing again
COOLDOWN_DN = 90       # after decreasing pods, wait at least 90s

def pods_from_forecast(pred_60s):
    """Convert a 60s forecast into a pod count (conservative: use 95th percentile)."""
    p95 = np.percentile(pred_60s, 95)      # high-end of expected load in next minute
    want = math.ceil(p95 / RPS_PER_POD)    # pods = ceil(expected_rps / capacity_per_pod)
    return int(np.clip(want, 0, MAX_PODS)) # bound the result

# --------------------------------------------
# Simulation
# --------------------------------------------
WARMUP = 5   # seconds a brand-new pod needs before it can serve (cold start time)

for pattern in ["periodic+spike", "steady", "ramp"]:
    # 1. Generate traffic for this pattern
    rps = make_workload(pattern, seconds)
    t = np.arange(seconds)  # define t here for DataFrame

    # 2. Run simulation for both modes (reactive + predictive)
    results = [] # we’ll collect per-mode outputs here

    for mode in ["reactive", "predictive"]:
        pods = 0                     # current pod count
        last_change = -10**9         # timestamp of last scale change; start far in past
        cold_delay = np.zeros(seconds) # per-second extra delay users feel due to cold starts
        pods_t = np.zeros(seconds)     # pods over time (for plotting)

        for s in range(seconds):     # simulate each second
            # ---- decide desired pod count for this second ----
            if mode == "reactive":
                # reactive: respond to *current* rps only (no look-ahead)
                desired = math.ceil(rps[s] / RPS_PER_POD)
            else:
                # predictive: look 60s ahead using history so far, then be conservative (p95)
                fc = predict_next_minute(rps[:s]) if s > 0 else np.zeros(60)
                desired = pods_from_forecast(fc)

            # ---- apply cooldowns to avoid noisy thrashing ----
            now = s
            change_allowed = (
                (desired > pods and now - last_change >= COOLDOWN_UP) or
                (desired < pods and now - last_change >= COOLDOWN_DN)
            )
            if change_allowed and desired != pods:
                pods = desired
                last_change = now

            # ---- compute cold-start penalty users feel this second ----
            capacity = pods * RPS_PER_POD                # how much this many pods can handle
            if rps[s] > capacity:                        # if demand exceeds capacity
                overload = rps[s] - capacity             # how much we’re short by
                frac = np.clip(overload / max(rps[s], 1e-6), 0, 1)
                #   - fraction of requests that get delayed (bounded 0..1)
                cold_delay[s] = WARMUP * frac            # extra delay felt, up to WARMUP seconds
            else:
                cold_delay[s] = 0

            pods_t[s] = pods                              # record current pods for plotting

        # store outputs for this mode
        results.append({"mode": mode, "pods": pods_t, "delay": cold_delay})

    # -------------------------
    # 5) Compare and visualize
    # -------------------------
    df = pd.DataFrame({
        "t": t,
        "rps": rps,
        "pods_reactive":   results[0]["pods"],
        "pods_predictive": results[1]["pods"],
        "delay_reactive":  results[0]["delay"],
        "delay_predictive":results[1]["delay"],
    })

    # -------------------------
    # Print summary metrics
    # -------------------------
    print(f"\n===== Summary Metrics for {pattern} =====")
    summary = {
        "Reactive": {
            "Avg cold delay (s)": round(df["delay_reactive"].mean(), 3),
            "Max cold delay (s)": round(df["delay_reactive"].max(), 3),
            "Avg pods": round(df["pods_reactive"].mean(), 3)
        },
        "Predictive": {
            "Avg cold delay (s)": round(df["delay_predictive"].mean(), 3),
            "Max cold delay (s)": round(df["delay_predictive"].max(), 3),
            "Avg pods": round(df["pods_predictive"].mean(), 3)
        }
    }
    for mode, metrics in summary.items():
        print(f"\n{mode}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # ---- Plot 1: traffic shape ----
    plt.figure(figsize=(11,4))
    plt.title(f"Traffic (RPS) — {pattern}")
    plt.plot(df["t"], df["rps"])                      # simple line
    plt.xlabel("seconds"); plt.ylabel("rps"); plt.tight_layout()
    plt.savefig(f"{pattern}_01_traffic.png")          # save unique file per pattern
    plt.close()

    # ---- Plot 2: pods over time ----
    plt.figure(figsize=(11,4))
    plt.title(f"Pods over time (reactive vs predictive) — {pattern}")
    plt.step(df["t"], df["pods_reactive"],   where="post", label="reactive")
    plt.step(df["t"], df["pods_predictive"], where="post", label="predictive")
    plt.legend(); plt.xlabel("seconds"); plt.ylabel("pods"); plt.tight_layout()
    plt.savefig(f"{pattern}_02_pods.png")
    plt.close()

    # ---- Plot 3: cold-start penalty ----
    plt.figure(figsize=(11,4))
    plt.title(f"Cold-start penalty — {pattern}")
    plt.plot(df["t"], df["delay_reactive"],  label="reactive")
    plt.plot(df["t"], df["delay_predictive"],label="predictive")
    plt.legend(); plt.xlabel("seconds"); plt.ylabel("seconds"); plt.tight_layout()
    plt.savefig(f"{pattern}_03_cold_start_penalty.png")
    plt.close()

    # ---- Plot 4: traffic + pods overlay ----
    fig, ax1 = plt.subplots(figsize=(11,4))
    ax1.set_xlabel("seconds")
    ax1.set_ylabel("RPS", color="tab:blue")
    ax1.plot(df["t"], df["rps"], color="tab:blue", label="traffic (RPS)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Pods", color="tab:orange")
    ax2.step(df["t"], df["pods_reactive"], where="post", color="tab:green", label="reactive pods")
    ax2.step(df["t"], df["pods_predictive"], where="post", color="tab:orange", label="predictive pods")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.legend(loc="upper right")
    plt.title(f"Traffic vs Pods — {pattern}")
    plt.savefig(f"{pattern}_04_traffic_vs_pods.png")
    plt.close()

    print(f"Saved plots for {pattern}: *_traffic.png, *_pods.png, *_cold_start_penalty.png, *_traffic_vs_pods.png")
