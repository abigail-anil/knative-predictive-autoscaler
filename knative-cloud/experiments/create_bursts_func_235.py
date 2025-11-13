#!/usr/bin/env python3
import pandas as pd

# Define your traffic pattern
#pattern = [10, 50, 100, 500, 700, 900, 500, 200, 50, 0, 0, 0, 300, 600, 900]
pattern = [10,50,100,500,700,900,0,0,0,600,900,0,0,800,900]

# Convert to DataFrame
df = pd.DataFrame({"y": pattern})
df.index.name = "minute_index"
df.to_csv("data/bursty_func_235.csv", index=False)

print("Saved data/bursty_func_235.csv with 15-minute burst pattern.")
