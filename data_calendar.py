# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi']=300

#%% Inputs
data_dir ='data/g3p3'
fig_dir  = 'figures'
os.makedirs(fig_dir, exist_ok=True)

# ordered: longest match first so YYYYMMDD_HHMMSS is tried before YYYYMMDD_HH
_patterns = [
    (re.compile(r'\d{8}\.\d{6}'), '%Y%m%d.%H%M%S'),
    (re.compile(r'\d{8}_\d{6}'),  '%Y%m%d_%H%M%S'),
    (re.compile(r'\d{8}_\d{2}(?!\d)'), '%Y%m%d_%H'),
]

def extract_datetime(filename):
    for regex, fmt in _patterns:
        m = regex.search(filename)
        if m:
            try:
                return datetime.strptime(m.group(0), fmt)
            except ValueError:
                pass
    return None

#%% Collect datetimes per subfolder
subfolders = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
])

times_per_folder = {}
for sf in subfolders:
    folder_path = os.path.join(data_dir, sf)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    dts = sorted(set(filter(None, (extract_datetime(f) for f in files))))
    times_per_folder[sf] = dts

#%% Plot
fig, ax = plt.subplots(figsize=(14, 0.6 * len(subfolders) + 1.5))

for i, sf in enumerate(subfolders):
    dts = times_per_folder[sf]
    if dts:
        ax.scatter(dts, [i] * len(dts), marker='|', s=100, linewidths=1.2, color=f'C{i % 10}')

ax.set_yticks(range(len(subfolders)))
ax.set_yticklabels(subfolders, fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate()
ax.set_xlabel('Time (UTC)')
ax.set_title('Data availability by subfolder')
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()

now_str = datetime.strftime(datetime.now(),'%Y%m%d.%H%M%S')
out_path = os.path.join(fig_dir, f'{now_str}.data_calendar.png')
fig.savefig(out_path, dpi=150)
print(f'Saved: {out_path}')
plt.show()
