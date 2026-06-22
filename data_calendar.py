# -*- coding: utf-8 -*-
import os
import sys
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import yaml
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi']=300

#%% Inputs
#users inputs
if len(sys.argv)==1:
    path_config=os.path.abspath('configs/config_g3p3.yaml') #config path
    path_out=os.path.abspath('figures') #config path
else:
    path_config=sys.argv[1]#config path
    path_out=sys.argv[2]

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

#%% Initialization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
os.makedirs(path_out,exist_ok=True)
    
subfolders = sorted([
    d for d in os.listdir(config['path_data'])
    if os.path.isdir(os.path.join(config['path_data'], d))
])


times_per_folder = {}
for sf in subfolders:
    folder_path = os.path.join(config['path_data'], sf)
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
out_path = os.path.join(path_out, f'{now_str}.data_calendar.png')
fig.savefig(out_path, dpi=150)
print(f'Saved: {out_path}')
plt.show()
