import json
import numpy as np
import matplotlib.pyplot as plt

dataset = 'preterm'
# dataset = 'ECG5000'
json_file = f'./data/{dataset}_0808/shapelet_with_importance.json'

with open(json_file) as f:
    shape_info = json.load(f)
shape_list = []
for i, shape in enumerate(shape_info):
        
    obj = {
        'id': i,
        'len': shape['len'],
        'gain': shape['gain'],
        'vals': shape['wave'][0],
        'imp': shape['imp']
    }
    shape_list.append(obj)

ids = [s['id'] for s in shape_list]
gains = [s['gain'] for s in shape_list]
imps = [s['imp'] for s in shape_list]

x = np.arange(len(ids))
width = 0.35

fig, ax1 = plt.subplots()

rects1 = ax1.bar(x - width/2, gains, width, label='Gain', color='tab:blue')
ax1.set_xlabel('Shapelet ID')
ax1.set_ylabel('Gain', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(x)
ax1.set_xticklabels(ids)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, imps, width, label='Importance', color='tab:orange')
ax2.set_ylabel('Importance', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.suptitle(f'Comparison of Gain and Importance for {dataset}')
fig.tight_layout()
plt.show()