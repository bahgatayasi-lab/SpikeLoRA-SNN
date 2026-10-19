"""Render seed-averaged E2 condition effects from the verified reanalysis CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
frame=pd.read_csv(here/'results'/'R3_E2_condition_effects.csv')
tasks=['SR','WS','WP','EC']
fig,ax=plt.subplots(figsize=(6.4,3.45))
for f,offset,marker in [(0.1,-.18,'o'),(.2,0,'s'),(.5,.18,'^')]:
    v=frame[frame.fraction==f].set_index('task').loc[tasks,'relative_pct'].to_numpy()
    ax.plot(np.arange(4)+offset,v,linestyle='none',marker=marker,markersize=7,label=f'{int(100*f)}% adaptation')
ax.axhline(0,linewidth=.8,linestyle='--')
ax.set_xticks(range(4),tasks)
ax.set_ylabel('Relative RMSE reduction (%)\nfrom three-seed method means')
ax.set_xlabel('Forecasting task')
ax.set_xlim(-.6,3.6)
ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.18),frameon=False,fontsize=9)
ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(here.parent/'Fig4_E2_condition_effects_R3.pdf',bbox_inches='tight')
plt.close(fig)
