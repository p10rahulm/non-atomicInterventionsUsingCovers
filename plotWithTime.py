import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import make_interp_spline
import numpy as np
from datetime import datetime

# filePath = "outputs/regretWithT_0.1pi0.2eps3layersPureBanditRegretYabeRegretCIRegret.csv"
filePath = "outputs/regretWithTime_0.0pi0.05eps2degreeDirectExpRegretYabeRegretCIRegret20230218_151619.csv"

df = pd.read_csv(filePath, index_col=0)
df.columns = [name.replace("Regret", "") for name in df.columns]
print(df.columns)

axes = []
markers = ['^', 'h', 'o']
palette = sns.color_palette("muted")
# sns.set(rc={'figure.figsize':(3.5,2.5)})
plt.figure(figsize=(6, 4))
legendNames = ['DirectExploration', 'PropInf', 'CoveringInterventions']
for i in range(len(df.columns)):
    y = df.iloc[:, i]
    x = df.index[:]
    X_Y_Spline = make_interp_spline(df.index, df.iloc[:, i])
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)

    ax = sns.lineplot(y=y, x=x, label=legendNames[i], marker=markers[i], markersize=7, color=palette[i])
    # ax = sns.lineplot(y=Y_,x=X_, label=df.columns[i], marker = markers[i], markersize=7,color =palette[i])
    # ax = sns.lineplot(y=Y_,x=X_, label=df.columns[i],color =palette[i])
    # ax = sns.regplot(data=df.iloc[:, i], label=df.columns[i])
    # ax = sns.regplot(x=df.index, y=df.iloc[:, i], order=0.5)
    axes.append(ax)

ax.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
ax.grid(b=True, which='major', color=(0.9, 0.9, 0.9), linewidth=1.0)
ax.grid(b=True, which='minor', color=(0.9, 0.9, 0.9), linewidth=0.5)

plt.xlabel('Time Horizon T', fontsize=14)
plt.ylabel('Simple Regret', fontsize=14)

# manipulate ticks
ax.set_xlim(left=0, right=x.max())
ax.set_ylim(bottom=0, top=0.05)
yvals = ax.get_yticks()
ax.set_yticklabels(['{:,.2}'.format(x) for x in yvals])
xvals = ax.get_xticks()
ax.set_xticklabels(['{:,.1f}'.format(x / 1000) + 'K' for x in xvals])
ax.tick_params(axis='both', which='major', labelsize=12)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True,prop={'size': 7},framealpha=0.5)
# ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
#           ncol=3, fancybox=True, shadow=True, prop={'size': 11})
ax.legend(loc='best', fancybox=True, shadow=True, prop={'size': 11})
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")

# plt.savefig('outputs/plots/' + 'regretWithTime' + date_time +'.svg', format='svg', dpi=1200)
plt.savefig('outputs/plots/' + 'regretWithTime' + date_time + '.png', format='png')
plt.show()
