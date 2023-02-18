import seaborn as sns, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.interpolate import make_interp_spline, Rbf, InterpolatedUnivariateSpline, CubicSpline, interp1d, KroghInterpolator
import numpy as np
from datetime import datetime


filePath = "outputs/regretWithT_0.1pi0.2eps3layersPureBanditRegretYabeRegretCIRegret.csv"
df = pd.read_csv(filePath, index_col=0)
df.columns = [name.replace("Regret","") for name in df.columns]
print(df.columns)

axes = []
markers = ['^','h','o']
palette  = sns.color_palette("muted")


for i in range(len(df.columns)):
    y = df.iloc[:, i]
    x = df.index
    # ax = sns.lmplot(y=df.iloc[:,i],x=df.index, ci=None, order=4, truncate=False, data=df)
    X_ = np.linspace(x.min(), x.max(), 50)

    X_Y_Spline = make_interp_spline(df.index, df.iloc[:, i])
    SplineY = X_Y_Spline(X_)

    rbf = Rbf(x, y)
    RBFY = rbf(X_)

    ius = InterpolatedUnivariateSpline(x, y)
    iusY = ius(X_)

    cubic = CubicSpline(x, y)
    cubicY = cubic(X_)
    cubicY2 = cubic(x)

    interped1d = interp1d(x, y)
    interpedY = interped1d(X_)
    interpedY2 = interped1d(x)

    # ax = sns.lineplot(y=df.iloc[:,i],x=df.index, label=df.columns[i], marker = markers[i], markersize=7,color =palette[i])

    # ax = sns.lineplot(y=SplineY,x=X_, label=df.columns[i], marker = markers[i], markersize=7,color =palette[i*3])
    # ax = sns.lineplot(y=RBFY, x=X_, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+1])
    # ax = sns.lineplot(y=iusY, x=X_, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+2])
    # ax = sns.lineplot(y=cubicY, x=X_, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+2])
    ax = sns.lineplot(y=cubicY2, x=x, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+2])
    # ax = sns.lineplot(y=interpedY, x=X_, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+2])
    # ax = sns.lineplot(y=interpedY2, x=x, label=df.columns[i], marker=markers[i], markersize=7, color=palette[i*3+2])
    # ax = sns.lineplot(y=Y_,x=X_, label=df.columns[i],color =palette[i])
    # ax = sns.regplot(data=df.iloc[:, i], label=df.columns[i])
    # ax = sns.regplot(x=df.index, y=df.iloc[:, i], order=0.5)
    axes.append(ax)

ax.get_xaxis().set_minor_locator(mtick.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mtick.AutoMinorLocator())
ax.grid(b=True, which='major', color=(0.9,0.9,0.9), linewidth=1.0)
ax.grid(b=True, which='minor', color=(0.9,0.9,0.9), linewidth=0.5)

plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel('Average Regret', fontsize=14)

# manipulate ticks
ax.set_xlim(left=0,right=100000)
yvals = ax.get_yticks()
ax.set_yticklabels(['{:,.1%}'.format(x) for x in yvals])
xvals = ax.get_xticks()
ax.set_xticklabels(['{:,.1f}'.format(x/1000) + 'K' for x in xvals])
ax.tick_params(axis='both', which='major', labelsize=12)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# plt.legend(bbox_to_anchor=(1.1, 1.05))
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True,prop={'size': 7},framealpha=0.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
          ncol=3, fancybox=True, shadow=True, prop={'size': 11})
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M%S")

plt.savefig('outputs/plots/' + 'regretWithTime' + date_time +'.svg')
plt.show()
