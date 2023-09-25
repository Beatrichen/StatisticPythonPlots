import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import matplotlib.ticker as mticker

df = pd.read_csv(r'BGM_covid.csv')
fig, axes = plt.subplots(nrows=1)

df.plot(x="date", y="new_deaths", color='r', ax=axes, kind='scatter')
df.plot(x="date", y="new_cases", color='b', ax=axes, kind='scatter')
df.plot(x="date", y="new_tests", color='g', ax=axes, kind='scatter')
#df.plot(x="date", y="new_tests", color='g', ax=axes, kind='scatter')
axes.set_xlabel('days', fontsize=8, fontweight='bold')
axes.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
axes.tick_params(axis='x', labelrotation=45)
plt.xticks(rotation=90)
axes.set_xticks(axes.get_xticks()[::10])
axes.set_yscale("log")
axes.legend(["new_deaths", "new_cases", "new_test"])
plt.title('Covid', size=8, fontweight='bold')


plt.tight_layout()
plt.rcParams['ytick.labelsize'] = 'small'
plt.show()