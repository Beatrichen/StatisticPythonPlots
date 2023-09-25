import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

covid = pd.read_csv(r'ITA_covid_model_1.csv')
#print(covid)
print("sample_data statistics:", covid.describe())
ax = sns.regplot(x="icu_patients", y="new_deaths", data=covid)
plt.xticks(rotation=90)
plt.tight_layout()

sns.jointplot(y="new_deaths", x="icu_patients", data=covid, kind="reg")


plt.xticks(rotation=90)
plt.show()


plt.show()