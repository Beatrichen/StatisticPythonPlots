import plotly.express as px
import pandas as pd

state_df = pd.read_csv(r'countries_EU_df.csv')
l = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER", "TOTAL"]
print(l)

for word in range(len(l)):
    color = l[word]
    print(color, type(color))
    fig = px.choropleth(state_df, locations="state", color=color,  # lifeExp is a column of gapminder
                                hover_name="state",  # column to add to hover information
                                color_continuous_scale=px.colors.sequential.Plasma)
    fig.show()