import pandas as pd 
import numpy as np 
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from IPython.display import display, HTML

df = pd.read_csv("../input/housing-prices/train.csv")

def scrollable_table(df, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += df.to_html()
    html += '</div>'
    return html

numerical_features = df.select_dtypes(include=[np.number])
summary_statistics = numerical_features.describe().T
html_numerical = scrollable_table(summary_statistics, 'numerical_features', 'Summary statistics for numerical features')

categorical_features = df.select_dtypes(include=[object])
cat_summary_statistics = categorical_features.describe().T
html_categorical = scrollable_table(cat_summary_statistics, 'categorical_features', 'Summary statistics for categorical features')

null_values = df.isnull().sum()
html_null_values = scrollable_table(null_values.to_frame(), 'null_values', 'Null values in the dataset')

missing_percentage = (df.isnull().sum() / len(df)) * 100

html_missing_percentage = scrollable_table(missing_percentage.to_frame(), 'missing_percentage', 'Percentage of missing values for each feature')

rows_with_missing_values = df[df.isnull().any(axis=1)]
html_rows_with_missing_values = scrollable_table(rows_with_missing_values.head(), 'rows_with_missing_values', 'Rows with missing values')


dwelling_types = df['BldgType'].value_counts()
dwelling_prices = df.groupby('BldgType')['SalePrice'].mean()

formatted_dwelling_prices = ['$' + f'{value:,.2f}' for value in dwelling_prices.values]

fig1 = go.Figure(data=[go.Bar(
    x=dwelling_types.index,
    y=dwelling_types.values,
    marker_color='rgb(76, 175, 80)',
    text=dwelling_types.values,
    textposition='outside',
    width=0.4,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig1.update_layout(
    title='Distribution of Building Types',
    xaxis_title='Building Type',
    yaxis_title='Count',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig2 = go.Figure(data=[go.Bar(
    x=dwelling_prices.index,
    y=dwelling_prices.values,
    marker_color='rgb(156, 39, 176)',
    text=formatted_dwelling_prices,
    textposition='outside',
    width=0.4,
    marker=dict(line=dict(width=2, color='rgba(0,0,0,1)'), opacity=1)
)])
fig2.update_layout(
    title='Average Sale Price by Building Type',
    xaxis_title='Building Type',
    yaxis_title='Price',
    plot_bgcolor='rgba(34, 34, 34, 1)',
    paper_bgcolor='rgba(34, 34, 34, 1)',
    font=dict(color='white')
)

fig1.show()
fig2.show()
