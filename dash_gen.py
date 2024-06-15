from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt 
import re
from flask import Flask, jsonify, request, render_template



df = pd.read_excel(r'Пример переданного файла.xlsx')


# Задаем средние значения и матрицу ковариации
mean = [2, 20]
cov = [[1, 0.95], [0.95, 1]]  # Корреляция между x и y равна 0.8

# Генерируем случайные значения с учетом корреляции
# np.random.seed(0)
data = np.random.multivariate_normal(mean, cov, 10000)

# Извлекаем значения x и y
x = data[:, 0]
y = data[:, 1]

#Функция для PERT-распределения
def pert(a, b, c, *, size=1, lamb=4):
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    
    return a + np.random.beta(alpha, beta, size=size) * r

#Функция для распознавания уравнения и вычисления его
def evaluate_expression(expression, df):
    # Находим все переменные в выражении
    variables = re.findall(r'[a-zA-Zа-яА-Я][a-zA-Zа-яА-Я0-9_]*', expression)

    # Создаем переменные из датафрейма
    for var in set(variables):
        if var in df['Параметр'].values:
            globals()[var] = df.loc[df['Параметр'] == var, 'Array'].values[0]
        # else:
        #     # print(f"Переменная '{var}' не найдена в датафрейме.")

    # Вычисляем выражение
    try:
        result = eval(expression)
    except NameError as e:
        # print(f"Ошибка: {e}")
        return None

    return result

#Функция построчного выполнения Монте-Карло по типу распределения
def apply_func(row, n):
    if row['Тип'] == 'binomial' and pd.isna(row['Array']):
        # print(row.name, 'binomial')
        a = np.random.binomial(row['Зн1'], row['Зн2'], size=n)
        return a
    
    elif row['Тип'] == 'geometric' and pd.isna(row['Array']):
        # print(row.name, 'geometric')
        a = np.random.geometric(row['Зн1'], size=n)
        return a
    
    elif row['Тип'] == 'hypergeometric' and pd.isna(row['Array']):
        # print(row.name,)
        a = np.random.hypergeometric(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
        return a
    elif row['Тип'] == 'poisson' and pd.isna(row['Array']):
        # print(row.name,)
        a = np.random.poisson(row['Зн1'], size=n)
        return a
    
    elif row['Тип'] == 'uniform' and pd.isna(row['Array']):
        # print(row.name,)
        a = np.random.uniform(row['Зн1'], row['Зн2'], size=n)
        return a
    
    elif row['Тип'] == 'lognormal' and pd.isna(row['Array']):
        # print(row.name, 'lognormal')
        a = np.random.lognormal(row['Зн1'], row['Зн2'], size=n)
        return a
    
    elif row['Тип'] == 'triangular' and pd.isna(row['Array']):
        # print(row.name, 'triangular')
        a = np.random.triangular(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
        return a
    elif row['Тип'] == 'normal' and pd.isna(row['Array']):
        # print(row.name, 'normal')
        a = np.random.normal(loc=row['Зн1'], scale=row['Зн2'], size=n)
        return a
    
    elif row['Тип'] == 'pert' and pd.isna(row['Array']):
        # print(row.name, 'pert')
        a = pert(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
        return a

    elif row['Тип'] == 'weibull' and pd.isna(row['Array']):
        # print(row.name, 'weibull')
        a = np.random.weibull(row['Зн1'], size=n)
        return a
    
    elif row['Тип'] == 'const' and pd.isna(row['Array']):
        # print(row.name, 'const')
        a = np.full(n, row['Зн1'])
        return a

    else:
        return row['Array']

#Функция запуска расчета уравнения
def apply_func_eq(row, df):
    if row['Тип']=='equation' and pd.isna(row['Array']):
        str_eq = row['Зн1']
        # print(row.name, 'equation')
        a = evaluate_expression(str_eq, df)
        return a

    else:
        return row['Array']
    
#Функция для повторного запуска расчета уравнения на случай, если неизвестные параметры рассчитались только что в другом уравнении
def apply_func_eq_2(row, df):
    if pd.isna(row['Array']).all():
        str_eq = row['Зн1']
        # print(row.name, 'equation')
        a = evaluate_expression(str_eq, df)
        return a

    else:
        return row['Array']

n = int(df[df['Тип']=='size']['Зн1'].values[0])
param_df = df[(df['Тип']!='size')&(df['Тип']!='target')].reset_index(drop=True)
names = list(param_df['Параметр'].values)
param_df['Array'] = pd.Series(dtype='object')
param_df['Array'] = param_df.apply(apply_func, args=(n,), axis=1)
param_df['Array'] = param_df.apply(apply_func_eq, args=(param_df, ), axis=1)
for i in range(param_df[param_df['Тип']=='equation']['Тип'].count()):
    param_df['Array'] = param_df.apply(apply_func_eq_2, args=(param_df,), axis=1)

def div_maker(fig):
    return html.Div(
        children=dcc.Graph(figure=fig),
        style={"display": "flex", "justify-content": "center"},
    )


def pert(a, b, c, *, size=1, lamb=4):
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    return a + np.random.beta(alpha, beta, size=size) * r


n = 1000000
pert1 = pert(6, 7, 9, size=n)
pert2 = np.random.normal(loc=5.0, scale=1.0, size=n)
pert3 = np.random.lognormal(mean=1.0, sigma=0.20, size=n)
pert4 = pert2 * pert3
pert5 = np.random.binomial(10, 0.33, size=n)
pert6 = pert(1, 3, 7, size=n)
arr_all = pert1+pert2/pert3+pert4-pert5*pert6

def plot_first():
    fig = px.histogram(arr_all, 
                   color_discrete_sequence=['aliceblue']
                #   title = "Распределение целевого параметра"
                  )

    fig.add_vline(x=np.median(arr_all), 
                line_width=3, 
                line_dash="dash", 
                line_color="blue")             

    fig.add_annotation(x=np.median(arr_all), y=-0.12,
                text="Медиана",
                showarrow=False,
                yref="paper")

    fig.add_vline(x=np.quantile(arr_all, 0.05), 
                line_width=3, 
                line_dash="dash", 
                line_color='#ff7f0e')

    fig.add_annotation(x=np.quantile(arr_all, 0.05), y=-0.12,
                text="5 %",
                showarrow=False,
                yref="paper")

    fig.add_vline(x=np.quantile(arr_all, 0.95), 
                line_width=3, 
                line_dash="dash", 
                line_color='#ff7f0e')

    fig.add_annotation(x=np.quantile(arr_all, 0.95), y=-0.12,
                text="95 %",
                showarrow=False,
                yref="paper")


    fig.update_layout(margin=dict(l=20, r=20, t=30, b=160))

    # # add annotation
    text = "Среднее="+str(np.median(arr_all))
    fig.add_annotation(dict(font=dict(color='black',size=10),
                                            x=0,
                                            y=-0.22,
                                            showarrow=False,
                                            text='Статистические данные',
                                            textangle=0,
                                            xanchor='left',
                                            xref="paper",
                                            yref="paper"))

    fig.add_annotation(dict(font=dict(color='black',size=10),
                                            x=0,
                                            y=-0.30,
                                            showarrow=False,
                                            text=text,
                                            textangle=0,
                                            xanchor='left',
                                            xref="paper",
                                            yref="paper"))

    fig.add_annotation(dict(font=dict(color='black',size=10),
                                            x=0,
                                            y=-0.38,
                                            showarrow=False,
                                            text="...",
                                            textangle=0,
                                            xanchor='left',
                                            xref="paper",
                                            yref="paper"))

    return div_maker(fig)


coef_list = []
target_min_list = []
target_max_list = []
target_mean_list = []
diff = []
base = []

# array_list = [arr, arr_norm, arr_log]
array_list = [pert1, pert2, pert3, pert4, pert5, pert6]
for i in array_list:
    model = LinearRegression()
    model.fit(i.reshape(-1, 1), arr_all)
    min_i = np.min(i)
    max_i = np.max(i)
    mean_i = np.mean(i)
    diff_i = np.max(i) - np.min(i)
    base_i = np.mean(arr_all) -mean_i+ min_i
    coef_list.append(model.coef_[0])
    target_min_list.append(model.coef_[0]*min_i)
    target_max_list.append(model.coef_[0]*max_i)
    target_mean_list.append(model.coef_[0]*mean_i)
    diff.append(diff_i)
    base.append(base_i)

X_labels = ['Параметр_1', 'Параметр_2', 'Параметр_3', 'Параметр_4', 'Параметр_5', 'Параметр_6']
sorted_idx = np.argsort(np.abs(diff))#[::-1]
sorted_diff = np.array(diff)[sorted_idx]
sorted_min_target = np.array(target_min_list)[sorted_idx]
sorted_max_target = np.array(target_max_list)[sorted_idx]
sorted_base = np.array(base)[sorted_idx]
sorted_labels = [X_labels[i] for i in sorted_idx]




def plot_second ():

    fig = go.Figure()
    fig.add_trace(go.Bar(y=sorted_labels, x=sorted_diff,
                    base=sorted_base,
                    marker_color='rgb(158,202,225)',
                    marker_line_color='rgb(8,48,107)',
                    orientation='h',
                    marker_line_width=1.5,
                    opacity= 0.7,
                    text = sorted_diff,
                    textposition='auto',
                    texttemplate = "Диапазон: %{x:,s} "
    ))

    fig.update_layout(
        height=500,
        margin=dict(t=50,l=10,b=10,r=10),
    # title_text="Диаграмма Торнадо",
    # title_font_family="sans-serif",
    # #legend_title_text=’Financials’,
    # title_font_size = 25,
    # title_font_color="darkblue",
    # title_x=0.5 #to adjust the position along x-axis of the title
    )
    fig.update_layout(barmode='overlay', 
                    #xaxis_tickangle=-45, 
                    legend=dict(
                        x=0.80,
                        y=0.01,
    bgcolor='rgba(255, 255, 255, 0)',
    bordercolor='rgba(255, 255, 255, 0)'
    ),
                    yaxis=dict(
    title='Показатель',
    titlefont_size=16,
    tickfont_size=14
    ),
                    bargap=0.30)

    fig.add_vline(x=np.median(arr_all), 
                line_width=3, 
                line_dash="dash", 
                line_color="blue") 


    fig.update_layout(margin=dict(l=20, r=20, t=50, b=35))#,paper_bgcolor="LightSteelBlue")

    fig.add_annotation(dict(font=dict(color='black',size=12),
                                            x=np.mean(arr_all),
                                            y=-0.08,
                                            showarrow=False,
                                            text='Среднее по таргету',
                                            yref="paper"))


    return div_maker(fig)

plot_hist = plot_first()
plot_tornado = plot_second()
title_first = html.H3(children='Распределение целевого параметра', className='first__subtitle')
title_second = html.H3(children='Диаграмма торнадо', className='subtitle')

server = Flask(__name__)
app = Dash(__name__, server=server)
app._favicon = "icon.png"
app.title = "Дашборд" 
app.layout = html.Div(children=[
    title_first,
    html.Div(children=[plot_hist], className='hist__plot'),
    title_second,
    html.Div(children=[plot_tornado], className='tornado__plot')
]
    )

# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )

@server.route('/json-example')
def json_example():

    newApp = Dash()
    newApp._favicon = "icon.png"
    newApp.title = "Дашборд"
    
    newApp.layout = html.Div(children=[
        title_first,
        title_second,
        html.Div(children=[plot_tornado], className='tornado__plot')
    ]
        )
    


    return newApp.index()

if __name__ == '__main__':
    app.run(debug=True)