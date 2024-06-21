from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, State
import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt 
import re
from flask import Flask, jsonify, request, render_template
from scipy.stats import kurtosis, skew, rankdata
from scipy.stats.distributions import norm
import itertools

# Функция для преобразования строки в float при необходимости
def convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return value


def pert(a, b, c, *, size=1, lamb=4):
    r = c - a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    
    return a + np.random.beta(alpha, beta, size=size) * r

def evaluate_expression(expression, df):
    # Находим все переменные в выражении
    variables = re.findall(r'[a-zA-Zа-яА-Я][a-zA-Zа-яА-Я0-9_]*', expression)

    # Создаем переменные из датафрейма
    for var in set(variables):
        if var in df['Параметр'].values:
            globals()[var] = df.loc[df['Параметр'] == var, 'Array'].values[0]
        else:
            print(f"Переменная '{var}' не найдена в датафрейме.")

    # Вычисляем выражение
    try:
        result = eval(expression)
    except NameError as e:
        print(f"Ошибка: {e}")
        return None

    return result

def apply_func(row, n):
    if row['Тип'] == 'binomial' and pd.isna(row['Array']):
        result_of_MC = np.random.binomial(row['Зн1'], row['Зн2'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'geometric' and pd.isna(row['Array']):
        result_of_MC = np.random.geometric(row['Зн1'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'hypergeometric' and pd.isna(row['Array']):
        result_of_MC = np.random.hypergeometric(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'poisson' and pd.isna(row['Array']):
        result_of_MC = np.random.poisson(row['Зн1'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'uniform' and pd.isna(row['Array']):
        result_of_MC = np.random.uniform(row['Зн1'], row['Зн2'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'lognormal' and pd.isna(row['Array']):
        result_of_MC = np.random.lognormal(row['Зн1'], row['Зн2'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'triangular' and pd.isna(row['Array']):
            result_of_MC = np.random.triangular(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
            return result_of_MC

    elif row['Тип'] == 'normal' and pd.isna(row['Array']):
        result_of_MC = np.random.normal(loc=row['Зн1'], scale=row['Зн2'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'pert' and pd.isna(row['Array']):
        result_of_MC = pert(row['Зн1'], row['Зн2'], row['Зн3'], size=n)
        return result_of_MC

    elif row['Тип'] == 'weibull' and pd.isna(row['Array']):
        result_of_MC = np.random.weibull(row['Зн1'], size=n)
        return result_of_MC
    
    elif row['Тип'] == 'const' and pd.isna(row['Array']):
        result_of_MC = np.full(n, row['Зн1'])
        return result_of_MC

    else:
        return row['Array']

def apply_func_eq(row, df):
    if row['Тип']=='equation' and pd.isna(row['Array']):
        str_eq = row['Зн1']
        result_of_evaluation = evaluate_expression(str_eq, df)
        return result_of_evaluation

    else:
        return row['Array']

def apply_func_eq_2(row, df):
    if row['Тип']=='equation' and pd.isna(row['Array']).all():
        str_eq = row['Зн1']
        result_of_evaluation = evaluate_expression(str_eq, df)
        return result_of_evaluation

    else:
        return row['Array']

def apply_func_target(row, df):
    if row['Тип']=='target' and pd.isna(row['Array']):
        str_eq = row['Зн1']
        result_of_evaluation = evaluate_expression(str_eq, df)
        return result_of_evaluation

    else:
        return row['Array']

def chol(A):
    """
    Calculate the lower triangular matrix of the Cholesky decomposition of
    a symmetric, positive-definite matrix.
    """
    A = np.array(A)
    assert A.shape[0] == A.shape[1], "Input matrix must be square"

    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = (
                (A[i][i] - s) ** 0.5 if (i == j) else (1.0 / L[j][j] * (A[i][j] - s))
            )

    return np.array(L)

def induce_correlations(data, corrmat):
    """
    Induce a set of correlations on a column-wise dataset
    
    Parameters
    ----------
    data : 2d-array
        An m-by-n array where m is the number of samples and n is the
        number of independent variables, each column of the array corresponding
        to each variable
    corrmat : 2d-array
        An n-by-n array that defines the desired correlation coefficients
        (between -1 and 1). Note: the matrix must be symmetric and
        positive-definite in order to induce.
    
    Returns
    -------
    new_data : 2d-array
        An m-by-n array that has the desired correlations.
        
    """
    # Create an rank-matrix
    data_rank = np.vstack([rankdata(datai) for datai in data.T]).T

    # Generate van der Waerden scores
    data_rank_score = data_rank / (data_rank.shape[0] + 1.0)
    data_rank_score = norm(0, 1).ppf(data_rank_score)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the desired correlation matrix
    p = chol(corrmat)

    # Calculate the current correlations
    t = np.corrcoef(data_rank_score, rowvar=0)

    # Calculate the lower triangular matrix of the Cholesky decomposition
    # of the current correlation matrix
    q = chol(t)

    # Calculate the re-correlation matrix
    s = np.dot(p, np.linalg.inv(q))

    # Calculate the re-sampled matrix
    new_data = np.dot(data_rank_score, s.T)

    # Create the new rank matrix
    new_data_rank = np.vstack([rankdata(datai) for datai in new_data.T]).T

    # Sort the original data according to new_data_rank
    for i in range(data.shape[1]-1):
        vals, order = np.unique(
            np.hstack((data_rank[:, i], new_data_rank[:, i])), return_inverse=True
        )
        old_order = order[: new_data_rank.shape[0]]
        new_order = order[-new_data_rank.shape[0] :]
        tmp = data[np.argsort(old_order), i][new_order]
        data[:, i] = tmp[:]

    return data

def correlate(params, corrmat):
    """
    Force a correlation matrix on a set of statistically distributed objects.
    This function works on objects in-place.
    
    Parameters
    ----------
    params : array
        An array of of uv objects.
    corrmat : 2d-array
        The correlation matrix to be imposed
    
    """
    # Put each ufunc's samples in a column-wise matrix
    data = np.vstack([param for param in params]).T

    # Apply the correlation matrix to the sampled data
    new_data = induce_correlations(data, corrmat)

    # Re-set the samples to the respective variables
    new_params = []
    for i in range(len(params)):
        new_params.append(new_data[:, i])#[i] = new_data[:, i]
    
    return new_params





def div_maker(fig):
    return html.Div(
        children=dcc.Graph(figure=fig),
        style={"display": "flex", "justify-content": "center"},
    )



def plot_first(arr_all, target, j=0):
    fig = px.histogram(arr_all, 
                        color_discrete_sequence=['white']
                        )
        
    fig.update_xaxes(title_text='Значение')

    fig.update_yaxes(title_text='Количество')

    fig.update_layout(showlegend=False)

    fig.add_vline(x=np.median(arr_all), 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="blue")             

    fig.add_annotation(x=np.median(arr_all), y=-0.17,
                    text="Медиана",
                    showarrow=False,
                    yref="paper")

    fig.add_vline(x=np.quantile(arr_all, 0.05), 
                    line_width=3, 
                    line_dash="dash", 
                    line_color='#ff7f0e')

    fig.add_annotation(x=np.quantile(arr_all, 0.05), y=-0.17,
                    text="5 %",
                    showarrow=False,
                    yref="paper")

    fig.add_vline(x=np.quantile(arr_all, 0.95), 
                    line_width=3, 
                    line_dash="dash", 
                    line_color='#ff7f0e')

    fig.add_annotation(x=np.quantile(arr_all, 0.95), y=-0.17,
                    text="95 %",
                    showarrow=False,
                    yref="paper")


    fig.update_layout(margin=dict(l=20, r=20, t=30, b=60))

    return div_maker(fig)





def plot_second (arr_all, sorted_labels, sorted_diff, sorted_base):
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
            # title_font_family="sans-serif",
            #legend_title_text=’Financials’,
            # title_font_size = 25,
            # title_font_color="darkblue",
            title_x=0.5 #to adjust the position along x-axis of the title
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
                            title='Показатель'
                            # titlefont_size=16,
                            # tickfont_size=14
                        ),
                    bargap=0.30)

    fig.add_vline(x=np.mean(arr_all), 
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

# plot_hist = plot_first()
# plot_tornado = plot_second()
title_first = html.H3(children='Распределение целевого параметра', className='first__subtitle')
title_second = html.H3(children='Диаграмма торнадо', className='subtitle')

logo = html.Img(className='logo', src=r'assets/Рисунок5.png', alt='Логотип RiskVision', id='testts')
head_article_first_header = html.H3(children='О сервисе', className='article__subtitle')
head_article_second_header = html.H3(children='Как пользоваться', className='article__subtitle')
MK_link = html.A(children='Монте-Карло. ', 
                 href='https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9C%D0%BE%D0%BD%D1%82%D0%B5-%D0%9A%D0%B0%D1%80%D0%BB%D0%BE',
                 target="_blank",
                 className='link')
head_article_first_text = html.P(children=['Сервис для моделирования рисков методом ', MK_link, ' Вы задаете формулу и параметры - получаете диаграммы и статистику по целевому показателю'], className='article__text')
head_article_second_text = html.P(children=['Укажите число итераций, введите параметры и их характеристики, задайте формулу расчёта. Нажмите "Рассчитать"'], className='article__text')
head_article_first = html.Article(children=[head_article_first_header,head_article_first_text], className='article')
head_article_second = html.Article(children=[head_article_second_header,head_article_second_text], className='article')
head_content = html.Div(children=[head_article_first, head_article_second], className='content__article')
main_section = html.Section(children=[logo, head_content], className='section')

iteration_number_lable = html.Label(children='Количество итераций Монте-Карло:')
iteration_number_input = dcc.Input(type="text", className='iteration_number_input', value='10000')
iteration_number = html.Div(children=[iteration_number_lable, iteration_number_input], className='stable__input_place')
param_p = html.P(children='Параметры', className='article__text')
button_add_param = html.Button(children='+', className='button__add', type='button', id ='addButtonParam')
param_space = html.Div(children=[button_add_param], className='params__space', id='paramSpace')
corr_p = html.P(children='Корреляции', className='article__text')
button_add_corr = html.Button(children='+', className='button__add', type='button', id ='addButtonCorr')
correlation__space = html.Div(children=[button_add_corr], className='correlation__space')

formula_lable = html.Label(children='Введите формулу: y =')
formula_input = dcc.Input(type="text", className='formula_input', pattern='^[a-zA-Zа-яёА-ЯЁ\s\+\\\_\*\-\)\(0-9]+$')
formula = html.Div(children=[formula_lable, formula_input], className='stable__input_place')

form_content = html.Form(children=[iteration_number, param_p, param_space, corr_p, correlation__space, formula], className='form__content', id='mainForm')
button_submit = html.Button(children='Рассчитать', className='button__submit', type='button', id='calculateButton')
form_section = html.Section(children=[form_content, button_submit], className='section__form')
preloader_content = html.Div(children=[html.Div(), html.Div(), html.Div(), html.Div()], className='lds-ring')
preload_text = html.Div(children='Идёт расчёт', className='preload_text')
preload_section = html.Section(children=[preload_text, preloader_content], className='preload_hide', id='preloader')
main = html.Div(children=[main_section, form_section, preload_section], className='content')

# test_div = html.Div(children='test', id='testDiv')
# test_div2 = html.Div(children='test2', id='testDiv2')
footer_p_AA = html.P(children='Александр Алымов', className='article__text')
footer_p_PM = html.P(children='Пётр Мельников', className='article__text')
footer = html.Div(children=[footer_p_AA, footer_p_PM], className='footer')

visPlace = html.Div(id='visPlace')

index_page = html.Div(children=[main, visPlace, footer], className='page')

server = Flask(__name__)
app = Dash(__name__, server=server)
app._favicon = "icon.png"
app.title = "Дашборд" 
app.layout = html.Div(children=[
    dcc.Store(id='getComm'),
    index_page])

app.clientside_callback(
    """
    function(id) {

        function createInput() {
            template = document.createElement('div');
            template.classList.add('inpute__space');
            buttonMinus = document.createElement('div');
            buttonMinus.textContent = '-';
            buttonMinus.classList.add('button__minus');
            template.appendChild(buttonMinus);  

            nameInput = document.createElement('div');
            nameInput.classList.add('one_inpute_place');
            nameLable = document.createElement('lable');
            nameLable.textContent = 'Имя';
            nameLable.setAttribute("for", "param");
            nameInput.appendChild(nameLable);
            nameText = document.createElement('input');
            nameText.setAttribute("id", "param");
            nameText.classList.add('input');
            nameInput.appendChild(nameText);
            template.appendChild(nameInput);
            
            distrInput = document.createElement('div');
            distrInput.classList.add('one_inpute_place');
            distrLable = document.createElement('lable');
            distrLable.textContent = 'Распределение';
            distrLable.setAttribute("for", "distribution");
            distrInput.appendChild(distrLable);
            distrText = document.createElement('select');
            distrText.setAttribute("id", "distribution");
            distrText.classList.add('input');
            option1 = document.createElement('option');
            option1.textContent = 'PERT';
            option1.setAttribute("value", "pert");
            distrText.appendChild(option1);
            option2 = document.createElement('option');
            option2.textContent = 'Нормальное';
            option2.setAttribute("value", "normal");
            distrText.appendChild(option2);
            option3 = document.createElement('option');
            option3.textContent = 'Биноминальное';
            option3.setAttribute("value", "binomial");
            distrText.appendChild(option3);
            option4 = document.createElement('option');
            option4.textContent = 'Геометрическое';
            option4.setAttribute("value", "geometric");
            distrText.appendChild(option4);
            option5 = document.createElement('option');
            option5.textContent = 'Гипергеометрическое';
            option5.setAttribute("value", "hypergeometric");
            distrText.appendChild(option5);
            option6 = document.createElement('option');
            option6.textContent = 'Распределение Пуассона';
            option6.setAttribute("value", "poisson");
            distrText.appendChild(option6);
            option7 = document.createElement('option');
            option7.textContent = 'Равномерное';
            option7.setAttribute("value", "uniform");
            distrText.appendChild(option7);
            option8 = document.createElement('option');
            option8.textContent = 'Логнормальное';
            option8.setAttribute("value", "lognormal");
            distrText.appendChild(option8);
            option9 = document.createElement('option');
            option9.textContent = 'Треугольное';
            option9.setAttribute("value", "triangular");
            distrText.appendChild(option9);
            option10 = document.createElement('option');
            option10.textContent = 'Распределение Вейбулла';
            option10.setAttribute("value", "weibull");
            distrText.appendChild(option10);
            option11 = document.createElement('option');
            option11.textContent = 'Константа';
            option11.setAttribute("value", "const");
            distrText.appendChild(option11);
            option12 = document.createElement('option');
            option12.textContent = 'Формула';
            option12.setAttribute("value", "equation");
            distrText.appendChild(option12);
            distrInput.appendChild(distrText);
            template.appendChild(distrInput);

            firstCustomInput = document.createElement('div');
            firstCustomInput.classList.add('first_custom__input');
            firstCustomLable = document.createElement('lable');
            firstCustomLable.textContent = 'Минимум';
            firstCustomLable.setAttribute("for", "first");
            firstCustomLable.setAttribute("id", "firstLabel");
            firstCustomLable.classList.add('first__label');
            firstCustomInput.appendChild(firstCustomLable);
            firstCustomText = document.createElement('input');
            firstCustomText.setAttribute("id", "first");
            firstCustomText.setAttribute("type", "number");
            firstCustomText.classList.add('input');
            firstCustomInput.appendChild(firstCustomText);
            template.appendChild(firstCustomInput);

            secondCustomInput = document.createElement('div');
            secondCustomInput.classList.add('second_custom__input');
            secondCustomLable = document.createElement('lable');
            secondCustomLable.textContent = 'Ожидаемое';
            secondCustomLable.setAttribute("for", "second");
            secondCustomLable.setAttribute("id", "secondLabel");
            secondCustomLable.classList.add('second__label');
            secondCustomInput.appendChild(secondCustomLable);
            secondCustomText = document.createElement('input');
            secondCustomText.setAttribute("id", "second");
            secondCustomText.setAttribute("type", "number");
            secondCustomText.classList.add('input');
            secondCustomInput.appendChild(secondCustomText);
            template.appendChild(secondCustomInput);

            optionCustomInput = document.createElement('div');
            optionCustomInput.classList.add('option_custom__input');
            optionCustomLable = document.createElement('lable');
            optionCustomLable.textContent = 'Максимум';
            optionCustomLable.setAttribute("for", "option");
            optionCustomLable.setAttribute("id", "optionLabel");
            optionCustomLable.classList.add('option__label');
            optionCustomInput.appendChild(optionCustomLable);
            optionCustomText = document.createElement('input');
            optionCustomText.setAttribute("id", "option");
            optionCustomText.setAttribute("type", "number");
            optionCustomText.classList.add('input');
            optionCustomInput.appendChild(optionCustomText);
            template.appendChild(optionCustomInput);

            return template
        };

        const addButton = document.getElementById("addButtonParam");
        addButton.addEventListener('click', () => {
            addButton.before(createInput())
        });
        
        
        
        return window.dash_clientside.no_update
    }
    """,
    Output("addButtonParam","id"),
    Input("addButtonParam","id"),
    )


app.clientside_callback(
    """

    function(id) {
        const paramForm = document.getElementById("mainForm");
        paramForm.addEventListener('click', (evt) => {
            if (evt.target.classList.contains('button__minus')) {
                evt.target.parentElement.remove()
            }
        });
        paramForm.addEventListener('change', (evt) => {
            const collectionInputs = evt.target.parentElement.parentElement.getElementsByTagName('input');
            const collectionLables = evt.target.parentElement.parentElement.getElementsByTagName('lable');
            const optionInput = collectionInputs.option
            const optionLabel = collectionLables.optionLabel
            const firstInput = collectionInputs.first
            const firstLabel = collectionLables.firstLabel
            const secondInput = collectionInputs.second
            const secondLabel = collectionLables.secondLabel
            if (evt.target.value === "normal") {
                optionInput.style.display="none"
                secondInput.style.display="block"
                optionLabel.style.display="none"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Среднее'
                secondLabel.textContent = 'SD'
            }else if (evt.target.value === "pert"){
                optionInput.style.display="block"
                secondInput.style.display="block"
                optionLabel.style.display="block"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Минимум'
                secondLabel.textContent = 'Ожидаемое'
                optionLabel.textContent = 'Максимум'
            }else if (evt.target.value === "binomial"){
                optionInput.style.display="none"
                secondInput.style.display="block"
                optionLabel.style.display="none"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Количество итераций'
                secondLabel.textContent = 'Вероятность успеха (единицы)'
            }else if (evt.target.value === "geometric"){
                optionInput.style.display="none"
                secondInput.style.display="none"
                optionLabel.style.display="none"
                secondLabel.style.display="none"
                firstLabel.textContent = 'Вероятность успеха (единицы)'

            }else if (evt.target.value === "hypergeometric"){
                optionInput.style.display="block"
                secondInput.style.display="block"
                optionLabel.style.display="block"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Количество хороших путей'
                secondLabel.textContent = 'Количество плохих путей'
                optionLabel.textContent = 'Количество проходов по путям без повторений (не больше суммы хороших и плохих)'

            }else if (evt.target.value === "poisson"){
                optionInput.style.display="none"
                secondInput.style.display="none"
                optionLabel.style.display="none"
                secondLabel.style.display="none"
                firstLabel.textContent = 'Лямбда'
            }else if (evt.target.value === "uniform"){
                optionInput.style.display="none"
                secondInput.style.display="block"
                optionLabel.style.display="none"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Минимум'
                secondLabel.textContent = 'Максимум'
            }else if (evt.target.value === "lognormal"){
                optionInput.style.display="none"
                secondInput.style.display="block"
                optionLabel.style.display="none"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Среднее'
                secondLabel.textContent = 'SD'
            }else if (evt.target.value === "triangular"){
                optionInput.style.display="block"
                secondInput.style.display="block"
                optionLabel.style.display="block"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Минимум'
                secondLabel.textContent = 'Ожидаемое'
                optionLabel.textContent = 'Максимум'

            }else if (evt.target.value === "weibull"){
                optionInput.style.display="none"
                secondInput.style.display="none"
                optionLabel.style.display="none"
                secondLabel.style.display="none"
                firstLabel.textContent = 'Модуль Вейбулла'
            }else if (evt.target.value === "const"){
                optionInput.style.display="none"
                secondInput.style.display="none"
                optionLabel.style.display="none"
                secondLabel.style.display="none"
                firstLabel.textContent = 'Значение'
            }else if (evt.target.value === "equation"){
                optionInput.style.display="block"
                secondInput.style.display="block"
                optionLabel.style.display="block"
                secondLabel.style.display="block"
                firstLabel.textContent = 'Значение'
                secondLabel.textContent = 'Нижний лимит'
                optionLabel.textContent = 'Верхний лимит'
            }
        });

        return window.dash_clientside.no_update
    }
    """,
    Output("mainForm","id"),
    Input("mainForm","id")
)

app.clientside_callback(
    """
    function(id) {
        function createInput() {
            template = document.createElement('div');
            template.classList.add('inpute__space');
            buttonMinus = document.createElement('div');
            buttonMinus.textContent = '-';
            buttonMinus.classList.add('button__minus');
            template.appendChild(buttonMinus);

            nameInputF = document.createElement('div');
            nameInputF.classList.add('one_inpute_place');
            nameLableF = document.createElement('lable');
            nameLableF.textContent = 'Параметр 1';
            nameLableF.setAttribute("for", "param1");
            nameInputF.appendChild(nameLableF);
            nameTextF = document.createElement('input');
            nameTextF.setAttribute("id", "param1");
            nameTextF.classList.add('input');
            nameInputF.appendChild(nameTextF);
            template.appendChild(nameInputF);

            nameInputS = document.createElement('div');
            nameInputS.classList.add('one_inpute_place');
            nameLableS = document.createElement('lable');
            nameLableS.textContent = 'Параметр 2';
            nameLableS.setAttribute("for", "param2");
            nameInputS.appendChild(nameLableS);
            nameTextS = document.createElement('input');
            nameTextS.setAttribute("id", "param2");
            nameTextS.classList.add('input');
            nameInputS.appendChild(nameTextS);
            template.appendChild(nameInputS);

            nameInputC = document.createElement('div');
            nameInputC.classList.add('one_inpute_place');
            nameLableC = document.createElement('lable');
            nameLableC.textContent = 'Корреляция [-1:1]';
            nameLableC.setAttribute("for", "paramCor");
            
            nameInputC.appendChild(nameLableC);
            nameTextC = document.createElement('input');
            nameTextC.setAttribute("id", "paramCor");
            nameTextC.setAttribute("type", "number");
            nameTextC.setAttribute("min", "-1");
            nameTextC.setAttribute("max", "1");
            nameTextC.classList.add('input');
            nameInputC.appendChild(nameTextC);
            template.appendChild(nameInputC);

            return template
        }

        const addButton = document.getElementById("addButtonCorr");
        addButton.addEventListener('click', () => {
            addButton.before(createInput())
        });

        return window.dash_clientside.no_update
        
    }
    """,
    Output("addButtonCorr","id"),
    Input("addButtonCorr","id")
)


app.clientside_callback(
    """
    function(id) {
        const preload = document.getElementById('preloader')
        const cont = document.getElementsByTagName('input')
        const opt = document.getElementsByTagName('select')
        preload.classList = ['preload_section']
        let temp = 'teaaaa'
        let allInputs = [];
        let allDistr = [];
        let allValues = [];
        for(x=0; x < cont.length; x++) {
            allInputs.push(cont[x].value);}
        for(x=0; x < opt.length; x++) {
            allDistr.push(opt[x].value);}
        allValues.push(allInputs[0]);
        let counter = 1
        for(x=0; x < allDistr.length; x++) {
            allValues.push(allDistr[x]);
            allValues.push(allInputs[counter]);
            allValues.push(allInputs[counter+1]);
            allValues.push(allInputs[counter+2]);
            allValues.push(allInputs[counter+3]);
            counter += 4
        }
         for(x=counter; x < allInputs.length; x++) {
            allValues.push(allInputs[x]);
        }
        temp = allValues.toString()
        return temp

    }
    """,
    Output("getComm","data"),
    Input("calculateButton","n_clicks"),
    prevent_initial_call=True
)

# window.dash_clientside.no_update

# @callback(
#     Output("visPlace","children"),
#     Input("getComm","data"),
#     prevent_initial_call=True
# )

@callback(
    Output("visPlace","children"),
    Output("preloader", 'className'),
    Input("getComm","data"),
    prevent_initial_call=True
)
def str_to_plot(string):
    df = pd.DataFrame(columns=['Тип', 'Параметр', 'Зн1', 'Зн2', 'Зн3'])
    list1 = string.split(',')
    n = int(list1[0])
    if n > 1000000:
        n = 1000000
    target = list1[-1]
    list2 = list1[1:-1]
    typ = ['normal', 'pert', 'binomial', 'geometric', 'hypergeometric', 'poisson', 'uniform', 'lognormal', 'triangular', 'weibull', 'const', 'equation']
    while len(list2)>2:
        if list2[0] in typ:
            arr = list2[0:5]
            list2 = list2[5:]
            df.loc[len(df.index)] = arr
        else:
            arr = list2[0:4]
            df.loc[len(df.index), 'Тип'] = 'correlation'
            df.loc[(len(df.index)-1), ['Параметр', 'Зн1', 'Зн2']] = arr
            list2 = list2[4:]
    df.loc[len(df.index), ['Тип', 'Параметр', 'Зн1']] = ['target', 'Целевой показатель', target]
    df['Зн2'] = np.where(df['Зн2']=='', np.nan, df['Зн2'])
    df['Зн3'] = np.where(df['Зн3']=='', np.nan, df['Зн3'])
    df['Зн1'] = df['Зн1'].apply(convert_to_float)
    df['Зн2'] = df['Зн2'].apply(convert_to_float)
    df['Зн3'] = df['Зн3'].apply(convert_to_float)

 # подготовка данных
    names = list(df[(df['Тип']!='target')&(df['Тип']!='correlation')]['Параметр'].values)
    df['Array'] = pd.Series(dtype='object')
    df['Array'] = df.apply(apply_func, args=(n,), axis=1)
    correlat = df[df['Тип']=='correlation'].reset_index(drop=True)
    param_df = df[(df['Тип']!='target')&(df['Тип']!='correlation')].reset_index(drop=True)

        # Создаю матрицу корреляции
    if 'correlation' in df['Тип'].unique():
        m_set = set(list(correlat['Параметр'].unique()) + list(correlat['Зн1'].unique()))
        m_size = len(m_set)
        params = list(m_set)
        m_corr = np.eye(m_size)
        m_set

        for _, row in correlat.iterrows():
            param1 = row['Параметр']
            param2 = row['Зн1']
            corr = row['Зн2']
            
            i = params.index(param1)
            j = params.index(param2)
            m_corr[i, j] = corr
            m_corr[j, i] = corr
        
        # Список параметров для корреляции
        corr_list = []
        for i in range(len(params)):
            a = param_df[param_df['Параметр']==params[i]]['Array'].values[0]
            corr_list.append(a) 

        # Сама корреляция
        corr_list_1 = correlate(corr_list, m_corr)

        # словарь с новыми значениями из corr_list_1
        update_dict = {param: corr_list_1[i] for i, param in enumerate(params)}
        # присваивание новых значений
        df.loc[(df['Параметр'].isin(update_dict.keys()))&(df['Тип']!='correlation'), 'Array'] = df['Параметр'].map(update_dict)

    #  расчет уравнений и таргета 
    df['Array'] = df.apply(apply_func_eq, args=(df, ), axis=1)
    for i in range(df[df['Тип']=='equation']['Тип'].count()):
        df['Array'] = df.apply(apply_func_eq_2, args=(df,), axis=1)
    df['Array'] = df.apply(apply_func_target, args=(df, ), axis=1)

    # разделение по типу параметров
    param_df = df[(df['Тип']!='size')&(df['Тип']!='target')&(df['Тип']!='correlation')].reset_index(drop=True)
    target = df[df['Тип']=='target'].reset_index(drop=True)
    # num_tg = target.loc[:, 'Параметр'].count()

    # Визуализация
    arr_all = target.loc[0, 'Array']
    


    kurtosis_c = html.P(children=[f'Коэффициент эксцесса:  {round(kurtosis(arr_all), 3)}'], className='article__text')    # print('Коэффициент эксцесса  ', kurtosis(arr_all))
    skew_c = html.P(children=[f'Коэффициент асимметрии:  {round(skew(arr_all), 3)}'], className='article__text')    # print('Коэффициент асимметрии', skew(arr_all))
    min_c = html.P(children=[f'Минимум: {round(min(arr_all), 3)}'], className='article__text')    # print('Минимум ', min(arr_all))
    max_c = html.P(children=[f'Максимум: {round(max(arr_all), 3)}'], className='article__text')    # print('Максимум', max(arr_all))
    mean_c = html.P(children=[f'Среднее: {round(np.mean(arr_all), 3)}'], className='article__text')    # print('Среднее ', np.mean(arr_all))
    std_c = html.P(children=[f'Стандартное отклонение: {round(np.std(arr_all), 3)}'], className='article__text')    # print('Стандартное отклонение ', np.std(arr_all))
    median_c = html.P(children=[f'Медиана:  {round(np.median(arr_all), 3)}'], className='article__text')    # print('Медиана ', np.median(arr_all))
    q05_c = html.P(children=[f'5%: {round(np.quantile(arr_all, 0.05), 3)}'], className='article__text')    # print('5%  ', np.quantile(arr_all, 0.05))
    q1_c = html.P(children=[f'10%: {round(np.quantile(arr_all, 0.1), 3)}'], className='article__text')    # print('10% ', np.quantile(arr_all, 0.1))
    q2_c = html.P(children=[f'20%: {round(np.quantile(arr_all, 0.2), 3)}'], className='article__text')    # print('20% ', np.quantile(arr_all, 0.2))
    q3_c = html.P(children=[f'30%: {round(np.quantile(arr_all, 0.3), 3)}'], className='article__text')    # print('30% ', np.quantile(arr_all, 0.3))
    q4_c = html.P(children=[f'40%: {round(np.quantile(arr_all, 0.4), 3)}'], className='article__text')    # print('40% ', np.quantile(arr_all, 0.4))
    q5_c = html.P(children=[f'50%: {round(np.quantile(arr_all, 0.5), 3)}'], className='article__text')    # print('50% ', np.quantile(arr_all, 0.5))
    q6_c = html.P(children=[f'60%: {round(np.quantile(arr_all, 0.6), 3)}'], className='article__text')    # print('60% ', np.quantile(arr_all, 0.6))
    q7_c = html.P(children=[f'70%: {round(np.quantile(arr_all, 0.7), 3)}'], className='article__text')    # print('70% ', np.quantile(arr_all, 0.7))
    q8_c = html.P(children=[f'80%: {round(np.quantile(arr_all, 0.8), 3)}'], className='article__text')    # print('80% ', np.quantile(arr_all, 0.8))
    q9_c = html.P(children=[f'90%: {round(np.quantile(arr_all, 0.9), 3)}'], className='article__text')    # print('90% ', np.quantile(arr_all, 0.9))
    q95_c = html.P(children=[f'95%: {round(np.quantile(arr_all, 0.95), 3)}'], className='article__text')    # print('95% ', np.quantile(arr_all, 0.95))
    stat_f = html.Div(children=[kurtosis_c, skew_c, min_c, max_c, mean_c, std_c, median_c])
    stat_s = html.Div(children=[q05_c, q1_c, q2_c, q3_c, q4_c, q5_c, q6_c, q7_c, q8_c, q9_c, q95_c])

    stat_div = html.Div(children=[stat_f, stat_s], className='stat_place')

    coef_list = []
    target_min_list = []
    target_max_list = []
    target_mean_list = []
    diff = []
    base = []

    for a in range(len(param_df['Array'])):
        i = param_df['Array'][a]
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
        
    sorted_idx = np.argsort(np.abs(diff))#[::-1]
    sorted_diff = np.array(diff)[sorted_idx]
    # sorted_min_target = np.array(target_min_list)[sorted_idx]
    # sorted_max_target = np.array(target_max_list)[sorted_idx]
    sorted_base = np.array(base)[sorted_idx]
    sorted_labels = [names[i] for i in sorted_idx]

    title_first = html.H3(children='Распределение целевого параметра', className='first__subtitle')
    title_second = html.H3(children='Диаграмма торнадо', className='subtitle')
    histPlot = html.Div(children=plot_first(arr_all, target), className='hist__plot', id='histPlot')
    tornadoPlot = html.Div(children=plot_second(arr_all, sorted_labels, sorted_diff, sorted_base), className='tornado__plot', id='tornadoPlot')
    divdiv = html.Div(children=[stat_div, title_first, histPlot, title_second, tornadoPlot], className = 'page', id = 'plots')

    return divdiv, ['preload_hide']

app.clientside_callback(
    '''
    function(id) {
        const plots = document.getElementById('plots')
        plots.scrollIntoView()
        return window.dash_clientside.no_update
    }
    ''',
    Output("preloader","id"),
    Input("preloader","className"),
    prevent_initial_call=True
)

# @callback(
#     Output("visPlace","id"),
#     Input("visPlace","id"),
#     State("preloader", 'style')
#     prevent_initial_call=True
# )
# def setdysplay(n, preload_s):
#     preload_s= {'display': 'none'} 

# @app.callback(
#     Output("testDiv","children"),
#     [Input('dim', 'data')]
# )
# def someF():
#     print('yep')


# @callback(
#     Output('graph-content', 'figure'),
#     Input('dropdown-selection', 'value')
# )

if __name__ == '__main__':
    
    # закомментировать перед деплоем
    # app.run(debug=False)

    # раскомментировать перед деплоем
    context = ('/etc/letsencrypt/live/wf-onco.ru/fullchain.pem', '/etc/letsencrypt/live/wf-onco.ru/privkey.pem')
    app.run(host='wf-onco.ru', ssl_context=context)