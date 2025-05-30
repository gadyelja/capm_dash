import streamlit as st 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import plotly.express as px
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import GLSAR
from functools import reduce

from io import BytesIO
import pandas as pd
import plotly.io as pio


st.set_page_config(layout="wide")

st.markdown("""
    <h1 style="text-align: center;">LMIC DASHBOARD</h1>
""", unsafe_allow_html=True) # Título

# Importar bases 
file1 = 'list_of_countries_index.xlsx'
file2 = 'historical indexes.xlsx'
population = pd.read_excel(file1, sheet_name=8)
lmic = pd.read_excel(file1 , sheet_name= 9)
mark_cap = pd.read_excel(file1 , sheet_name= 7)
# returns = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name = 3 )
famafrench = pd.read_excel(file1, sheet_name=  10)
msci = pd.read_excel(file1, sheet_name=  11)
prices = pd.read_excel(file1, sheet_name=  1)
rate_ex = pd.read_excel(file1, sheet_name=  5)

tab1, tab2, tab3 = st.tabs(["🌎 Map and groups", '📊 Regresions', '⏬Export data'])

with tab1: 
    # filtrar por L, LM, UM, HM
    categorias = {
        'L' : 'Low Income Country',
        'LM': 'Low-Middle Income Country',
        'UM': 'Upper-Middle Income Country',
        'H' : 'High Income Country' 
    }

    # Filtrar los df
    mark_cap.set_index('Country', inplace = True)
    mark_cap = mark_cap.iloc[:, 0:1]
    prices.set_index('Country', inplace = True)
    rate_ex.set_index('Country', inplace = True)
    population.set_index('Country', inplace=True)
    lmic.set_index('Country', inplace = True)

    prices = prices.iloc[:, 1:].dropna(how = 'all')
    rate_ex = rate_ex.iloc[:, 1:].dropna(how = 'all')

    # Calcular retornos en USD
    usd_price = prices / rate_ex
    returns = usd_price.pct_change(axis = 1)

    common_countries = prices.index.intersection(mark_cap.index).intersection(population.index).intersection(lmic.index)
    mark_cap = mark_cap.loc[common_countries]
    returns = returns.loc[common_countries]
    population = population.loc[common_countries]
    lmic = lmic.loc[common_countries]


    col1, col2 = st.columns(2)


    with col2: 
        opciones = st.multiselect('Choose the Income level: ', options = list(categorias.values()), default=list(categorias.values()))
        categorias_invertido = {v: k for k, v in categorias.items()}
        # Obtener códigos correspondientes (L, LM, UM, H)
        codigos = [categorias_invertido[op] for op in opciones]
        # Filtrar DataFrame
        filtrado = lmic[lmic["LMIC"].isin(codigos)]
        # Mostrar resultados
        st.dataframe(filtrado)
    
    common_countries1 = prices.index.intersection(mark_cap.index).intersection(population.index).intersection(filtrado.index)
    mark_cap = mark_cap.loc[common_countries1]
    returns = returns.loc[common_countries1]
    population = population.loc[common_countries1]
    lmic = lmic.loc[common_countries1]



    # Entrada de rango
    with col1: 
        st.subheader('Choose range for groups of countries: ')

        option = st.radio("segmentation method :", ['Terciles',"Quartiles", "Customized"])
        if option == "Terciles":
            percentiles = np.percentile(population['Population'], [33, 50, 66])
            group1_max, group2_max, group3_max = percentiles
        elif option == "Quartiles":
            percentiles = np.percentile(population['Population'], [25, 50, 75])
            group1_max, group2_max, group3_max = percentiles[:3]
        else:
            # 📌 OPCIÓN 2: Entrada manual
            st.subheader("Choose the range: ")
            group1_max = st.number_input("Upper range for lowest population countries:", min_value=0, value = 3_000_000)
            group2_max = st.number_input("Upper range for lower-middle population countries:", min_value=group1_max, value= 10_000_000)
            group3_max = st.number_input("Upper range for highest population countries:", min_value=group2_max, value= 30_000_000)
        
        st.write(f'Lowest population group: 0 - {group1_max}')
        st.write(f'Lower middle population group: {group1_max} - {group2_max}')
        st.write(f'Upper middle population group: {group2_max} - {group3_max}')
        st.write(f'Highest population group: > {group3_max}')


    # 📌 Definir los grupos basados en la opción seleccionada
    group1 = population[population["Population"] <= group1_max]
    group2 = population[(population["Population"] > group1_max) & (population["Population"] <= group2_max)]
    group3 = population[(population["Population"] > group2_max)  & (population["Population"] <= group3_max)]
    group4 = population[(population["Population"] > group3_max)]

    # 📌 Función para asignar grupos según la población
    def assign_group(population):
        if population < group1_max:
            return "Group 1: Lowest"
        elif population < group2_max:
            return "Group 2: Lower middle"
        elif population < group3_max:
            return "Group 3: Upper middle"
        else:
            return "Group 4: Highest"

    # Asignar grupos
    population["Group"] = population["Population"].apply(assign_group)

    # 📌 Definir colores para los 4 grupos
    color_map = {
        "Group 1: Lowest": "lightblue",
        "Group 2: Lower middle": "green",
        "Group 3: Upper medium": "orange",
        "Group 4: Highest": "red"
    }

    # 📌 Crear el mapa con Plotly
    fig = px.choropleth(
        population, 
        locations="Suffix",  # Código de país ISO 3
        color="Group",
        title="Distribution of countries by population: ",
        color_discrete_map=color_map,
        projection="natural earth"
    )

    fig.update_layout(height=700, width=1200)
    st.plotly_chart(fig)    

    # Paises por grupo
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.write('Lowest population countries')
        st.dataframe(group1, use_container_width=True)
        st.write('Number of countries in group 1: ', group1['Suffix'].count())

    with col2: 
        st.write('Lower middle population countries')
        st.dataframe(group2, use_container_width=True)
        st.write('Number of countries in group 2: ', group2['Suffix'].count())
        
    with col3: 
        st.write('Upper middle population countries')
        st.dataframe(group3, use_container_width=True) 
        st.write('Number of countries in group 3: ', group3['Suffix'].count())

    with col4: 
        st.write('Highest population countries')
        st.dataframe(group4, use_container_width=True) 
        st.write('Number of countries in group 4: ', group4['Suffix'].count())
    
with tab1:    
    # Winsor2
    def winsorize_series(series, lower_percentile=1, upper_percentile=99):
        lower = np.percentile(series, lower_percentile)
        upper = np.percentile(series, upper_percentile)
        return np.clip(series, lower, upper)


    returns = returns.apply(lambda row: winsorize_series(row), axis=1) 

    # Filtrar retornos y market cap por grupo
    price1 = usd_price[usd_price.index.isin(group1.index)]
    price2 = usd_price[usd_price.index.isin(group2.index)]
    price3 = usd_price[usd_price.index.isin(group3.index)]
    price4 = usd_price[usd_price.index.isin(group4.index)]

    return_1 = returns[returns.index.isin(group1.index)]
    return_2 = returns[returns.index.isin(group2.index)]
    return_3 = returns[returns.index.isin(group3.index)]
    return_4 = returns[returns.index.isin(group4.index)]

    mark_cap1 = mark_cap[mark_cap.index.isin(group1.index)]
    mark_cap2 = mark_cap[mark_cap.index.isin(group2.index)]
    mark_cap3 = mark_cap[mark_cap.index.isin(group3.index)]
    mark_cap4 = mark_cap[mark_cap.index.isin(group4.index)]
    ################################################################################################################################################################################

    def plot_group_prices(group_prices, group_name):
        df = group_prices.transpose().reset_index()
        df = df.rename(columns={'index': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])

        df_melted = df.melt(id_vars='Date', var_name='País', value_name='Precio USD')

        fig = px.line(df_melted, x='Date', y='Precio USD', color='País', title=f'{group_name} - Prices in USD',
                    labels= {'Date' : 'Date', 'Precio USD' : 'Prices in USD', 'País' : 'Country'})
        
        fig.update_layout(height=500, width=1000)
        st.plotly_chart(fig)


    def plot_group_returns(group_returns, group_name):
        df = group_returns.transpose().reset_index()
        df = df.rename(columns={'index': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])

        df_melted = df.melt(id_vars='Date', var_name='País', value_name='Retorno')

        fig = px.line(df_melted, x='Date', y='Retorno', color='País', title=f'{group_name} - Returns',
                    labels = {'Date': 'Date', 'Retorno': 'Returns', 'País': 'Country'})
        fig.update_layout(height=500, width=1000)
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('📉 Graph of prices by group')
        plot_group_prices(price1, "Lowest population countries")
        plot_group_prices(price2, "Lower middle population countries")
        plot_group_prices(price3, "Upper middle population countries")
        plot_group_prices(price4, "Highest population countries")

    with col2:
        st.subheader("📉 Graphs of retuns by group")
        plot_group_returns(return_1, "Lowest population countries")
        plot_group_returns(return_2, "Lower middle population countries")
        plot_group_returns(return_3, "Upper middle population countries")
        plot_group_returns(return_4, "Highest population countries")


################################################################################################################################################################################
with tab2:
    
    # Organizar df's
    group_1 = return_1.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
    group_1['Date'] = pd.to_datetime(group_1['Date'])
    group_1 = group_1.merge(mark_cap1, on="Country", how="left")
    group_1 = group_1.merge(famafrench, on = 'Date', how = 'left')
    group_1 = group_1.merge(msci, on = 'Date', how = 'left')
    group_1 = group_1[group_1["Returns"].notna()].copy()
    sum_mark_cap1 = group_1.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
    group_1 = group_1.merge(sum_mark_cap1, on="Date", how="left")
    group_1["wr"] = group_1["Returns"] * (group_1["mark_cap"] )
    swr = group_1.groupby('Date')['wr'].sum().reset_index()
    group_1 = group_1.merge(swr, on = 'Date', how = 'left')
    group_1 = group_1.drop_duplicates(subset='Date').reset_index(drop=True)
    group_1['Weighted_Returns'] = group_1['wr_y'] / group_1['sum_mark_cap']
    group_1 = group_1[group_1["msci"].notna()].copy()
    group_1 = group_1[group_1["Mkt-RF"].notna()].copy()
    group_1['Mean_Returns'] = group_1.groupby('Date')['Returns'].transform('mean')
    group_1['Re-Rf'] = group_1['Mean_Returns'] - group_1['RF']
    group_1['Wr-Rf'] = group_1['Weighted_Returns'] - group_1['RF']
    group_1 = group_1.drop('Country', axis=1)

    group_2 = return_2.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
    group_2['Date'] = pd.to_datetime(group_2['Date'])
    group_2 = group_2.merge(mark_cap2, on="Country", how="left")
    group_2 = group_2.merge(famafrench, on = 'Date', how = 'left')
    group_2 = group_2.merge(msci, on = 'Date', how = 'left')
    group_2 = group_2[group_2["Returns"].notna()].copy()
    sum_mark_cap2 = group_2.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
    group_2 = group_2.merge(sum_mark_cap2, on="Date", how="left")
    group_2["wr"] = group_2["Returns"] * (group_2["mark_cap"] )
    swr2 = group_2.groupby('Date')['wr'].sum().reset_index()
    group_2 = group_2.merge(swr2, on = 'Date', how = 'left')
    group_2 = group_2.drop_duplicates(subset='Date').reset_index(drop=True)
    group_2['Weighted_Returns'] = group_2['wr_y'] / group_2['sum_mark_cap']
    group_2 = group_2[group_2["msci"].notna()].copy()
    group_2 = group_2[group_2["Mkt-RF"].notna()].copy()
    group_2['Mean_Returns'] = group_2.groupby('Date')['Returns'].transform('mean')
    group_2['Re-Rf'] = group_2['Mean_Returns'] - group_2['RF']
    group_2['Wr-Rf'] = group_2['Weighted_Returns'] - group_2['RF']
    group_2 = group_2.drop('Country', axis=1)


    group_3 = return_3.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
    group_3['Date'] = pd.to_datetime(group_3['Date'])
    group_3 = group_3.merge(mark_cap3, on="Country", how="left")
    group_3 = group_3.merge(famafrench, on = 'Date', how = 'left')
    group_3 = group_3.merge(msci, on = 'Date', how = 'left')
    group_3 = group_3[group_3["Returns"].notna()].copy()
    sum_mark_cap3 = group_3.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
    group_3 = group_3.merge(sum_mark_cap3, on="Date", how="left")
    group_3["wr"] = group_3["Returns"] * (group_3["mark_cap"] )
    swr3 = group_3.groupby('Date')['wr'].sum().reset_index()
    group_3 = group_3.merge(swr3, on = 'Date', how = 'left')
    group_3 = group_3.drop_duplicates(subset='Date').reset_index(drop=True)
    group_3['Weighted_Returns'] = group_3['wr_y'] / group_3['sum_mark_cap']
    group_3 = group_3[group_3["msci"].notna()].copy()
    group_3 = group_3[group_3["Mkt-RF"].notna()].copy()
    group_3['Mean_Returns'] = group_3.groupby('Date')['Returns'].transform('mean')
    group_3['Re-Rf'] = group_3['Mean_Returns'] - group_3['RF']
    group_3['Wr-Rf'] = group_3['Weighted_Returns'] - group_3['RF']
    group_3 = group_3.drop('Country', axis=1)

    group_4 = return_4.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
    group_4['Date'] = pd.to_datetime(group_4['Date'])
    group_4 = group_4.merge(mark_cap4, on="Country", how="left")
    group_4 = group_4.merge(famafrench, on = 'Date', how = 'left')
    group_4 = group_4.merge(msci, on = 'Date', how = 'left')
    group_4 = group_4[group_4["Returns"].notna()].copy()
    sum_mark_cap4 = group_4.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
    group_4 = group_4.merge(sum_mark_cap4, on="Date", how="left")
    group_4["wr"] = group_4["Returns"] * (group_4["mark_cap"] )
    swr4 = group_4.groupby('Date')['wr'].sum().reset_index()
    group_4 = group_4.merge(swr4, on = 'Date', how = 'left')
    group_4 = group_4.drop_duplicates(subset='Date').reset_index(drop=True)
    group_4['Weighted_Returns'] = group_4['wr_y'] / group_4['sum_mark_cap']
    group_4 = group_4[group_4["msci"].notna()].copy()
    group_4 = group_4[group_4["Mkt-RF"].notna()].copy()
    group_4['Mean_Returns'] = group_4.groupby('Date')['Returns'].transform('mean')
    group_4['Re-Rf'] = group_4['Mean_Returns'] - group_4['RF']
    group_4['Wr-Rf'] = group_4['Weighted_Returns'] - group_4['RF']
    group_4 = group_4.drop('Country', axis=1)

    #######################################################################################################################
    # Simple model weighted
    # Group 1
    y9 = group_1['Wr-Rf'] * 100 
    x9 = group_1["Mkt-RF"]  * 100 
    x9 = sm.add_constant(x9)
    model9 = sm.OLS(y9, x9).fit() 

    # Group 2
    y10 = group_2['Wr-Rf'] * 100
    x10 = group_2["Mkt-RF"]  * 100
    x10 = sm.add_constant(x10)
    model10 = sm.OLS(y10, x10).fit() 

    # Group 3
    y11 = group_3['Wr-Rf'] * 100
    x11 = group_3["Mkt-RF"]  * 100
    x11 = sm.add_constant(x11)
    model11 = sm.OLS(y11, x11).fit()

    # Grouo 4
    y12 = group_4['Wr-Rf'] * 100
    x12 = group_4["Mkt-RF"]  * 100
    x12 = sm.add_constant(x12)
    model12 = sm.OLS(y12, x12).fit()

    #######################################################################################################################
    # Multifactorial model  weighted
    # group 1
    y5 = group_1['Wr-Rf'] *100
    msci1 = group_1['msci'] *100
    x5 = group_1["Mkt-RF"]  * 100
    x5 = pd.concat([x5, msci1], axis = 1)
    x5 = sm.add_constant(x5)
    model5 = sm.OLS(y5, x5).fit() 

    # group 2
    y6 = group_2['Wr-Rf'] *100
    msci2 = group_2['msci'] *100
    x6 = group_2["Mkt-RF"]  * 100
    x6 = pd.concat([x6, msci2], axis = 1)
    x6 = sm.add_constant(x6)
    model6 = sm.OLS(y6, x6).fit() 

    # group 3
    y7 = group_3['Wr-Rf'] *100
    msci3 = group_3['msci'] *100
    x7 = group_3["Mkt-RF"]  * 100
    x7 = pd.concat([x7, msci3], axis = 1)
    x7 = sm.add_constant(x7)
    model7 = sm.OLS(y7, x7).fit() 

    # group 4
    y8 = group_4['Wr-Rf'] * 100
    msci4 = group_4['msci'] * 100
    x8 = group_4["Mkt-RF"]  * 100
    x8 = pd.concat([x8, msci3], axis = 1)
    x8 = sm.add_constant(x8)
    model8 = sm.OLS(y8, x8).fit() 

    ####################################################################################################

    # Funcion para extraer resultados
    def extract_model_summary(model):
        """Extrae coeficientes, errores estándar, p-valores, R² y otras métricas."""
        summary_df = pd.DataFrame({
            "Coefficients": model.params,
            # "Standard Errors": model.bse,
            # "p-values": model.pvalues,
            # "Lower range 95%": model.conf_int()[0],
            # "Upper range 95%": model.conf_int()[1]
            
        })
        summary_df.loc["R²", "Coefficients"] = model.rsquared
        # summary_df.loc["R² Adjusted", "Coefficients"] = model.rsquared_adj
        # summary_df.loc["F-Statistical", "Coefficients"] = model.fvalue
        return summary_df
    
    def add_tea_row(df, model, label="Yearly Excess Return(%)"):
        alpha_mensual = model.params['const']
        alpha_tea = ((1 + alpha_mensual / 30) ** 12) - 1
        alpha_tea_porcentaje = round(alpha_tea * 100, 2)  # Redondeado a 2 decimales
        tea_row = pd.DataFrame({df.columns[0]: [alpha_tea_porcentaje]}, index=[label])
        return pd.concat([df, tea_row])

    # Lista de modelos y nombres de grupo
    models = [model9, model10, model11, model12]
    group_names = ['Lowest Population', 'Lower-Middel Population', 'Upper-Medium Population', 'Highest Population']

    # Extraer, renombrar y combinar
    summary_tables = []

    for model, name in zip(models, group_names):
        summary = extract_model_summary(model)
        summary.columns = [f"{name}" for col in summary.columns]  # Renombra columnas por grupo
        summary = add_tea_row(summary, model) 
        summary_tables.append(summary)

    # Combinar horizontalmente por índice (las variables)
    final_table = pd.concat(summary_tables, axis=1)

    models1 = [model5, model6, model7, model8]
    group_names1 = ['Lowest Population', 'Lower-Middel Population', 'Upper-Medium Population', 'Highest Population']

    # Extraer, renombrar y combinar
    summary_tables1 = []

    for model, name in zip(models1, group_names1):
        summary1 = extract_model_summary(model)
        summary1.columns = [f"{name}" for col in summary.columns]  # Renombra columnas por grupo
        summary1 = add_tea_row(summary1, model) 
        summary_tables1.append(summary1)

    # Combinar horizontalmente por índice (las variables)
    final_table1 = pd.concat(summary_tables1, axis=1)

    
    st.markdown("""
    <h1 style="text-align: center;">SIMPLE MODEL</h1>
    """, unsafe_allow_html=True) 

    
    st.dataframe(final_table.style.format("{:.3f}"), use_container_width=True)
    
    st.markdown("""
    <h1 style="text-align: center;">MULTIFACTORIAL MODEL</h1>
    """, unsafe_allow_html=True) 
    
    st.dataframe(final_table1.style.format("{:.3f}"), use_container_width=True)


############################################################################################################################
# Correlaciones

# Importar base
prices1 = pd.read_excel(file2, sheet_name=0)
rateex1 = pd.read_excel(file2, sheet_name=2)
msci = pd.read_excel(file2, sheet_name=1)
msci_d =  pd.read_excel(file2, sheet_name= 3)
lmic1 = pd.read_excel(file1 , sheet_name= 9)
mark_cap11 = pd.read_excel(file1, sheet_name= 7)
population11 = pd.read_excel(file1, sheet_name=8)

mark_cap11.set_index('Country', inplace = True)
mark_cap11 = mark_cap11.iloc[:, 0:1]

# Calcular retornos
prices1.set_index('Country', inplace = True)
rateex1.set_index('Country', inplace = True)
lmic1.set_index('Country', inplace = True)
population11.set_index('Country', inplace = True)

rateusd = prices1 / rateex1
returns1 = rateusd.pct_change(axis = 1)

# Common countries
common_countries3 = returns1.index.intersection(returns1.index).intersection(mark_cap.index).intersection(lmic1.index).intersection(population11.index)
returns1  = returns1.loc[common_countries3]
mark_cap11  = mark_cap11.loc[common_countries3]
lmic1 = lmic1.loc[common_countries3]
population11 = population11.loc[common_countries3]

# common_countries4 = retunrns1.indexintersection(group_1.index)


# Eliminar paises vacios
returns1 = returns1.iloc[:, 1:].dropna(how = 'all')
mark_cap11 = mark_cap11.dropna(how = 'all')

# Filtrar bases segun nivel de ingreso
returns1 = returns1.merge(lmic1, on = 'Country', how = 'left')
mark_cap11 = mark_cap11.merge(lmic1, on = 'Country', how = 'left')

h_r = returns1[returns1['LMIC'] == 'H']
um_r = returns1[returns1['LMIC'] == 'UM']
lm_r = returns1[returns1['LMIC'] == 'LM']
l_r = returns1[returns1['LMIC'] == 'L']

h_mc = mark_cap11[mark_cap11['LMIC'] == 'H']
um_mc = mark_cap11[mark_cap11['LMIC'] == 'UM']
lm_mc = mark_cap11[mark_cap11['LMIC'] == 'LM']
l_mc = mark_cap11[mark_cap11['LMIC'] == 'L']

h_r = h_r.iloc[:, :-2]
um_r = um_r.iloc[:, :-2]
lm_r = lm_r.iloc[:, :-2]
l_r = l_r.iloc[:, :-2]

h_mc = h_mc.iloc[:, :-2]
um_mc = um_mc.iloc[:, :-2]
lm_mc = lm_mc.iloc[:, :-2]
l_mc = l_mc.iloc[:, :-2]

# Dividir por poblacion
# baja poblacion
st_mc = mark_cap11[population11['Population'] <=group1_max] 
st_r = returns1[population11['Population'] <= group1_max]

st_mc = st_mc.iloc[:, :-2]
st_r = st_r.iloc[:, :-2]

# Ponderar retornos
# Lower medium income countries
lmwr = lm_r.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lmwr['Date'] = pd.to_datetime(lmwr['Date'])
lmwr = lmwr.merge(lm_mc, on = 'Country', how = 'left')
lmwr = lmwr[lmwr['Returns'].notna()].copy()
lmsmc = lmwr.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lmwr = lmwr.merge(lmsmc, on = 'Date', how = 'left')
lmwr['wr'] = lmwr['Returns'] * (lmwr['mark_cap'])
lmswr = lmwr.groupby('Date')['wr'].sum().reset_index()
lmwr = lmwr.merge(lmswr, on = 'Date', how = 'left')
lmwr = lmwr.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lmwr['lmwr'] = lmwr['wr_y'] / lmwr['sum_mark_cap']

# Small states
st1 = st_r.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
st1['Date'] = pd.to_datetime(st1['Date'])
st1 = st1.merge(st_mc, on = 'Country', how = 'left')
st1 = st1[st1['Returns'].notna()].copy()
stsmc = st1.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
st1 = st1.merge(stsmc, on = 'Date', how = 'left')
st1['wr'] = st1['Returns'] * (st1['mark_cap'])
stwr = st1.groupby('Date')['wr'].sum().reset_index()
st1 = st1.merge(stwr, on = 'Date', how = 'left')
st1 = st1.drop_duplicates(subset = 'Date'). reset_index(drop = True)
st1['stwr'] = st1['wr_y'] / st1['sum_mark_cap']


####################################################################################################
# Correlacion movil
# msci developed - msci emerging
corr_df1 = pd.merge(msci_d[['Date', 'msci_global']], msci[['Date', 'msci_eme']], on = 'Date', how = 'left')

# msci developed - lower middle income countries
corr_df2 = pd.merge(msci_d[['Date', 'msci_global']], lmwr[['Date', 'lmwr']], on = 'Date', how = 'left')

# msci developed - small states
corr_df3 = pd.merge(msci_d[['Date', 'msci_global']], st1[['Date', 'stwr']], on = 'Date', how = 'left')

# Calcular cada correlacion movil
corr1 = corr_df1['msci_global'].rolling(window=36).corr(corr_df1['msci_eme'])
corr2 = corr_df2['msci_global'].rolling(window=36).corr(corr_df2['lmwr'])
corr3 = corr_df3['msci_global'].rolling(window=36).corr(corr_df3['stwr'])

# Arreglar dataframes
corr1 = corr1.rename('corr') # Asignar nombre a la serie
corr2 = corr2.rename('corr')
corr3 = corr3.rename('corr')

# añadir la fecha
corr1 = pd.merge(corr_df1[['Date']], corr1, left_index=True, right_index=True, how='left')
corr2 = pd.merge(corr_df2[['Date']], corr2, left_index=True, right_index=True, how='left')
corr3 = pd.merge(corr_df3[['Date']], corr3, left_index=True, right_index=True, how='left')

# Arreglar los df para el grafico
corr1.columns = ['Date', 'corr']
corr1['corr'] = pd.to_numeric(corr1['corr'], errors='coerce')
corr1['Date'] = pd.to_datetime(corr1['Date'], errors='coerce')

corr2.columns = ['Date', 'corr']
corr2['corr'] = pd.to_numeric(corr2['corr'], errors='coerce')
corr2['Date'] = pd.to_datetime(corr2['Date'], errors='coerce')

corr3.columns = ['Date', 'corr']
corr3['corr'] = pd.to_numeric(corr3['corr'], errors='coerce')
corr3['Date'] = pd.to_datetime(corr3['Date'], errors='coerce')

with tab1:
    
    st.markdown("""
        <h1 style="text-align: center;">CORRELATION IN A 3 YEAR ROLLING WINDOW</h1>
    """, unsafe_allow_html=True) 
    
    fig = px.line(corr1, x='Date', y='corr', title='Correlation: MSCI Developed Markets - MSCI Emerging Markets', labels={'corr': 'Correlation'})
    fig.update_traces(line_color='blue')
    fig.update_layout(
    template='plotly_white',
    title_font_size=20,
    font=dict(family="Arial", size=12),
    title_x=0.32
    )
    st.plotly_chart(fig)
    
    fig1 = px.line(corr2, x='Date', y='corr', title='Correlation: MSCI Developed Markets - Low Medium Income Countries', labels={'corr': 'Correlation'})
    fig1.update_traces(line_color='green')
    fig1.update_layout(
    template='plotly_white',
    title_font_size=20,
    font=dict(family="Arial", size=12),
    title_x=0.28
    )
    st.plotly_chart(fig1)

    fig2 = px.line(corr3, x='Date', y='corr', title='Correlaction: MSCI Developed Markets - Small States', labels={'corr': 'Correlation'})
    fig2.update_traces(line_color='red')
    fig2.update_layout(
    template='plotly_white',
    title_font_size=20,
    font=dict(family="Arial", size=12),
    title_x=0.32
    )
    st.plotly_chart(fig2)

    
# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\capm_dash.py"
