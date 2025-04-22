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


st.set_page_config(layout="wide")

st.markdown("""
    <h1 style="text-align: center;">LMIC DASHBOARD</h1>
""", unsafe_allow_html=True) # Título

# Importar bases 
population = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=8)
lmic = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 9)
mark_cap = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 7)
# returns = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name = 3 )
famafrench = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  10)
msci = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  11)
prices = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  1)
rate_ex = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  5)


tab1, tab2, tab3, tab4 = st.tabs(["🌎 Map and groups", "📈 Graphs", '📊 Regresions', '🔗 Correlations'])

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

        option = st.radio("segmentation method :", ["Terciles", "Quartiles", "Customized"])
        if option == "Terciles":
            percentiles = np.percentile(population['Population'], [25, 50, 75])
            group1_max, group2_max, group3_max = percentiles
        elif option == "Quartiles":
            percentiles = np.percentile(population['Population'], [20, 40, 60, 80])
            group1_max, group2_max, group3_max = percentiles[:3]
        else:
            # 📌 OPCIÓN 2: Entrada manual
            st.subheader("Choose the range: ")
            group1_max = st.number_input("Upper range for group 1:", min_value=0, value = 3_000_000)
            group2_max = st.number_input("Upper range for group 2:", min_value=group1_max, value= 10_000_000)
            group3_max = st.number_input("Upper range for group 3:", min_value=group2_max, value= 30_000_000)
        


    # 📌 Definir los grupos basados en la opción seleccionada
    group1 = population[population["Population"] <= group1_max]
    group2 = population[(population["Population"] > group1_max) & (population["Population"] <= group2_max)]
    group3 = population[(population["Population"] > group2_max)  & (population["Population"] <= group3_max)]
    group4 = population[(population["Population"] > group3_max)]

    # 📌 Función para asignar grupos según la población
    def assign_group(population):
        if population < group1_max:
            return "Group 1: Very Low"
        elif population < group2_max:
            return "Group 2: Low"
        elif population < group3_max:
            return "Group 3: Medium"
        else:
            return "Group 4: High"

    # Asignar grupos
    population["Group"] = population["Population"].apply(assign_group)

    # 📌 Definir colores para los 4 grupos
    color_map = {
        "Group 1: Very Low": "lightblue",
        "Group 2: Low": "green",
        "Group 3: Medium": "orange",
        "Group 4: High": "red"
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
        st.write('Countries of group 1: ')
        st.dataframe(group1, use_container_width=True)
        st.write('Number of countries in group 1: ', group1['Suffix'].count())

    with col2: 
        st.write('Countries of group 2: ')
        st.dataframe(group2, use_container_width=True)
        st.write('Number of countries in group 2: ', group2['Suffix'].count())
        
    with col3: 
        st.write('Countries of group 3: ')
        st.dataframe(group3, use_container_width=True) 
        st.write('Number of countries in group 3: ', group3['Suffix'].count())

    with col4: 
        st.write('Countries of group 4: ')
        st.dataframe(group4, use_container_width=True) 
        st.write('Number of countries in group 4: ', group4['Suffix'].count())
    
with tab2:    
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

        fig = px.line(df_melted, x='Date', y='Retorno', color='País', title=f'{group_name} - Returns by group',
                    labels = {'Date': 'Date', 'Retorno': 'Returns', 'País': 'Country'})
        fig.update_layout(height=500, width=1000)
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('📉 Graph of prices by group')
        plot_group_prices(price1, "Group 1")
        plot_group_prices(price2, "Group 2")
        plot_group_prices(price3, "Group 3")
        plot_group_prices(price4, "Group 4")

    with col2:
        st.subheader("📉 Graphs of retuns by group")
        plot_group_returns(return_1, "Group 1")
        plot_group_returns(return_2, "Group 2")
        plot_group_returns(return_3, "Group 3")
        plot_group_returns(return_4, "Group 4")


################################################################################################################################################################################
with tab3:
    # Media de los retornos
    return_1_mean = return_1.mean()
    return_2_mean = return_2.mean()
    return_3_mean = return_3.mean()
    return_4_mean = return_4.mean()

    return_1_mean = return_1_mean.to_frame(name = 'Returns')
    return_2_mean = return_2_mean.to_frame(name = 'Returns')
    return_3_mean = return_3_mean.to_frame(name = 'Returns')
    return_4_mean = return_4_mean.to_frame(name = 'Returns')

    # Tratar famafrench
    famafrench['RF'] = pd.to_numeric(famafrench['RF'], errors='coerce')
    famafrench['Mkt-RF'] = pd.to_numeric(famafrench['Mkt-RF'], errors='coerce')# Volver a numero
    famafrench.set_index('Date', inplace=True)

    # Ri - RF
    return1_rf = (return_1_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
    return2_rf = (return_2_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
    return3_rf = (return_3_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
    return4_rf = (return_4_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')


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



    # Simple model non weighted
    # Grupo 1
    y1 = group_1["Re-Rf"] 
    x1 = group_1["Mkt-RF"] 
    x1 = sm.add_constant(x1)
    model1 = sm.OLS(y1, x1).fit()

    # Grupo 2
    y2 = group_2['Re-Rf']
    x2 = group_2['Mkt-RF']
    x2 = sm.add_constant(x2)
    model2 = sm.OLS(y2, x2).fit()

    # Grupo 3
    y3 = group_3['Re-Rf']
    x3 = group_3['Mkt-RF']
    x3 = sm.add_constant(x3)
    model3 = sm.OLS(y3, x3).fit()

    # Group 4
    y4 = group_4['Re-Rf']
    x4 = group_4['Mkt-RF']
    x4 = sm.add_constant(x4)
    model4 = sm.OLS(y4, x4).fit()

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
    # Multifactorial model non weighted
    # Group 1
    y13 = group_1['Re-Rf'] *100
    msci13 = group_1['msci'] *100
    x13 = group_1["Mkt-RF"]  * 100
    x13 = pd.concat([x13, msci13], axis = 1)
    x13 = sm.add_constant(x13)
    model13 = sm.OLS(y13, x13).fit() 

    # Group 2
    y14 = group_2['Re-Rf'] *100
    msci14 = group_2['msci'] *100
    x14 = group_2["Mkt-RF"]  * 100
    x14 = pd.concat([x14, msci14], axis = 1)
    x14 = sm.add_constant(x14)
    model14 = sm.OLS(y14, x14).fit() 

    # Group 3
    y15 = group_3['Re-Rf'] *100
    msci15 = group_3['msci'] *100
    x15 = group_3["Mkt-RF"]  * 100
    x15 = pd.concat([x15, msci15], axis = 1)
    x15 = sm.add_constant(x15)
    model15 = sm.OLS(y15, x15).fit() 

    # Group 4
    y16 = group_4['Re-Rf'] *100
    msci16 = group_4['msci'] *100
    x16 = group_4["Mkt-RF"]  * 100
    x16 = pd.concat([x16, msci16], axis = 1)
    x16 = sm.add_constant(x16)
    model16 = sm.OLS(y16, x16).fit() 

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
    y8 = group_4['Wr-Rf'] *100
    msci4 = group_4['msci'] *100
    x8 = group_4["Mkt-RF"]  * 100
    x8 = pd.concat([x8, msci3], axis = 1)
    x8 = sm.add_constant(x8)
    model8 = sm.OLS(y8, x8).fit() 

    ####################################################################################################

    def extract_model_summary(model):
        """Extrae coeficientes, errores estándar, p-valores, R² y otras métricas."""
        summary_df = pd.DataFrame({
            "Coefficients": model.params,
            "Standard Errors": model.bse,
            "p-values": model.pvalues,
            "Lower range 95%": model.conf_int()[0],
            "Upper range 95%": model.conf_int()[1]
        })
        summary_df.loc["R²", "Coefficients"] = model.rsquared
        summary_df.loc["R² Adjusted", "Coefficients"] = model.rsquared_adj
        summary_df.loc["F-Statistical", "Coefficients"] = model.fvalue
        return summary_df


    st.markdown("""
        <h1 style="text-align: center;">SIMPLE MODEL NON WEIGHTED</h1>
    """, unsafe_allow_html=True) 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write('Simple model non weighted for group 1:' )
        modelo1 = extract_model_summary(model1)
        st.dataframe(modelo1, use_container_width=True)     

    with col2:
        st.write('Simple model non weighted for group 2:' )
        modelo2 = extract_model_summary(model2)
        st.dataframe(modelo2, use_container_width=True)     

    with col3:
        st.write('Simple model non weighted for group 3:' )
        modelo3 = extract_model_summary(model3)
        st.dataframe(modelo3, use_container_width=True)  
        
    with col4:
        st.write('Simple model non weighted for group 4:' )
        modelo4 = extract_model_summary(model4)
        st.dataframe(modelo4, use_container_width=True)        


    st.markdown("""
        <h1 style="text-align: center;">SIMPLE MODEL WEIGHTED</h1>
    """, unsafe_allow_html=True) 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write('Simple model non weighted for group 1:' )
        modelo9 = extract_model_summary(model9)
        st.dataframe(modelo9, use_container_width=True)     

    with col2:
        st.write('Simple model non weighted for group 2:' )
        modelo10 = extract_model_summary(model10)
        st.dataframe(modelo10, use_container_width=True)     

    with col3:
        st.write('Simple model non weighted for group 3:' )
        modelo11 = extract_model_summary(model11)
        st.dataframe(modelo11, use_container_width=True)  
        
    with col4:
        st.write('Simple model non weighted for group 4:' )
        modelo12 = extract_model_summary(model12)
        st.dataframe(modelo12, use_container_width=True)    
        
    st.markdown("""
        <h1 style="text-align: center;">MULTIFACTORIAL MODEL NON WEIGHTED</h1>
    """, unsafe_allow_html=True) 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write('Simple model non weighted for group 1:' )
        modelo13 = extract_model_summary(model13)
        st.dataframe(modelo13, use_container_width=True)     

    with col2:
        st.write('Simple model non weighted for group 2:' )
        modelo14 = extract_model_summary(model14)
        st.dataframe(modelo14, use_container_width=True)     

    with col3:
        st.write('Simple model non weighted for group 3:' )
        modelo15 = extract_model_summary(model15)
        st.dataframe(modelo15, use_container_width=True)  
        
    with col4:
        st.write('Simple model non weighted for group 4:' )
        modelo16 = extract_model_summary(model16)
        st.dataframe(modelo16, use_container_width=True)   
        
        
    st.markdown("""
        <h1 style="text-align: center;">MULTIFACTORIAL MODEL WEIGHTED</h1>
    """, unsafe_allow_html=True) 

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write('Simple model non weighted for group 1:' )
        modelo5 = extract_model_summary(model5)
        st.dataframe(modelo5, use_container_width=True)     

    with col2:
        st.write('Simple model non weighted for group 2:' )
        modelo6 = extract_model_summary(model6)
        st.dataframe(modelo6, use_container_width=True)     

    with col3:
        st.write('Simple model non weighted for group 3:' )
        modelo7 = extract_model_summary(model7)
        st.dataframe(modelo7, use_container_width=True)  
        
    with col4:
        st.write('Simple model non weighted for group 4:' )
        modelo8 = extract_model_summary(model8)
        st.dataframe(modelo8, use_container_width=True)   
        
############################################################################################################################
# Correlaciones

# Importar base
prices1 = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\historical indexes.xlsx', sheet_name=0)
rateex1 = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\historical indexes.xlsx', sheet_name=2)
msci = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\historical indexes.xlsx', sheet_name=1)

# Calcular retornos
prices1.set_index('Country', inplace = True)
rateex1.set_index('Country', inplace = True)

rateusd = prices1 / rateex1
returns1 = rateusd.pct_change(axis = 1)

# Eliminar paises vacios
returns1 = returns1.iloc[:, 1:].dropna(how = 'all')

# Filtrar paises en comun
common_countries3 = returns1.index.intersection(population.index).intersection(lmic.index)
returns1 = returns1.loc[common_countries3]


returns1 = returns1.apply(lambda row: winsorize_series(row), axis=1) 

return_11 = returns1[returns1.index.isin(group1.index)]
return_12 = returns1[returns1.index.isin(group2.index)]
return_13 = returns1[returns1.index.isin(group3.index)]
return_14 = returns1[returns1.index.isin(group4.index)]

return_11_mean = return_11.mean()
return_12_mean = return_12.mean()
return_13_mean = return_13.mean()
return_14_mean = return_14.mean()

return_11_mean = return_11_mean.to_frame(name = 'Returns')
return_12_mean = return_12_mean.to_frame(name = 'Returns')
return_13_mean = return_13_mean.to_frame(name = 'Returns')
return_14_mean = return_14_mean.to_frame(name = 'Returns')


# Correlacion movil
return_11_mean.index = pd.to_datetime(return_11_mean.index)
return_12_mean.index = pd.to_datetime(return_11_mean.index)
return_13_mean.index = pd.to_datetime(return_11_mean.index)
return_14_mean.index = pd.to_datetime(return_11_mean.index)
msci['Date'] = pd.to_datetime(msci['Date'])
msci = msci.set_index('Date')

df_combined = pd.merge(return_11_mean, msci[['MSCI']], left_index=True, right_index=True, how='inner')
df_combined1 = pd.merge(return_12_mean, msci[['MSCI']], left_index=True, right_index=True, how='inner')
df_combined2 = pd.merge(return_13_mean, msci[['MSCI']], left_index=True, right_index=True, how='inner')
df_combined3 = pd.merge(return_14_mean, msci[['MSCI']], left_index=True, right_index=True, how='inner')


rolling_corr = df_combined['Returns'].rolling(window=36).corr(df_combined['MSCI'])
rolling_corr1 = df_combined1['Returns'].rolling(window=36).corr(df_combined['MSCI'])
rolling_corr2 = df_combined2['Returns'].rolling(window=36).corr(df_combined['MSCI'])
rolling_corr3 = df_combined3['Returns'].rolling(window=36).corr(df_combined['MSCI'])

with tab4:
    # Crear un DataFrame combinando todas las series de correlación
    df_corr = pd.DataFrame({
        'Fecha': rolling_corr.index,
        'Group 1': rolling_corr.values,
        'Group 2': rolling_corr1.values,
        'Group 3': rolling_corr2.values,
        'Group 4': rolling_corr3.values
    })

    # Transformar a formato largo (melt)
    df_melted = df_corr.melt(id_vars='Fecha', var_name='Serie', value_name='Correlación')

    # Selección dinámica de series
    series_opciones = df_melted['Serie'].unique().tolist()

    color_map = {
        'Group 1': '#1f77b4',  # azul oscuro
        'Group 2': '#2ca02c',  # azul claro
        'Group 3': '#ffdd57',  # rosado claro
        'Group 4': '#d62728',  # rojo oscuro
}

    # Crear gráfico con Plotly
    fig = px.line(
        df_melted,
        x='Fecha',
        y='Correlación',
        color='Serie',
        color_discrete_map=color_map,
        title='Correlation in a 3-year rolling window',
        labels={'Fecha': 'Date', 'Correlación': 'Correlation'}
    )

    fig.update_layout(height=500, width=1000)

    # Mostrar en Streamlit
    st.plotly_chart(fig)

    
    return_11 = return_11.reset_index()
    return_12 = return_12.reset_index()
    return_13 = return_13.reset_index()
    return_14 = return_14.reset_index()
    
    st.write(return_11['Country'].count())
    st.write(return_12['Country'].count())
    st.write(return_13['Country'].count())
    st.write(return_14['Country'].count())
    
    


# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\capm_dash.py"