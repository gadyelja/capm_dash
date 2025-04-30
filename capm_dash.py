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


tab1, tab2 = st.tabs(["🌎 Map and groups", '📊 Regresions'])

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
lmic1 = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 9)
mark_cap11 = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 7)
population11 = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=8)

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
# Upper medium income level
um_mc1 = um_mc[population11['Population'] <=group1_max] 
um_r1 = um_r[population11['Population'] <= group1_max]
um_mc2 = um_mc[(population11['Population'] > group1_max) & (population11['Population'] <=group2_max)] 
um_r2 = um_r[(population11['Population'] > group1_max) & (population['Population'] <= group2_max)]
um_mc3 = um_mc[(population11['Population'] > group2_max) & (population11['Population'] <=group3_max)] 
um_r3 = um_r[(population11['Population'] > group2_max) & (population['Population'] <= group3_max)]
um_mc4 = um_mc[population11['Population'] > group3_max] 
um_r4 = um_r[population11['Population'] > group3_max]

# Lower medium income level
lm_mc1 = lm_mc[population11['Population'] <=group1_max] 
lm_r1 = lm_r[population11['Population'] <= group1_max]
lm_mc2 = lm_mc[(population11['Population'] > group1_max) & (population11['Population'] <=group2_max)] 
lm_r2 = lm_r[(population11['Population'] > group1_max) & (population['Population'] <= group2_max)]
lm_mc3 = lm_mc[(population11['Population'] > group2_max) & (population11['Population'] <=group3_max)] 
lm_r3 = lm_r[(population11['Population'] > group2_max) & (population['Population'] <= group3_max)]
lm_mc4 = lm_mc[population11['Population'] > group3_max] 
lm_r4 = lm_r[population11['Population'] > group3_max]

# Low income level
l_mc1 = l_mc[population11['Population'] <=group1_max] 
l_r1 = l_r[population11['Population'] <= group1_max]
l_mc2 = l_mc[(population11['Population'] > group1_max) & (population11['Population'] <=group2_max)] 
l_r2 = l_r[(population11['Population'] > group1_max) & (population['Population'] <= group2_max)]
l_mc3 = l_mc[(population11['Population'] > group2_max) & (population11['Population'] <=group3_max)] 
l_r3 = l_r[(population11['Population'] > group2_max) & (population['Population'] <= group3_max)]
l_mc4 = l_mc[population11['Population'] > group3_max] 
l_r4 = l_r[population11['Population'] > group3_max]
    

# Ponderar retornos
hwr = h_r.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hwr['Date'] = pd.to_datetime(hwr['Date'])
hwr = hwr.merge(h_mc, on = 'Country', how = 'left')
hwr = hwr[hwr['Returns'].notna()].copy()
hsmc = hwr.groupby('Date')['mark_cap'].sum().rename('hsmc')
hwr = hwr.merge(hsmc, on = 'Date', how = 'left')
hwr['wr'] = hwr['Returns'] * (hwr['mark_cap'])
hswr = hwr.groupby('Date')['wr'].sum().reset_index()
hwr = hwr.merge(hswr, on = 'Date', how = 'left')
hwr = hwr.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hwr['hwr'] = hwr['wr_y'] / hwr['hsmc']

# Upper medium income countries
hmwr = um_r.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hmwr['Date'] = pd.to_datetime(hmwr['Date'])
hmwr = hmwr.merge(um_mc, on = 'Country', how = 'left')
hmwr = hmwr[hmwr['Returns'].notna()].copy()
hmsmc = hmwr.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
hmwr = hmwr.merge(hmsmc, on = 'Date', how = 'left')
hmwr['wr'] = hmwr['Returns'] * (hmwr['mark_cap'])
hmswr = hmwr.groupby('Date')['wr'].sum().reset_index()
hmwr = hmwr.merge(hmswr, on = 'Date', how = 'left')
hmwr = hmwr.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hmwr['umwr'] = hmwr['wr_y'] / hmwr['sum_mark_cap']

hmwr1 = um_r1.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hmwr1['Date'] = pd.to_datetime(hmwr1['Date'])
hmwr1 = hmwr1.merge(um_mc1, on = 'Country', how = 'left')
hmwr1 = hmwr1[hmwr1['Returns'].notna()].copy()
hmsmc1 = hmwr1.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
hmwr1 = hmwr1.merge(hmsmc1, on = 'Date', how = 'left')
hmwr1['wr'] = hmwr1['Returns'] * (hmwr1['mark_cap'])
hmswr1 = hmwr1.groupby('Date')['wr'].sum().reset_index()
hmwr1 = hmwr1.merge(hmswr1, on = 'Date', how = 'left')
hmwr1 = hmwr1.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hmwr1['umwr1'] = hmwr1['wr_y'] / hmwr1['sum_mark_cap']

hmwr2 = um_r2.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hmwr2['Date'] = pd.to_datetime(hmwr2['Date'])
hmwr2 = hmwr2.merge(um_mc2, on = 'Country', how = 'left')
hmwr2 = hmwr2[hmwr2['Returns'].notna()].copy()
hmsmc2 = hmwr2.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
hmwr2 = hmwr2.merge(hmsmc2, on = 'Date', how = 'left')
hmwr2['wr'] = hmwr2['Returns'] * (hmwr2['mark_cap'])
hmswr2 = hmwr2.groupby('Date')['wr'].sum().reset_index()
hmwr2 = hmwr2.merge(hmswr2, on = 'Date', how = 'left')
hmwr2 = hmwr2.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hmwr2['umwr2'] = hmwr2['wr_y'] / hmwr2['sum_mark_cap']

hmwr3 = um_r3.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hmwr3['Date'] = pd.to_datetime(hmwr3['Date'])
hmwr3 = hmwr3.merge(um_mc3, on = 'Country', how = 'left')
hmwr3 = hmwr3[hmwr3['Returns'].notna()].copy()
hmsmc3 = hmwr3.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
hmwr3 = hmwr3.merge(hmsmc3, on = 'Date', how = 'left')
hmwr3['wr'] = hmwr3['Returns'] * (hmwr3['mark_cap'])
hmswr3 = hmwr3.groupby('Date')['wr'].sum().reset_index()
hmwr3 = hmwr3.merge(hmswr3, on = 'Date', how = 'left')
hmwr3 = hmwr3.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hmwr3['umwr3'] = hmwr3['wr_y'] / hmwr3['sum_mark_cap']

hmwr4 = um_r4.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
hmwr4['Date'] = pd.to_datetime(hmwr4['Date'])
hmwr4 = hmwr4.merge(um_mc4, on = 'Country', how = 'left')
hmwr4 = hmwr4[hmwr4['Returns'].notna()].copy()
hmsmc4 = hmwr4.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
hmwr4 = hmwr4.merge(hmsmc4, on = 'Date', how = 'left')
hmwr4['wr'] = hmwr4['Returns'] * (hmwr4['mark_cap'])
hmswr4 = hmwr4.groupby('Date')['wr'].sum().reset_index()
hmwr4 = hmwr4.merge(hmswr4, on = 'Date', how = 'left')
hmwr4 = hmwr4.drop_duplicates(subset = 'Date'). reset_index(drop = True)
hmwr4['umwr4'] = hmwr4['wr_y'] / hmwr4['sum_mark_cap']

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

lmwr1 = lm_r1.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lmwr1['Date'] = pd.to_datetime(lmwr1['Date'])
lmwr1 = lmwr1.merge(lm_mc1, on = 'Country', how = 'left')
lmwr1 = lmwr1[lmwr1['Returns'].notna()].copy()
lmsmc1 = lmwr1.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lmwr1 = lmwr1.merge(lmsmc1, on = 'Date', how = 'left')
lmwr1['wr'] = lmwr1['Returns'] * (lmwr1['mark_cap'])
lmswr1 = lmwr1.groupby('Date')['wr'].sum().reset_index()
lmwr1 = lmwr1.merge(lmswr1, on = 'Date', how = 'left')
lmwr1 = lmwr1.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lmwr1['lmwr1'] = lmwr1['wr_y'] / lmwr1['sum_mark_cap']
    
lmwr2 = lm_r2.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lmwr2['Date'] = pd.to_datetime(lmwr2['Date'])
lmwr2 = lmwr2.merge(lm_mc2, on = 'Country', how = 'left')
lmwr2 = lmwr2[lmwr2['Returns'].notna()].copy()
lmsmc2 = lmwr2.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lmwr2 = lmwr2.merge(lmsmc2, on = 'Date', how = 'left')
lmwr2['wr'] = lmwr2['Returns'] * (lmwr2['mark_cap'])
lmswr2 = lmwr2.groupby('Date')['wr'].sum().reset_index()
lmwr2 = lmwr2.merge(lmswr2, on = 'Date', how = 'left')
lmwr2 = lmwr2.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lmwr2['lmwr2'] = lmwr2['wr_y'] / lmwr2['sum_mark_cap']
    
lmwr3 = lm_r3.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lmwr3['Date'] = pd.to_datetime(lmwr3['Date'])
lmwr3 = lmwr3.merge(lm_mc3, on = 'Country', how = 'left')
lmwr3 = lmwr3[lmwr3['Returns'].notna()].copy()
lmsmc3 = lmwr3.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lmwr3 = lmwr3.merge(lmsmc3, on = 'Date', how = 'left')
lmwr3['wr'] = lmwr3['Returns'] * (lmwr3['mark_cap'])
lmswr3 = lmwr3.groupby('Date')['wr'].sum().reset_index()
lmwr3 = lmwr3.merge(lmswr3, on = 'Date', how = 'left')
lmwr3 = lmwr3.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lmwr3['lmwr3'] = lmwr3['wr_y'] / lmwr3['sum_mark_cap']

lmwr4 = lm_r4.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lmwr4['Date'] = pd.to_datetime(lmwr4['Date'])
lmwr4 = lmwr4.merge(lm_mc4, on = 'Country', how = 'left')
lmwr4 = lmwr4[lmwr4['Returns'].notna()].copy()
lmsmc4 = lmwr4.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lmwr4 = lmwr4.merge(lmsmc4, on = 'Date', how = 'left')
lmwr4['wr'] = lmwr4['Returns'] * (lmwr4['mark_cap'])
lmswr4 = lmwr4.groupby('Date')['wr'].sum().reset_index()
lmwr4 = lmwr4.merge(lmswr4, on = 'Date', how = 'left')
lmwr4 = lmwr4.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lmwr4['lmwr4'] = lmwr4['wr_y'] / lmwr4['sum_mark_cap']

# Lower income countries
lwr = l_r.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lwr['Date'] = pd.to_datetime(lwr['Date'])
lwr = lwr.merge(l_mc, on = 'Country', how = 'left')
lwr = lwr[lwr['Returns'].notna()].copy()
lsmc = lwr.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lwr = lwr.merge(lsmc, on = 'Date', how = 'left')
lwr['wr'] = lwr['Returns'] * (lwr['mark_cap'])
lswr = lwr.groupby('Date')['wr'].sum().reset_index()
lwr = lwr.merge(lswr, on = 'Date', how = 'left')
lwr = lwr.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lwr['lwr'] = lwr['wr_y'] / lwr['sum_mark_cap']

lwr1 = l_r1.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lwr1['Date'] = pd.to_datetime(lwr1['Date'])
lwr1 = lwr1.merge(l_mc1, on = 'Country', how = 'left')
lwr1 = lwr1[lwr1['Returns'].notna()].copy()
lsmc1 = lwr1.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lwr1 = lwr1.merge(lsmc1, on = 'Date', how = 'left')
lwr1['wr'] = lwr1['Returns'] * (lwr1['mark_cap'])
lswr1 = lwr1.groupby('Date')['wr'].sum().reset_index()
lwr1 = lwr1.merge(lswr1, on = 'Date', how = 'left')
lwr1 = lwr1.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lwr1['lwr1'] = lwr1['wr_y'] / lwr1['sum_mark_cap']

lwr2 = l_r2.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lwr2['Date'] = pd.to_datetime(lwr2['Date'])
lwr2 = lwr2.merge(l_mc2, on = 'Country', how = 'left')
lwr2 = lwr2[lwr2['Returns'].notna()].copy()
lsmc2 = lwr2.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lwr2 = lwr2.merge(lsmc2, on = 'Date', how = 'left')
lwr2['wr'] = lwr2['Returns'] * (lwr2['mark_cap'])
lswr2 = lwr2.groupby('Date')['wr'].sum().reset_index()
lwr2 = lwr2.merge(lswr2, on = 'Date', how = 'left')
lwr2 = lwr2.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lwr2['lwr2'] = lwr2['wr_y'] / lwr2['sum_mark_cap']

lwr3 = l_r3.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lwr3['Date'] = pd.to_datetime(lwr3['Date'])
lwr3 = lwr3.merge(l_mc3, on = 'Country', how = 'left')
lwr3 = lwr3[lwr3['Returns'].notna()].copy()
lsmc3 = lwr3.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lwr3 = lwr3.merge(lsmc3, on = 'Date', how = 'left')
lwr3['wr'] = lwr3['Returns'] * (lwr3['mark_cap'])
lswr3 = lwr3.groupby('Date')['wr'].sum().reset_index()
lwr3 = lwr3.merge(lswr3, on = 'Date', how = 'left')
lwr3 = lwr3.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lwr3['lwr3'] = lwr3['wr_y'] / lwr3['sum_mark_cap']

lwr4 = l_r4.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
lwr4['Date'] = pd.to_datetime(lwr4['Date'])
lwr4 = lwr4.merge(l_mc4, on = 'Country', how = 'left')
lwr4 = lwr4[lwr4['Returns'].notna()].copy()
lsmc4 = lwr4.groupby('Date')['mark_cap'].sum().rename('sum_mark_cap')
lwr4 = lwr4.merge(lsmc4, on = 'Date', how = 'left')
lwr4['wr'] = lwr4['Returns'] * (lwr4['mark_cap'])
lswr4 = lwr4.groupby('Date')['wr'].sum().reset_index()
lwr4 = lwr4.merge(lswr4, on = 'Date', how = 'left')
lwr4 = lwr4.drop_duplicates(subset = 'Date'). reset_index(drop = True)
lwr4['lwr4'] = lwr4['wr_y'] / lwr4['sum_mark_cap']

# Correlacion movil

# High - Upper middle income countries
corr_df1 = pd.merge(hwr[['Date', 'hwr']], hmwr[['Date', 'umwr']], on = 'Date', how = 'left')
corr_df11 = pd.merge(hwr[['Date', 'hwr']], hmwr1[['Date', 'umwr1']], on = 'Date', how = 'left')
corr_df12 = pd.merge(hwr[['Date', 'hwr']], hmwr2[['Date', 'umwr2']], on = 'Date', how = 'left')
corr_df13 = pd.merge(hwr[['Date', 'hwr']], hmwr3[['Date', 'umwr3']], on = 'Date', how = 'left')
corr_df14 = pd.merge(hwr[['Date', 'hwr']], hmwr4[['Date', 'umwr4']], on = 'Date', how = 'left')
corr_df1 = corr_df1.dropna()
corr_df11 = corr_df11.dropna()
corr_df12 = corr_df12.dropna()
corr_df13 = corr_df13.dropna()
corr_df14 = corr_df14.dropna()

# High - low middle income countries
corr_df2 = pd.merge(hwr[['Date', 'hwr']], lmwr[['Date', 'lmwr']], on = 'Date', how = 'left')
corr_df21 = pd.merge(hwr[['Date', 'hwr']], lmwr1[['Date', 'lmwr1']], on = 'Date', how = 'left')
corr_df22 = pd.merge(hwr[['Date', 'hwr']], lmwr2[['Date', 'lmwr2']], on = 'Date', how = 'left')
corr_df23 = pd.merge(hwr[['Date', 'hwr']], lmwr3[['Date', 'lmwr3']], on = 'Date', how = 'left')
corr_df24 = pd.merge(hwr[['Date', 'hwr']], lmwr4[['Date', 'lmwr4']], on = 'Date', how = 'left')
corr_df2= corr_df2.dropna()
corr_df21= corr_df21.dropna()
corr_df22= corr_df22.dropna()
corr_df23= corr_df23.dropna()
corr_df24= corr_df24.dropna()

# High - low income countries
corr_df3 = pd.merge(hwr[['Date', 'hwr']], lwr[['Date', 'lwr']], on = 'Date', how = 'left')
corr_df31 = pd.merge(hwr[['Date', 'hwr']], lwr1[['Date', 'lwr1']], on = 'Date', how = 'left')
corr_df32 = pd.merge(hwr[['Date', 'hwr']], lwr2[['Date', 'lwr2']], on = 'Date', how = 'left')
corr_df33 = pd.merge(hwr[['Date', 'hwr']], lwr3[['Date', 'lwr3']], on = 'Date', how = 'left')
corr_df34 = pd.merge(hwr[['Date', 'hwr']], lwr4[['Date', 'lwr4']], on = 'Date', how = 'left')
corr_df3 = corr_df3.dropna()
corr_df31 = corr_df31.dropna()
corr_df32 = corr_df32.dropna()
corr_df33 = corr_df33.dropna()
corr_df34 = corr_df34.dropna()

# Graficos de correlacion movil
um_corr = corr_df1['umwr'].rolling(window = 36).corr(corr_df1['hwr'])
um_corr1 = corr_df11['umwr1'].rolling(window = 36).corr(corr_df11['hwr'])
um_corr2 = corr_df12['umwr2'].rolling(window = 36).corr(corr_df12['hwr'])
um_corr3 = corr_df13['umwr3'].rolling(window = 36).corr(corr_df13['hwr'])
um_corr4 = corr_df14['umwr4'].rolling(window = 36).corr(corr_df14['hwr'])

um_corr5 = corr_df2['lmwr'].rolling(window = 36).corr(corr_df2['hwr'])
um_corr6 = corr_df21['lmwr1'].rolling(window = 36).corr(corr_df21['hwr'])
um_corr7 = corr_df22['lmwr2'].rolling(window = 36).corr(corr_df22['hwr'])
um_corr8 = corr_df23['lmwr3'].rolling(window = 36).corr(corr_df23['hwr'])
um_corr9 = corr_df24['lmwr4'].rolling(window = 36).corr(corr_df24['hwr'])

um_corr10 = corr_df3['lwr'].rolling(window = 36).corr(corr_df3['hwr'])
um_corr11 = corr_df31['lwr1'].rolling(window = 36).corr(corr_df31['hwr'])
um_corr12 = corr_df32['lwr2'].rolling(window = 36).corr(corr_df32['hwr'])
um_corr13 = corr_df33['lwr3'].rolling(window = 36).corr(corr_df33['hwr'])
um_corr14 = corr_df34['lwr4'].rolling(window = 36).corr(corr_df34['hwr'])


with tab1:
    
    st.markdown("""
        <h1 style="text-align: center;">CORRELATION IN A 3 YEAR ROLLING WINDOW</h1>
    """, unsafe_allow_html=True) 
    
    # Crear un DataFrame combinando todas las series de correlación
    df_corr = pd.concat([
        um_corr.rename('High - Upper Middle Income Countries'),
        um_corr1.rename('Lowest population'),
        um_corr2.rename('Lower middle population'),
        um_corr3.rename('Upper middle population'),
        um_corr4.rename('Highest population')
    ], axis=1)
    df_corr = df_corr.reset_index().rename(columns={'index': 'Date'})
    
    # Transformar a formato largo (melt)
    df_um_corr = df_corr.melt(id_vars= 'Date', var_name='Serie', value_name='Correlación')

    # Selección dinámica de series
    series_opciones = df_um_corr['Serie'].unique().tolist()

    color_map = {
        'High - Upper Middle Income Countries': '#1f77b4',  # azul oscuro
        'Lowest population': '#2ca02c',  # azul claro
        'Lower middle population': '#ffdd57',  # rosado claro
        'Upper middle population': '#0000FF', 
        'Highest population': '#d62728',  # rojo oscuro
}
    # Repite las fechas hasta alcanzar el tamaño de df_um_corr
    fechas_repetidas = np.tile(hwr['Date'].values, int(np.ceil(len(df_um_corr) / len(hwr)))).tolist()
    # Corta exactamente a 2105 valores
    fechas_repetidas = fechas_repetidas[:len(df_um_corr)]
    # Asigna al DataFrame
    df_um_corr['Date'] = fechas_repetidas
    df_um_corr['Date'] = pd.to_datetime(df_um_corr['Date'])
    # df['Date'] = pd.to_datetime(df['Date'])
    fig = px.line(
        df_um_corr,
        x='Date',
        y='Correlación',
        color='Serie',
        color_discrete_map=color_map,
        title='High - Upper middle income countries',
        labels={'Fecha': 'Date', 'Correlación': 'Correlation'}
    )

    fig.update_layout(height=500, width=1000)

    # Mostrar en Streamlit
    st.plotly_chart(fig)

##############################################################################################################
    # Crear un DataFrame combinando todas las series de correlación
    df_corr1 = pd.concat([
        um_corr5.rename('High - Upper Middle Income Countries'),
        um_corr6.rename('Lowest population'),
        um_corr7.rename('Lower middle population'),
        um_corr8.rename('Upper middle population'),
        um_corr9.rename('Highest population')    ], axis=1)
    df_corr1 = df_corr1.reset_index().rename(columns={'index': 'Fecha'})

    # Transformar a formato largo (melt)
    df_lm_corr = df_corr1.melt(id_vars='Fecha', var_name='Serie', value_name='Correlación')

    # Selección dinámica de series
    series_opciones1 = df_lm_corr['Serie'].unique().tolist()

    color_map = {
        'High - Upper Middle Income Countries': '#1f77b4',  # azul oscuro
        'Lowest population': '#2ca02c',  # azul claro
        'Lower middle population': '#ffdd57',  # rosado claro
        'Upper middle population': '#0000FF', 
        'Highest population': '#d62728',  # rojo oscuro
}
    # Repite las fechas hasta alcanzar el tamaño de df_um_corr
    fechas_repetidas1 = np.tile(lmwr['Date'].values, int(np.ceil(len(df_lm_corr) / len(lmwr)))).tolist()
    # Corta exactamente a 2105 valores
    fechas_repetidas1 = fechas_repetidas1[:len(df_lm_corr)]
    # Asigna al DataFrame
    df_lm_corr['Date'] = fechas_repetidas1
    df_lm_corr['Date'] = pd.to_datetime(df_lm_corr['Date'])
    
    fig1 = px.line(
        df_lm_corr,
        x='Date',
        y='Correlación',
        color='Serie',
        color_discrete_map=color_map,
        title='High - Lower middle income countries',
        labels={'Fecha': 'Date', 'Correlación': 'Correlation'}
    )

    fig1.update_layout(height=500, width=1000)

    # Mostrar en Streamlit
    st.plotly_chart(fig1)


##############################################################################################################
    # Crear un DataFrame combinando todas las series de correlación
    df_corr2 = pd.concat([
        um_corr10.rename('High - Lower Middle Inconme Countries'),
        um_corr11.rename('Lowest population'),
        um_corr12.rename('Lower middle population'),
        um_corr13.rename('Upper middle population'),
        um_corr14.rename('Highest population')
    ], axis=1)
    df_corr2 = df_corr2.reset_index().rename(columns={'index': 'Fecha'})

    # Transformar a formato largo (melt)
    df_l_corr = df_corr2.melt(id_vars='Fecha', var_name='Serie', value_name='Correlación')

    # Selección dinámica de series
    series_opciones2 = df_l_corr['Serie'].unique().tolist()

    color_map = {
        'High - Lower Middle Inconme Countries': '#1f77b4',  # azul oscuro
        'Lowest population': '#2ca02c',  # azul claro
        'Lower middle population': '#ffdd57',  # rosado claro
        'Upper middle population': '#0000FF',
        'Highest': '#d62728',  # rojo oscuro
}
    # Repite las fechas hasta alcanzar el tamaño de df_um_corr
    fechas_repetidas2 = np.tile(lwr['Date'].values, int(np.ceil(len(df_l_corr) / len(lwr)))).tolist()
    # Corta exactamente a 2105 valores
    fechas_repetidas2 = fechas_repetidas2[:len(df_l_corr)]
    # Asigna al DataFrame
    df_l_corr['Date'] = fechas_repetidas2
    df_l_corr['Date'] = pd.to_datetime(df_l_corr['Date'])
    
    fig2 = px.line(
        df_l_corr,
        x='Fecha',
        y='Correlación',
        color='Serie',
        color_discrete_map=color_map,
        title='High - Low income countries',
        labels={'Fecha': 'Date', 'Correlación': 'Correlation'}
    )

    fig2.update_layout(height=500, width=1000)

    # Mostrar en Streamlit
    st.plotly_chart(fig2)

    



# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\capm_dash.py"