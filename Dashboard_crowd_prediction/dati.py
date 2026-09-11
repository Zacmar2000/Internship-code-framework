import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
import os
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error,mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
#------ DATAFRAME PER GESTIONE REDENTORE ALLA GIUDECCA ----


#--------REDENTORE GIUDECCA 2025 ---------
ols = pd.read_csv("./predizioni/redentore_giudecca/ols_predizioni.csv")
ols.drop(columns=["Unnamed: 0"],inplace=True)
ols.rename(columns={"0":"P-OLS"},inplace = True)
ols.head()
sarimax_scalato = pd.read_csv("./predizioni/redentore_giudecca/forecast_mean_scalato.csv")
sarimax_scalato.drop(columns=["Unnamed: 0"],inplace=True)
sarimax_scalato.rename(columns={"predicted_mean":"P-SARIMAX-SCALATO"},inplace = True)
sarimax_scalato.reset_index(drop=True,inplace=True)
#sarimax_scalato.head()
sarimax = pd.read_csv("./predizioni/redentore_giudecca/forecast_mean.csv")
sarimax.drop(columns=["Unnamed: 0"],inplace=True)
sarimax.rename(columns={"predicted_mean":"P-SARIMAX"},inplace = True)

sarimax_log = pd.read_csv("./predizioni/redentore_giudecca/forecast_mean_log.csv")
sarimax_log.drop(columns=["Unnamed: 0"],inplace=True)
sarimax_log.rename(columns={"predicted_mean":"P-SARIMAX-LOG"},inplace = True)

sarimax_box = pd.read_csv("./predizioni/redentore_giudecca/forecast_mean_box.csv")
sarimax_box.drop(columns=["Unnamed: 0"],inplace=True)
sarimax_box.rename(columns={"0":"P-SARIMAX-BOX"},inplace = True)

reali = pd.read_csv("./predizioni/redentore_giudecca/test.csv")
reali.drop(columns=["Unnamed: 0"],inplace=True)
reali.rename(columns={"P":"P-REALI"},inplace = True)
#reali= reali.loc[:,'REALI']*1.2

XGBoost = pd.read_csv("./predizioni/redentore_giudecca/xgboostzaccaria.csv")
XGBoost = XGBoost.loc[(XGBoost['data']<"2025-07-20 09:00") & (XGBoost['data']>="2025-07-19 09:00")].reset_index(drop=True)
#print(len(XGBoost))
XGBoost.drop(columns=["data"],inplace=True)
XGBoost.rename(columns={"P":"P-XGBoost"},inplace = True)

redentore_giudecca_2025 = pd.concat([ols, sarimax_scalato, reali, sarimax, sarimax_log, sarimax_box, XGBoost], axis=1)
redentore_giudecca_2025['Data']= pd.date_range(start ="2025-07-19 07:00", end ="2025-07-20 06:45", freq="15min")#Dalle 7 della mattina del 19 fino le 7 del 20 luglio
redentore_giudecca_2025['TYPE'] = "RedentoreGiudecca" + "2025"


#--------REDENTORE GIUDECCA 2026 ---------

sarimax_2026 = pd.read_csv("./predizioni/redentore_giudecca/sarimax_2026.csv").reset_index(drop=True)
sarimax_2026.drop(columns=["Unnamed: 0"],inplace=True)
sarimax_2026.rename(columns={"predicted_mean":"P-SARIMAX"},inplace = True)
sarimax_2026.reset_index(drop=True,inplace=True)


redentore_giudecca_2026 = pd.concat([sarimax_2026,],axis=1)
redentore_giudecca_2026['Data']= pd.date_range(start = "2026-07-18 07:00",end = "2026-07-19 06:45", freq="15min")#Dalle 7 della mattina del 18 fino le 7 del 19 luglio

redentore_giudecca_2026['TYPE'] = "RedentoreGiudecca"+"2026"




#---------  REDENTORE VENEZIA 2025 ----------
reali = pd.read_csv("./predizioni/redentore_venezia/Redentore_Venezia.csv")
reali = reali.groupby('Timestamp').sum().reset_index()
reali['Timestamp'] = pd.to_datetime(reali['Timestamp'])
reali['Anno'] = reali['Timestamp'].dt.year
reali.rename(columns={"P":"P-REALI","Data":"DataC","Timestamp":"Data","Ni":"ITA-REALI","Ns":"STRA-REALI","Vi":"INTRA-REALI","Vp":"PEND-REALI","Vr":"RES-REALI","Ve":"EXTRA-REALI"},inplace = True)
reali = reali[reali['Anno']==2025].reset_index(drop=True)
reali = reali[['P-REALI','Data']].copy()
reali['Data'] = pd.to_datetime(reali['Data']).dt.tz_localize(None)

XGBoost_redentore_ve = pd.read_csv("./predizioni/redentore_venezia/xgboost_venezia_2025.csv")
#XGBoost_redentore_ve = XGBoost_redentore_ve.loc[(XGBoost_redentore_ve['data']<"2025-07-20 09:00") & (XGBoost_redentore_ve['data']>="2025-07-19 09:00")].reset_index(drop=True)
XGBoost_redentore_ve['data']=pd.to_datetime(XGBoost_redentore_ve['data'])-timedelta(hours=2)
XGBoost_redentore_ve.rename(columns={"P":"P-XGBoost","data":"Data"},inplace = True)


tbats_2025_p_red_ve = pd.read_csv("./predizioni/redentore_venezia/pred_tbats_2025.csv")
tbats_2025_p_red_ve.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_p_red_ve.rename(columns={"0":"P-TBATS"},inplace = True)
tbats_2025_p_red_ve['Data']= pd.date_range(start = "2025-07-19 07:00",end = "2025-07-20 06:45", freq="15min")

df_to_merge = [
    XGBoost_redentore_ve,
    tbats_2025_p_red_ve,
]
df_finale = reali.copy()
for df in df_to_merge:
    df_finale=pd.merge(df_finale,df,on='Data',how='left')
redentore_venezia_2025 = df_finale

redentore_venezia_2025['TYPE'] = "RedentoreVenezia" + "2025"


#------ REDENTORE VENEZIA 2026 --------




#------ CARNEVALE VENEZIA 2025 --------

reali = pd.read_csv("./predizioni/carnevale/df.csv")
reali = reali[reali['Anno']==2025].reset_index(drop=True)
reali.rename(columns={"P":"P-REALI","Data":"DataC","Timestamp":"Data","Ni":"ITA-REALI","Ns":"STRA-REALI","Vi":"INTRA-REALI","Vp":"PEND-REALI","Vr":"RES-REALI","Ve":"EXTRA-REALI"},inplace = True)
reali.drop(columns=["Unnamed: 0",'Gm','Gf','F1','F2','F3','F4','F5','F6',"Anno","Mese","Giorno","Ora","Minuto","DataC","Tempo","Tb","Tc"],inplace=True)
reali['Data']=pd.to_datetime(reali['Data'])

end = reali.iloc[-1*96-1]['Data']
start = reali.iloc[-96*5]['Data']

tbats_2025_p = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_p.csv")
tbats_2025_p.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_p.rename(columns={"0":"P-TBATS"},inplace = True)
tbats_2025_p['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_p_scalato = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_scalato.csv")
tbats_2025_p_scalato.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_p_scalato.rename(columns={"0":"P-TBATS-SCALATO"},inplace = True)
tbats_2025_p_scalato['Data']= pd.date_range(start = start,end = end, freq="15min")


tbats_2025_ita = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_ita.csv")
tbats_2025_ita.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_ita.rename(columns={"0":"ITA-TBATS"},inplace = True)
tbats_2025_ita['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_stra = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_stra.csv")
tbats_2025_stra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_stra.rename(columns={"0":"STRA-TBATS"},inplace = True)
tbats_2025_stra['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_ita_stra = pd.read_csv("./predizioni/carnevale/prediction_tot_2025_ita_stra.csv")
tbats_2025_ita_stra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_ita_stra.rename(columns={"0":"ITA+STRA-TBATS"},inplace = True)
tbats_2025_ita_stra['Data']= pd.date_range(start = start,end = end, freq="15min")


tbats_2025_intra = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_intra.csv")
tbats_2025_intra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_intra.rename(columns={"0":"INTRA-TBATS"},inplace = True)
tbats_2025_intra['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_extra = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_extra.csv")
tbats_2025_extra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_extra.rename(columns={"0":"EXTRA-TBATS"},inplace = True)
tbats_2025_extra['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_pend = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_pend.csv")
tbats_2025_pend.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_pend.rename(columns={"0":"PEND-TBATS"},inplace = True)
tbats_2025_pend['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_res = pd.read_csv("./predizioni/carnevale/pred_tbats_2025_res.csv")
tbats_2025_res.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_res.rename(columns={"0":"RES-TBATS"},inplace = True)
tbats_2025_res['Data']= pd.date_range(start = start,end = end, freq="15min")

tbats_2025_intra_extra = pd.read_csv("./predizioni/carnevale/prediction_tot_2025_intra_extra.csv")
tbats_2025_intra_extra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2025_intra_extra.rename(columns={"0":"INTRA+EXTRA-TBATS"},inplace = True)
tbats_2025_intra_extra['Data']= pd.date_range(start = start,end = end, freq="15min")


prophet = pd.read_csv("./predizioni/redentore_giudecca/prophet.csv")
prophet = pd.DataFrame(prophet['yhat'])
prophet.rename(columns={"yhat":"P-PROPHET"},inplace = True)
prophet['Data']= pd.date_range(start = start,end = end, freq="15min")

XGBoost_carnevale = pd.read_csv("./predizioni/carnevale/final_xgboost_carnevale_2025.csv")
XGBoost_carnevale.drop(columns=["data"],inplace=True)
XGBoost_carnevale.rename(columns={"P":"P-XGBoost"},inplace = True)
XGBoost_carnevale['Data']= reali['Data']
#XGBoost.drop(columns=["data"],inplace=True)

XGBoost_carnevale_scalato = pd.read_csv("./predizioni/carnevale/xgboost_scalato_carnevale2025.csv")
XGBoost_carnevale_scalato.drop(columns=["data"],inplace=True)
XGBoost_carnevale_scalato.rename(columns={"P":"P-XGBoost-SCALATO"},inplace = True)
XGBoost_carnevale_scalato['Data']= reali['Data']

XGBoost_carnevale_ita_stra = pd.read_csv("./predizioni/carnevale/xgboost_italiani_stranieri_carnevale2025.csv")
XGBoost_carnevale_ita_stra.drop(columns=["data"],inplace=True)
XGBoost_carnevale_ita_stra.rename(columns={"P":"ITA+STRA-XGBoost","Ni":"ITA-XGBoost","Ns":"STRA-XGBoost"},inplace = True)
XGBoost_carnevale_ita_stra['Data']= reali['Data']

XGBoost_carnevale_ita_stra_scalato = pd.read_csv("./predizioni/carnevale/xgboost_scalato_italiani_stranieri_carnevale2025.csv")
XGBoost_carnevale_ita_stra_scalato.drop(columns=["data"],inplace=True)
XGBoost_carnevale_ita_stra_scalato.rename(columns={"P":"ITA+STRA-XGBoost-SCALATO","Ni":"ITA-XGBoost-SCALATO","Ns":"STRA-XGBoost-SCALATO"},inplace = True)
XGBoost_carnevale_ita_stra_scalato['Data']= reali['Data']




sarima_2025_carnevale = pd.read_csv("./predizioni/carnevale/pred_sarima_2025.csv")
sarima_2025_carnevale.drop(columns=["Unnamed: 0"],inplace=True)
sarima_2025_carnevale.rename(columns={"predicted_mean":"P-SARIMAX"},inplace = True)
sarima_2025_carnevale['Data']= pd.date_range(start = start,end = end, freq="15min")


sarima_2025_carnevale_giornata = pd.read_csv("./predizioni/carnevale/pred_sarima_2025_giornata.csv")
sarima_2025_carnevale_giornata.drop(columns=["Unnamed: 0"],inplace=True)
sarima_2025_carnevale_giornata.rename(columns={"predicted_mean":"P-SARIMAX-GIORNATA"},inplace = True)
sarima_2025_carnevale_giornata['Data']= pd.date_range(start = start,end = end, freq="15min")

df_to_merge = [
    XGBoost_carnevale_scalato,
    XGBoost_carnevale_ita_stra,
    XGBoost_carnevale_ita_stra_scalato,
    tbats_2025_p_scalato,
    sarima_2025_carnevale,
    sarima_2025_carnevale_giornata,
    XGBoost_carnevale,
    prophet,tbats_2025_p,
    tbats_2025_ita,tbats_2025_stra,
    tbats_2025_ita_stra,tbats_2025_intra,
    tbats_2025_extra,tbats_2025_pend,
    tbats_2025_res,tbats_2025_intra_extra
]
df_finale = reali.copy()
for df in df_to_merge:
    df_finale=pd.merge(df_finale,df,on='Data',how='left')
carnevale_venezia_2025 = df_finale

carnevale_venezia_2025['TYPE'] = "CarnevaleVenezia" + "2025"

#------ CARNEVALE VENEZIA 2026 --------

date= pd.date_range(start = "2026-02-13 23:00",end = "2026-02-17 22:45", freq="15min")
date_tot = pd.date_range(start = "2026-01-29 23:00",end = "2026-02-18 22:45", freq="15min")


reali_2026 = pd.read_csv("./predizioni/carnevale/2026/carnevale_tot_2026.csv")
reali_2026 = reali_2026[reali_2026['Anno']==2026].reset_index(drop=True)
reali_2026.rename(columns={"P":"P-REALI","Data":"DataC","Timestamp":"Data","Ni":"ITA-REALI","Ns":"STRA-REALI","Vi":"INTRA-REALI","Vp":"PEND-REALI","Vr":"RES-REALI","Ve":"EXTRA-REALI"},inplace = True)
reali_2026.drop(columns=['Gm','Gf','F1','F2','F3','F4','F5','F6',"Anno","Mese","Giorno","Ora","Minuto","DataC","Tempo","Tb","Tc"],inplace=True)
reali_2026['Data']= date_tot


sarima_2026_carnevale = pd.read_csv("./predizioni/carnevale/2026/previsione_sarima_2026 (1).csv")
sarima_2026_carnevale.drop(columns=["Unnamed: 0"],inplace=True)
sarima_2026_carnevale.rename(columns={"predicted_mean":"P-SARIMAX"},inplace = True)
sarima_2026_carnevale['Data']= date



tbats_2026_carnevale = pd.read_csv("./predizioni/carnevale/2026/previsione_tbats_2026.csv")
tbats_2026_carnevale.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2026_carnevale.rename(columns={"0":"P-TBATS"},inplace = True)
tbats_2026_carnevale['Data'] = date

tbats_2026_carnevale_ita = pd.read_csv("./predizioni/carnevale/2026/previsione_tbats_2026_ita.csv")
tbats_2026_carnevale_ita.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2026_carnevale_ita.rename(columns={"0":"ITA-TBATS"},inplace = True)
tbats_2026_carnevale_ita['Data'] = date

tbats_2026_carnevale_stra = pd.read_csv("./predizioni/carnevale/2026/previsione_tbats_2026_stra.csv")
tbats_2026_carnevale_stra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2026_carnevale_stra.rename(columns={"0":"STRA-TBATS"},inplace = True)
tbats_2026_carnevale_stra['Data'] = date

tbats_2026_carnevale_ita_stra = pd.read_csv("./predizioni/carnevale/2026/previsione_tbats_2026_ita_stra.csv")
tbats_2026_carnevale_ita_stra.drop(columns=["Unnamed: 0"],inplace=True)
tbats_2026_carnevale_ita_stra.rename(columns={"0":"ITA+STRA-TBATS"},inplace = True)
tbats_2026_carnevale_ita_stra['Data'] = date

XGBoost_carnevale_2026 = pd.read_csv("./predizioni/carnevale/2026/final_xgboost_carnevale_2026.csv")
XGBoost_carnevale_2026.drop(columns=["data"],inplace=True)
XGBoost_carnevale_2026.rename(columns={"P":"P-XGBoost"},inplace = True)
XGBoost_carnevale_2026['Data']= date_tot

standard_XGBoost_carnevale_2026 = pd.read_csv("./predizioni/carnevale/2026/standard_xgboost_carnevale_2026.csv")
standard_XGBoost_carnevale_2026.drop(columns=["data"],inplace=True)
standard_XGBoost_carnevale_2026.rename(columns={"P":"P-XGBoost-SCALATO"},inplace = True)
standard_XGBoost_carnevale_2026['Data']= date_tot

XGBoost_carnevale_ita_stra_2026 = pd.read_csv("./predizioni/carnevale/2026/3_Covid_xgboost_italiani_stranieri_carnevale_2026.csv")
XGBoost_carnevale_ita_stra_2026.drop(columns=["data"],inplace=True)
XGBoost_carnevale_ita_stra_2026.rename(columns={"P":"ITA+STRA-XGBoost","Ni":"ITA-XGBoost","Ns":"STRA-XGBoost"},inplace = True)
XGBoost_carnevale_ita_stra_2026['Data']= date_tot

standard_XGBoost_carnevale_ita_stra_2026 = pd.read_csv("./predizioni/carnevale/2026/standard_3_xgboost_italiani_stranieri_carnevale_2026.csv")
standard_XGBoost_carnevale_ita_stra_2026.drop(columns=["data"],inplace=True)
standard_XGBoost_carnevale_ita_stra_2026.rename(columns={"P":"ITA+STRA-XGBoost-SCALATO","Ni":"ITA-XGBoost-SCALATO","Ns":"STRA-XGBoost-SCALATO"},inplace = True)
standard_XGBoost_carnevale_ita_stra_2026['Data']= date_tot



df_to_merge = [
    tbats_2026_carnevale,
    tbats_2026_carnevale_ita,
    tbats_2026_carnevale_stra,
    tbats_2026_carnevale_ita_stra,
    sarima_2026_carnevale,
    XGBoost_carnevale_2026,
    standard_XGBoost_carnevale_2026,
    XGBoost_carnevale_ita_stra_2026,
    standard_XGBoost_carnevale_ita_stra_2026
]
df_finale = reali_2026.copy()
for df in df_to_merge:
    df_finale=pd.merge(df_finale,df,on='Data',how='left')
carnevale_venezia_2026 = df_finale

carnevale_venezia_2026['TYPE'] = "CarnevaleVenezia"+"2026"



##concateno tutto e lo chiamo -- nomeeventoAnno seguendo questa logica in modo da allegerire anche data visualition

df_plot_global = pd.concat([redentore_giudecca_2025, redentore_giudecca_2026, redentore_venezia_2025, carnevale_venezia_2025, carnevale_venezia_2026])

##Funzione per generazione dei MAPE: 


def metrics_for_event(types='RedentoreGiudecca2025',df = df_plot_global):

     stats = pd.DataFrame()

    #  if '2026' in types:
    #      return stats
     # Mi prendo il dataframe con solo i valori esistenti per ogni casistica
     df_type = df.loc[df['TYPE'] == types, :]
     df_type = df_type.drop(columns = ['Data','TYPE'])
     #df_type.dropna(how = 'all')
     df_type = df_type.dropna(axis=1, how='all')

     col_names = [col.split("-", 1) for col in df_type.columns]

     diz_col = {}
     for chiave, valore in col_names:
         if valore == 'REALI':
             continue
         else:
             diz_col.setdefault(chiave, []).append(valore)

     for i in diz_col.keys():
         if i == 'ITA+STRA':
             cat_name = ('P-REALI')
         elif i == 'INTRA+EXTRA':
             cat_name = ('ITA-REALI')
         else:

             cat_name = (str(i)+'-REALI')

         for col in diz_col[i]:

             if i in ['ITA','STRA']:

                 row_name = 'ITA+STRA-'+str(col)

             elif i in ['INTRA','EXTRA','PEND','RES']:

                 row_name = 'INTRA+EXTRA-'+str(col)

             else:

                 row_name = str(i)+'-'+str(col)

             row_name_vera = str(i)+'-'+str(col)

             df_subset = df_type[[cat_name,row_name_vera]].dropna(how='any')
             stats.loc[row_name, cat_name] = round(mean_absolute_percentage_error(df_subset[row_name_vera],df_subset[cat_name])*100,2)
             if len(df_subset)>0:
                stats.loc[row_name, "Numero Giorni"] = len(df_subset)/96
     if len(stats)>0:
        stats = stats.reset_index()
        stats.rename(columns={'index':'Tipo'}, inplace=True)
        var = stats.pop("Numero Giorni")
        stats['Numero Giorni']=var
     return stats
