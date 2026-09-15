from datetime import timedelta
from operator import index

from click import style
#Userò shiny core -- Più estensibile generico e se il progetto si ingrandisce è piu diviso
from shiny import App, reactive, ui,render
from shinywidgets import output_widget, render_widget
import plotly.io as pio
import plotly.express as px
from sklearn.metrics import mean_absolute_percentage_error

from dati import df_plot_global,metrics_for_event
import pandas as pd


def get_columns(df):
    nomi_colonne = list(df.select_dtypes(include='number').dropna(axis=1,how="all"))#Colonnne che non hanno ne NAN e sono di tipo numerico
    numeri_colonne = range(1,len(nomi_colonne)+1)
    return list(zip(nomi_colonne,numeri_colonne))


evento = {
    0: "None",
    1: "Redentore Giudecca",
    2: "Redentore Venezia",
    3: "Carnevale Venezia",
}

type_evento = {
    1: "RedentoreGiudecca2025",
    2: "RedentoreVenezia2025",
    3: "CarnevaleVenezia2025",
}

options_evento = {
    1: {
        2025: get_columns(df_plot_global.loc[df_plot_global['TYPE']=='RedentoreGiudecca2025']),
        2026: get_columns(df_plot_global.loc[df_plot_global['TYPE']=='RedentoreGiudecca2026']),
       },             # Redentore Giudecca
    2: {
        2025 : get_columns(df_plot_global.loc[df_plot_global['TYPE']=='RedentoreVenezia2025']),
        2026: get_columns(df_plot_global.loc[df_plot_global['TYPE']=='RedentoreVenezia2026']),
       },      # Redentore Venezia
    3: {
        2025: get_columns(df_plot_global.loc[(df_plot_global['TYPE']=='CarnevaleVenezia2025')]),
        2026: get_columns(df_plot_global.loc[(df_plot_global['TYPE']=='CarnevaleVenezia2026')]),
       },        # Carnevale Venezia
}

page1 = ui.page_fluid(
    ui.card(
    ui.layout_sidebar(
    ui.sidebar(
        ui.card_header("Impostazioni grafico"),
            ui.input_select(
            id="evento",
            label="Evento da scegliere",
            choices=evento,
            selected="0",

            ),
    ui.output_ui("widget_scelti"),
    ),
ui.div(
            output_widget("mostra_plot"),
            style="min-width: 1500px; min-height: 900px;",
        ),
    ),
        style= "background-color: white;"
    ),
    ui.card(
        ui.h2("Statistiche per evento"),
        ui.panel_conditional("input.evento != 0", ui.input_switch("update_panel", "Solo i modelli selezionati:")),
    ui.accordion(
        ui.accordion_panel(
                "Mape Redentore Giudecca 2025",
                ui.output_data_frame("metrics_redentore_giudecca"),
            ),
        ui.accordion_panel(
                "Mape Redentore Venezia 2025",
                ui.output_data_frame("metrics_redentore_venezia"),
            ),
        ui.accordion_panel(
                "Mape Carnevale Venezia 2025",
                ui.output_data_frame("metrics_carnevale_venezia"),
            ),
        id="accordion_stats",
        open = False,
        ),
    ),
ui.head_content(ui.tags.style(
    """
    body{
    background-color: #3A6963;
    }
    """

))

    )


app_ui = ui.page_navbar(
    ui.nav_panel("Home",page1),
    title = "Analisi Predittiva",

)


def server(input):
    @reactive.Calc
    def scelta_evento(): # ritorna None se l'input è none altrimenti int(evento)
        #print("input.evento ",input.evento())
        evento = input.evento()
        if evento is None:
            return None
        else:
            return int(evento)
    #RITORNA IL NUMERO DEL EVENTO SCELTO ES 1 --- REDENTORE GIUDECCA
    @reactive.Calc
    def visualizza_anni():
        evento = scelta_evento()
        if evento is None:
            return []
        return [str(a) for a in options_evento.get(evento, {}).keys()]
    #DEVE RITORNARE TUTTA LA LISTA DEI POSSIBILI ANNI PER QUEL EVENTO

    @reactive.Calc
    def anno_scelto():
        if "anno" not in input:
            return None

        anno = input.anno()
        if not anno:
            return None
        #print(anno)
        return int(anno)

    def get_modelli(evento: int, anno: int):
        if evento is None or anno is None:
            return []
        if evento not in options_evento:
            return []
        if anno not in options_evento[evento]:
            return []
        return [nome for nome, _ in options_evento[evento][anno]]

    @render.ui
    def widget_scelti():
        if scelta_evento()!=0:
            return ui.div(

                    ui.input_select(
                id="anno",
                label="Evento da scegliere",
                choices=visualizza_anni(),
                selected=(
                    input.anno()
                    if ("anno" in input and input.anno() in visualizza_anni())
                    else visualizza_anni()[0]
                ),
                ),
                ui.input_selectize(
                "modelli_scelti",
                "Seleziona i modelli da aggiungere nel confronto:",
                choices= get_modelli(scelta_evento(), anno_scelto()), #ritorna un dictionary
                multiple=True,
                ),
                ui.input_checkbox("oralocale", "Ora Locale", False),
                ui.input_action_button("mostra", "Mostra"),
            style = "background-color: white;", #Imposta il colore sfondo altrimenti si sovrappone con
            )
        else:
            return None

    @render_widget
    #@reactive.effect #Tutti eventi che hanno effetti colaterali devono avere questo
    @reactive.event(input.mostra) #Verra eseguita solo in seguito ad un cambiamento dello stato di mostra
    def mostra_plot():
        modelli = input.modelli_scelti()
        evento_num = scelta_evento()
        if evento_num is None:
            return None
        anno = anno_scelto() #Non so se sia giusto dopo metti apposto
        nome_evento = evento[evento_num]
        mask = (str(nome_evento)+str(anno)).replace(" ","")
        df_plot= df_plot_global.loc[df_plot_global['TYPE']==mask].copy()
        #print(df_plot[['Data','P-SARIMAX']])
        df_plot['Data'] = pd.to_datetime(df_plot['Data'])
        #print(df_plot)
        df_plot_val = df_plot.melt(
            id_vars=['Data'],
            value_vars=modelli,  # Seleziona le colonne da tracciare
            var_name='Tipo',
            value_name='Valore'
        )
        df_plot_val['Data'] = pd.to_datetime(df_plot_val['Data'],utc=True)
        if input.oralocale():
            df_plot_val['Data'] = df_plot_val['Data'].dt.tz_convert('Europe/Rome')

        df_plot_val['Data']=df_plot_val['Data'].dt.strftime('%Y-%m-%d %H:%M:%S')
        fig = px.line(
            df_plot_val,
            x='Data',
            y='Valore',
            color='Tipo',  # Questo crea linee separate e una legenda
            title='Modelli',

        )

        fig.update_yaxes(
            range=[0, df_plot.max(numeric_only=True).max()],  # [Valore minimo, Valore massimo] desiderati
            title='Presenze',  # Puoi anche cambiare l'etichetta
        )
        fig.update_layout(
            width=1500,  # Imposta la larghezza in pixel
            height=900,  # Sovrascrive/conferma l'altezza
            hovermode='x unified',
        )
        if mask == "CarnevaleVenezia2025":
            fig.update_layout(
                updatemenus=[dict(type="buttons", direction="left", buttons=[dict(label="4g", method="relayout", args=[
                    {"xaxis.range": ["2025-02-28 23:00:00Z", "2025-03-04 23:00:00Z"]}]),
                                                                             dict(label="All", method="relayout",
                                                                                  args=[{"xaxis.autorange": True}])],
                                  x=0, y=1.05, xanchor="left")]

            )
        elif mask== "CarnevaleVenezia2026":

            fig.update_layout(
                updatemenus=[dict(type="buttons", direction="left", buttons=[dict(label="4g", method="relayout", args=[
                    {"xaxis.range": ["2026-02-13 23:00:00", "2026-02-17 23:00:00"]}]),
                                                                             dict(label="All", method="relayout",
                                                                                  args=[{"xaxis.autorange": True}])],
                                  x=0, y=1.05, xanchor="left")]

            )
        return fig


    

    @reactive.Calc
    def stats_evento_selezionato():
        if not input.update_panel():
            return None

        modelli = list(input.modelli_scelti())

        # --- LOGICA MODELLI (la tua, identica) ---
        cat_modelli = list(set([col.split("-", 1)[0] for col in modelli]))

        if 'ITA+STRA' in cat_modelli:
            if 'P' not in cat_modelli:
                cat_modelli.append('P')
            cat_modelli.remove('ITA+STRA')

        if 'INTRA+EXTRA' in cat_modelli:
            if 'ITA' not in cat_modelli:
                cat_modelli.append('ITA')
            cat_modelli.remove('INTRA+EXTRA')

        for cat in cat_modelli:
            real = f"{cat}-REALI"
            if real not in modelli:
                modelli = [real] + modelli

        evento_num = scelta_evento()
        if evento_num is None:
            return None

        df = df_plot_global.copy()
        df = df[list(modelli) + ["Data", "TYPE"]]

        return metrics_for_event(type_evento[evento_num], df)
    
    @render.data_frame
    def metrics_redentore_giudecca():
        if input.update_panel() and scelta_evento() == 1:
            return render.DataGrid(stats_evento_selezionato())
        return render.DataGrid(metrics_for_event("RedentoreGiudecca2025", df_plot_global))


    @render.data_frame
    def metrics_redentore_venezia():
        if input.update_panel() and scelta_evento() == 2:
            return render.DataGrid(stats_evento_selezionato())
        return render.DataGrid(metrics_for_event("RedentoreVenezia2025", df_plot_global))


    @render.data_frame
    def metrics_carnevale_venezia():
        if input.update_panel() and scelta_evento() == 3:
            return render.DataGrid(stats_evento_selezionato())
        return render.DataGrid(metrics_for_event("CarnevaleVenezia2025", df_plot_global))
    

    @reactive.effect
    def aggiorna_accordion():
        if not input.update_panel():
            ui.update_accordion("accordion_stats", show=False)
            return  # switch disattivo → nulla cambia

        evento = scelta_evento()
        if evento is None:
            return

        pannello_da_aprire = {
            1: "Mape Redentore Giudecca 2025",
            2: "Mape Redentore Venezia 2025",
            3: "Mape Carnevale Venezia 2025",
        }[evento]

        # Apro quello selezionato
        ui.update_accordion("accordion_stats", show=pannello_da_aprire)
    





app = App(app_ui, server)


