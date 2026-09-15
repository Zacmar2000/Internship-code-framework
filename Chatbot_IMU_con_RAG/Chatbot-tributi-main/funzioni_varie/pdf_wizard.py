from funzioni_varie.function_extraction import *

from collections import defaultdict
import html
import uuid
import traceback
import threading
from typing import Optional, Dict, Any
import streamlit as st


#######################
# Funzioni necessarie #
#######################

def path_for_extraction_cached(pdf_path, category="raw"):
    """
    Restituisce un percorso stabile per il salvataggio della cache,
    usando la cartella Wizard_extracted invece di Extracted, 
    ma mantenendo la category.
    """
    output_root = Path("Wizard_extracted") / category

    # Usa solo il nome del file originale, non tutto il path
    pdf_name = Path(pdf_path).stem  # es. "documento.pdf" -> "documento"
    pdf_path_extracted = output_root / f"{pdf_name}.pkl"

    # crea le cartelle se non esistono
    pdf_path_extracted.parent.mkdir(parents=True, exist_ok=True)

    return pdf_path_extracted


def substitute_tables_only(elements, api_key, pdf_path, gemini = True, model = "gemma-3-27b-it", y_number = None, num_tolerance = 0.001,
                      y_header = None, header_tolerance = 0.05):
    """
    Trasformiamo le tabelle estratte in testo usando un LLM

    y_number comprende i valori y_min e y_max per trovare e eliminare i numeri a fondo pagina
    y_number è nella forma (y_min,y_max)
    """

    if gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    """
    Funzione per chiedere al LLM di trasformare in testo le tabelle
    """

    def dataframe_to_semantic_text(df, model = model):
        prompt = f"""
        Trasforma la seguente tabella in una descrizione in linguaggio naturale.

        REGOLE OBBLIGATORIE:
        - Non aggiungere informazioni non presenti nella tabella.
        - Non fare inferenze, interpretazioni o spiegazioni.
        - Non correggere numeri o valori.
        - Non riassumere.
        - Non omettere alcun valore.
        - Non introdurre commenti, conclusioni o frasi generiche.
        - Non usare conoscenza esterna.

        OBIETTIVO:
        Convertire ogni riga della tabella in testo mantenendo:
        - Tutti i valori presenti.
        - Tutte le relazioni tra colonne e righe.
        - La gerarchia tra macro-categorie e sotto-categorie.

        GESTIONE CELLE VUOTE:
        - Se la struttura della tabella suggerisce che i valori vuoti rappresentano una continuazione gerarchica del valore precedente nella stessa colonna, trattali come invariati.
        - Se invece il valore vuoto sembra indicare un dato mancante o non applicabile, non inferire alcun valore.
        - Non fare assunzioni arbitrarie.
        - Se l’intera tabella è vuota, restituisci ESATTAMENTE: ""

        FORMATO DI OUTPUT:
        - Testo continuo o paragrafi.
        - Nessuna introduzione.
        - Nessuna conclusione.
        - Nessun commento esterno.
        - Mantieni simboli speciali (*, †) e note così come appaiono.

        Esempio di output atteso (a scopo illustrativo):
        Alta stagione (1 febbraio – 31 dicembre) — Tipologia: ALBERGHI.
        Per Venezia (Centro Storico, Giudecca e Isole con principale vocazione ricettiva), la tariffa base è pari a € 1,00 intero e € 0,50 ridotto (50%) per 1 stella, e € 2,00 intero e € 1,00 ridotto (50%) per 2 stelle.
        Per Lido e Isole, si applica una tariffa ridotta del 20%, pari a € 0,80 / € 0,40 (ridotto 50%) per 1 stella e € 1,60 / € 0,80 (ridotto 50%) per 2 stelle, con la nota che *la riduzione è del 10% per alberghi a 5 stelle*.
        Per la Terraferma, si applica una tariffa ridotta del 30%, pari a € 0,70 / € 0,30 (ridotto 50%) per 1 stella e € 1,40 / € 0,70 (ridotto 50%) per 2 stelle.

        Tabella:
        {df.to_markdown(index=False)}
        """

        try:
            response = client.chat.completions.create(
                model= model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
        except Exception as e:
            if "temperature" in str(e) and "unsupported" in str(e).lower():
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                raise
            

        return response.choices[0].message.content
    
    """
    Funzione per chiedere al LLM se unire o no tabelle consecutive (come nel caso in cui una tabella sia divisa tra due pagine diverse)
    """
    
    def dataframe_unite_decision(list_df, model = model):
        prompt = """
        Rispondi esclusivamente con 'si' oppure 'no'.

        Le tabelle qui sotto sono state estratte automaticamente da un PDF.
        Alcune tabelle possono essere parti della **stessa tabella logica**, anche se:
        - una contiene solo intestazioni, categorie o livelli gerarchici
        - un'altra contiene sotto-intestazioni o dati numerici
        - la tabella originale è stata spezzata verticalmente o orizzontalmente
        - le intestazioni possono essere ripetute, incomplete o distribuite su più blocchi

        Devi rispondere 'si' se le tabelle:
        - fanno riferimento allo **stesso contesto** (stessa stagione, periodo, argomento)
        - sono **semanticamente dipendenti** (una definisce le categorie dell'altra)
        - insieme ricostruiscono **un'unica tabella logica**
        - mantengono lo stesso ordine concettuale delle colonne o delle righe

        Devi rispondere 'no' se le tabelle:
        - descrivono periodi, stagioni o contesti diversi
        - rappresentano strutture autonome
        - hanno categorie non compatibili o non correlabili

        Ignora:
        - righe vuote
        - differenze di formattazione
        - intestazioni ridondanti dovute all’estrazione automatica

        Non fornire spiegazioni.
        """

        df_txt = ""

        for i,df in enumerate(list_df):
            df_txt += f"""Tabella {i+1}:
            {df.to_markdown(index=False)}
        """

        try:
            response = client.chat.completions.create(
                model= model,
                messages=[{"role": "user", "content": prompt + df_txt}],
                temperature=0
            )
        except Exception as e:
            if "temperature" in str(e) and "unsupported" in str(e).lower():
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt + df_txt}],
                )
            else:
                raise
    
        return response.choices[0].message.content
    
    """
    Crezione di elements (testo estratto) "pulito", togliendo header, footer e eventuali numeri di pagina e sostituendo le tabelle con il rispettivo testo ottenuto
    """
    
    clean_elements = elements.copy()  # lavoro su copia


    def is_page_number(element, num_tolerance = num_tolerance):
        if isinstance(element, dict):
            return None
        if y_number is None:
            return False
        else:
            coords = element.metadata.coordinates
            y_values = [y for x, y in coords.points]

            # Calcolo y_min e y_max
            y_min = min(y_values)
            y_max = max(y_values)

            return (y_min <= (y_number[0] + num_tolerance)  and y_max <= (y_number[1] + num_tolerance))
        
    
    def is_header(element, header_tolerance = header_tolerance, keep_intro = True):
        if y_header is None:
            return False
        if isinstance(element, dict):
            return None
        if keep_intro and getattr(element.metadata, "page_number", 1) == 1:
            return False
        else:
            coords = element.metadata.coordinates
            y_values = [y for x, y in coords.points]

            # Calcolo y_min e y_max
            y_min = min(y_values)
            y_max = max(y_values)

            return (y_min >= (y_header[0] - header_tolerance)  and y_max >= (y_header[1] - header_tolerance))

    
    

    table_map = {}

    for idx, el in enumerate(clean_elements):
        if isinstance(el, dict) and "df" in el:
            table_map[id(clean_elements[idx])] = el["df"]

    # Function to see if there is a possible table that is part of the precedent

    def next_table_index(start, clean_elements):
        j = start + 1
        while j < len(clean_elements) and (is_page_number(clean_elements[j]) or 
                                           is_header(clean_elements[j])):
            j += 1
        return j if j < len(clean_elements) and id(clean_elements[j]) in table_map else None


    i = 0

    changes_list = []

    while i < len(clean_elements):

        el = clean_elements[i]

        if id(el) not in table_map:
            i += 1
            continue

        df_list = [table_map[id(el)]]
        i_list = [i]
        final_i = i

        while True:
            next_i = next_table_index(final_i, clean_elements)
            if next_i is None:
                break

            df_tmp = table_map[id(clean_elements[next_i])]

            answer_norm = ""
            last_exception = None

            for _ in range(5):
                try:
                    answer = dataframe_unite_decision(df_list + [df_tmp], model = model)
                    if answer:
                        answer_norm = normalize_text(answer)
                        print(answer_norm)
                        if answer_norm in ["si", "no"]:
                            break
                except Exception as e:
                    last_exception = e

                time.sleep(10)
            else:
                raise RuntimeError("API fallita dopo 5 tentativi") from last_exception

            if answer_norm == "si":
                df_list.append(df_tmp)
                i_list.append(next_i)
                final_i = next_i
            else:
                break
        
        semantic_text = dataframe_to_semantic_text(pd.concat(df_list), model = model)
        new_text_element = Text(
            text=semantic_text,
        )
        new_text_element.metadata.page_number = el['page']
        new_text_element.metadata.coordinates = bbox_to_coordinates_metadata(el['bbox'], pdf_path)

        
        i += (1 + (final_i - i))
        changes_list.append((i_list,new_text_element))

    return changes_list

def substitute_single_table( df, instructions, api_key, gemini, model, reasoning_effort = None):
    if gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    prompt = f"""
    Trasforma la seguente tabella in una descrizione in linguaggio naturale.

    REGOLE OBBLIGATORIE:
    - Non aggiungere informazioni non presenti nella tabella.
    - Non fare inferenze, interpretazioni o spiegazioni.
    - Non correggere numeri o valori.
    - Non riassumere.
    - Non omettere alcun valore.
    - Non introdurre commenti, conclusioni o frasi generiche.
    - Non usare conoscenza esterna.

    OBIETTIVO:
    Convertire ogni riga della tabella in testo mantenendo:
    - Tutti i valori presenti.
    - Tutte le relazioni tra colonne e righe.
    - La gerarchia tra macro-categorie e sotto-categorie.

    GESTIONE CELLE VUOTE:
    - Se la struttura della tabella suggerisce che i valori vuoti rappresentano una continuazione gerarchica del valore precedente nella stessa colonna, trattali come invariati.
    - Se invece il valore vuoto sembra indicare un dato mancante o non applicabile, non inferire alcun valore.
    - Non fare assunzioni arbitrarie.
    - Se l’intera tabella è vuota, restituisci ESATTAMENTE: ""

    FORMATO DI OUTPUT:
    - Testo continuo o paragrafi.
    - Nessuna introduzione.
    - Nessuna conclusione.
    - Nessun commento esterno.
    - Mantieni simboli speciali (*, †) e note così come appaiono.

    Tabella:
    {df.to_markdown(index=False)}
    """

    if instructions:
        prompt += f"""
        Istruzioni specifiche per questa tabella:

        {instructions}
        """

    request_params = {
        "model": model,
        "input": prompt,
    }

    # Se è un modello reasoning
    if reasoning_effort is not None:
        request_params["reasoning"] = {
            "effort": reasoning_effort
        }
    else:
        request_params["temperature"] = 0

    response = client.responses.create(**request_params)

    return response.output_text

def clean_extraction(elements, changes_list, y_number = None, num_tolerance = 0.001,
                      y_header = None, header_tolerance = 0.05, use_image = False):
    
    clean_elements = elements.copy()

    if changes_list is None:
        changes_list_sorted = []
    else:
        changes_list_sorted = sorted(changes_list, key=lambda x: x[0][0])
    
    # Iteriamo all'indietro
    for idx, el in reversed(changes_list_sorted):
        start = idx[0]
        end = idx[-1] + 1 
        clean_elements[start:end] = [el]

    def is_page_number(element, num_tolerance = num_tolerance):
        if isinstance(element, dict):
            return None
        if y_number is None:
            return False
        else:
            coords = element.metadata.coordinates
            y_values = [y for x, y in coords.points]

            # Calcolo y_min e y_max
            y_min = min(y_values)
            y_max = max(y_values)

            return (y_min <= (y_number[0] + num_tolerance)  and y_max <= (y_number[1] + num_tolerance))
        
    
    def is_header(element, header_tolerance = header_tolerance, keep_intro = True):
        if y_header is None:
            return False
        if isinstance(element, dict):
            return None
        if keep_intro and getattr(element.metadata, "page_number", 1) == 1:
            return False
        else:
            coords = element.metadata.coordinates
            y_values = [y for x, y in coords.points]

            # Calcolo y_min e y_max
            y_min = min(y_values)
            y_max = max(y_values)

            return (y_min >= (y_header[0] - header_tolerance)  and y_max >= (y_header[1] - header_tolerance))
        
    

    clean_elements = [
        el for el in clean_elements
        if not (is_page_number(el) or is_header(el))
    ]

    if not use_image:
        clean_elements = [
            el for el in clean_elements if type(el).__name__ != "Image"
        ]

    def fully_unescape(text, max_iter=10):
        prev = text
        curr = prev
        for _ in range(max_iter):
            curr = html.unescape(prev)
            if curr == prev:
                break
            prev = curr
        return curr

    for el in clean_elements:
        el.apply(replace_unicode_quotes)
        if hasattr(el, "text") and el.text:
            el.text = fully_unescape(el.text)
    
        
    return clean_elements

# ---------- Funzioni di utilità ----------

def normalize(s):
    return " ".join(s.strip().split()).rstrip(".").lower()

PUNCT_RE = re.compile(r'([,.:;!?])')
MIN_LEN = 30

def progressive_chunks(text, min_len):
    """Ritorna sottostringhe progressive, dalla più lunga alla più corta, spezzate per punteggiatura."""
    parts = PUNCT_RE.split(text)
    chunks = []
    current = ""
    for i in range(0, len(parts), 2):
        current += parts[i]
        if i + 1 < len(parts):
            current += parts[i + 1]
        normalized = current.strip()
        if len(normalized) >= min_len:
            chunks.append(normalized)
    return chunks[::-1]

def get_remainder(full, matched):
    if not full.startswith(matched):
        return None
    remainder = full[len(matched):]
    return remainder.lstrip(" ,.:;")






def unlock_next_page(num = 1, update = True):
    next_page = st.session_state.current_page + num
    if next_page < st.session_state.total_pages:
        st.session_state.unlocked_pages.add(next_page)
        if update:
            st.session_state.current_page += num

def unlock_paragraph(num):
    if num < st.session_state.total_pages:
        st.session_state.unlocked_pages.add(num)
        st.session_state.current_page = num









#####################
# Wizard principale #
#####################










class PDFExtractionWizard:
    def __init__(self, pdf_path, api_key=None, gemini=False, model="gpt-4.1-nano",
                 table_overlap=0.7, page_start=None, page_end=None, streamlit = False):
        self.pdf_path = pdf_path
        self.api_key = api_key
        self.gemini = gemini
        self.model = model
        self.table_overlap = table_overlap
        self.page_start = page_start
        self.page_end = page_end
        self.streamlit = streamlit

        self.use_image = None
        self.elements = None
        self.elements_reworked = None
        self.clean_elements_not_final = None
        self.clean_elements = None
        self.header_footer_data = None
        self.camelot_data = None
        self.changes_list = None
        self.duplicate_list = None
        self.duplicate_decisions = None

        self.title = None

    def get_theme(self, dark_mode):
        if dark_mode:
            return {
                "bg_main": "#181818",     # sfondo generale widget
                "bg_panel": "#1e1e1e",    # pannelli interni
                "fg": "#e6e6e6",          # testo
                "border": "#333",
                "accent": "#3A5F0B"
            }
        else:
            return {
                "bg_main": "#ffffff",
                "bg_panel": "#e6e6e6a2",
                "fg": "#000000",
                "border": "#ddd",
                "accent": "#90CF3D"
            }


    def apply_theme(self, container, dark_mode):
        theme = self.get_theme(dark_mode)

        container.layout.background_color = theme["bg_main"]
        container.layout.padding = "5px"

        return theme
    
    def get(self, name):
        return getattr(self, name, None)


    # -----------------------
    # Step 1: Caricamento/estrazione testo
    # -----------------------
    def load_elements(self):
        pdf_path_raw = path_for_extraction_cached(self.pdf_path, category="raw")
        pdf_path_clean = path_for_extraction_cached(self.pdf_path, category="clean")
        
        if os.path.exists(pdf_path_raw):
            with open(pdf_path_raw, "rb") as f:
                self.elements = pickle.load(f)
        else:
            if self.streamlit:
                with st.spinner("Estrazione in corso..."):
                    self.elements = extract_text_unstructured(self.pdf_path, self.page_start, self.page_end)
            else:
                self.elements = extract_text_unstructured(self.pdf_path, self.page_start, self.page_end)
            with open(pdf_path_raw, "wb") as f:
                pickle.dump(self.elements, f)

        if not self.streamlit:
            print("Step 1: elementi caricati ✅")
        else:
            unlock_next_page()

        if os.path.exists(pdf_path_clean):
            with open(pdf_path_clean, "rb") as f:
                self.clean_elements = pickle.load(f)
            unlock_paragraph(num = 7)

        st.rerun()

    # -----------------------
    # Step 2: Header/Footer interattivo
    # -----------------------

    def step_header_footer_streamlit(self):
        """
        Streamlit version: header e footer separati fianco a fianco.
        Possibilità di disabilitare singolarmente header o footer.
        """

        if "hf_state" not in st.session_state or not st.session_state.hf_state:
            st.session_state.hf_state = {
                "header": {"tolerance": 0.01, "y_number": None, "enabled": True},
                "footer": {"tolerance": 0.01, "y_number": None, "enabled": True},
            }

        state = st.session_state.hf_state

        # Dark mode
        dark_mode = st.session_state.dark_mode

        bg_color = "#3a3a3a" if dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if dark_mode else "black"

        col1, spacer, col2 = st.columns([10, 2, 10])

        # FOOTER

        with col1:
            st.subheader("Footer")

            footer_state = state["footer"]


            # CASO DISABILITATO

            if not footer_state.get("enabled", True):

                st.info("Rimozione Footer disabilitata")

                if st.button("Attiva rimozione Footer"):
                    state["footer"] = {
                        "tolerance": 0.01,
                        "y_number": None,
                        "enabled": True
                    }
                    st.rerun()


            # CASO ABILITATO

            else:
                footer_tol = st.slider(
                    "Tolerance Footer",
                    min_value=0.0,
                    max_value=0.1,
                    value=footer_state.get("tolerance", 0.01),
                    step=0.001,
                    format="%.3f"
                )

                _, footer_lines = check_y_number(
                    self.elements,
                    footer_tol,
                    header=False,
                    print_elements=False,
                    extract_text=True
                )

                st.markdown(
                    f"<div style='background-color:{bg_color}; color:{fg_color}; "
                    f"padding:10px; font-family:monospace; border-radius:6px;'>"
                    f"{'<br>'.join(html.escape(line) for line in footer_lines)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                col_f1, col_f2 = st.columns(2)

                with col_f1:
                    if st.button("Salva Footer"):
                        footer_state["tolerance"] = footer_tol
                        footer_state["y_number"] = check_y_number(
                            self.elements,
                            footer_tol,
                            header=False,
                            print_elements=False
                        )
                        st.success("Footer salvato ✅")

                with col_f2:
                    if st.button("Non rimuovere Footer"):
                        state["footer"] = {
                            "tolerance": None,
                            "y_number": None,
                            "enabled": False
                        }
                        st.rerun()

        # HEADER

        with col2:
            st.subheader("Header")

            header_state = state["header"]


            # CASO DISABILITATO

            if not header_state.get("enabled", True):

                st.info("Rimozione Header disabilitata")

                if st.button("Attiva rimozione Header"):
                    state["header"] = {
                        "tolerance": 0.01,
                        "y_number": None,
                        "enabled": True
                    }
                    st.rerun()
            

            # CASO ABILITATO

            else:
                header_tol = st.slider(
                    "Tolerance Header",
                    min_value=0.0,
                    max_value=0.1,
                    value=state["header"]["tolerance"],
                    step=0.001,
                    format="%.3f",
                    disabled=not state["header"]["enabled"]
                )

                _, header_lines = check_y_number(
                    self.elements,
                    header_tol,
                    header=True,
                    print_elements=False,
                    extract_text=True
                )

                st.markdown(
                    f"<div style='background-color:{bg_color}; color:{fg_color}; padding:10px; "
                    f"font-family:monospace; border-radius:6px;'>"
                    f"{'<br>'.join(html.escape(line) for line in header_lines)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                col_h1, col_h2 = st.columns(2)

                with col_h1:
                    if st.button("Salva Header"):
                        if state["header"]["enabled"]:
                            state["header"]["tolerance"] = header_tol
                            state["header"]["y_number"] = check_y_number(
                                self.elements,
                                header_tol,
                                header=True,
                                print_elements=False
                            )
                            st.success("Header salvato ✅")

                with col_h2:
                    if st.button("Non rimuovere Header"):
                        state["header"] = {
                            "tolerance": None,
                            "y_number": None,
                            "enabled": False
                        }
                        st.rerun()


        # Salvataggio finale globale


        def section_valid(section):
            # Se è disabilitata è valida
            if not section.get("enabled", True):
                return True

            # Se è attiva deve avere y_number salvato
            return section.get("y_number") is not None


        header_valid = section_valid(state["header"])
        footer_valid = section_valid(state["footer"])

        all_valid = header_valid and footer_valid

        if st.button("Conferma configurazione", disabled=not all_valid):
            self.header_footer_data = state
            st.success("Configurazione salvata")
            unlock_next_page()
            st.rerun()

    # -----------------------
    # Step 2.5: Controllo immagini
    # -----------------------

    def check_image_text_streamlit(self):
        assert self.elements is not None

        # Trova testi nelle immagini
        text_images = [
            el.text.strip() 
            for el in self.elements 
            if hasattr(el, "text") and el.text.strip() and type(el).__name__ == "Image"
        ]

        if not text_images:
            st.success("Nessuna immagine con testo")
            unlock_next_page()
            st.rerun()
            return

        # Inizializza stato Streamlit
        if "image_text_state" not in st.session_state:
            st.session_state.image_text_state = {
                "use_image": False,
            }

        state = st.session_state.image_text_state

        dark_mode = getattr(st.session_state, "dark_mode", True)
        bg_color = "#3a3a3a" if dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if dark_mode else "black"

        st.subheader("Testo trovato nelle immagini")

        # Mostra testi trovati
        st.markdown(
            f"<div style='background-color:{bg_color}; color:{fg_color}; "
            f"padding:10px; font-family:monospace; border-radius:6px;'>"
            f"<b>Testi trovati nelle immagini:</b><br><br>"
            f"{'<br><br>'.join(html.escape(line) for line in text_images)}"
            f"</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        # Pulsanti
        with col1:
            if st.button("Usa testo immagini"):
                state["use_image"] = True
                self.use_image = True
                st.success("Testo immagini selezionato ✅")
                unlock_next_page()
                st.rerun()

        with col2:
            if st.button("Ignora testo immagini"):
                state["use_image"] = False
                self.use_image = False
                st.warning("Testo immagini ignorato ⚠️")
                unlock_next_page()
                st.rerun()

    # ---------------------------
    # Step 3: Camelot interattivo
    # ---------------------------

    def step_camelot_tables_streamlit(self):

        if "camelot_state" not in st.session_state:
            st.session_state.camelot_state = {
                "dark_mode": getattr(st.session_state, "dark_mode", True),
                "camelot_data_final": []
            }

        if "data_cache" not in st.session_state:
            st.session_state.data_cache = {}

        if "camelot_tables" not in st.session_state.data_cache:
            st.session_state.data_cache["camelot_tables"] = {}

        state = st.session_state.camelot_state
        camelot_data_final = state["camelot_data_final"]
        dark_mode = st.session_state.dark_mode

        bg_color = "#3a3a3a" if dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if dark_mode else "black"

        st.subheader("Scelta delle tabelle da trasformare in linguaggio naturale")

        # --- Selezione flavor ---
        flavor = st.selectbox("Flavor", ["lattice", "stream", "network", "hybrid"], index=0)

        # --- Recupera dalla cache o estrai ---
        if flavor in st.session_state.data_cache["camelot_tables"]:
            camelot_data = st.session_state.data_cache["camelot_tables"][flavor]
        else:
            with st.spinner("Estrazione tabelle in corso..."):
                try:
                    camelot_data = extract_tables(self.pdf_path, self.page_start, self.page_end, flavor)
                except Exception as e:
                    camelot_data = []
                    st.error(f"Errore estrazione flavor '{flavor}'")
                st.session_state.data_cache["camelot_tables"][flavor] = camelot_data

        # --- Filtra tabelle già aggiunte ---
        def is_overlapping(bbox1, bbox2):
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            horizontal_overlap = not (x1_max <= x2_min or x2_max <= x1_min)
            vertical_overlap = not (y1_max <= y2_min or y2_max <= y1_min)
            return horizontal_overlap and vertical_overlap

        def is_new_table(table_dict):
            for existing in camelot_data_final:
                if table_dict['page'] == existing['page'] and is_overlapping(table_dict['bbox'], existing['bbox']):
                    return False
            return True

        filtered_data = [t for t in camelot_data if is_new_table(t)]

        if not filtered_data:
            st.info(f"Tutte le tabelle estratte con {flavor} sono già state aggiunte.")
        
        if "checkboxes" not in st.session_state:
            st.session_state["checkboxes"] = {}

        
        if "checkboxes_interaction" not in st.session_state:
            st.session_state["checkboxes_interaction"] = 0
            

        # --- Pulsanti principali ---
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            if st.button("Aggiungi selezionate"):
                st.session_state["checkboxes_interaction"] += 1
                added_count = 0
                for idx, t in enumerate(filtered_data):
                    if st.session_state["checkboxes"][flavor].get(idx, False):
                        camelot_data_final.append(t)
                        added_count += 1
                st.success(f"Aggiornato con {added_count} tabelle.")
                st.session_state["checkboxes"] = {}
                st.rerun()

        with col2:
            if st.button("Seleziona tutte"):
                st.session_state["checkboxes_interaction"] += 1
                for idx in range(len(filtered_data)):
                    st.session_state["checkboxes"][flavor][idx] = True
                st.rerun()

        with col3:
            if st.button("Deseleziona tutte"):
                st.session_state["checkboxes_interaction"] += 1
                for idx in range(len(filtered_data)):
                    st.session_state["checkboxes"][flavor][idx] = False
                st.rerun()

        with col4:
            if st.button("Rimuovi scelte"):
                st.session_state["checkboxes_interaction"] += 1
                # reset globale di tutti i checkbox di tutti i flavor
                st.session_state["checkboxes"] = {}
                state["camelot_data_final"] = []
                st.rerun()


        # --- Conferma finale ---
        if st.button("Conferma scelte"):
            self.camelot_data = sorted(camelot_data_final, key=lambda x: (x["page"], -x["y_top"]))
            st.success(f"Selezione confermata con {len(camelot_data_final)} tabelle ")
            self.remove_wrong_tables()
            unlock_next_page()
            st.rerun()
        
        if flavor not in st.session_state["checkboxes"]:
            st.session_state["checkboxes"][flavor] = {}

        # --- Checkbox e DataFrame ---
        for idx, t in enumerate(filtered_data):

            if idx not in st.session_state["checkboxes"][flavor]:
                st.session_state["checkboxes"][flavor][idx] = False

            key = f"{flavor}_{idx}_{st.session_state['checkboxes_interaction']}_{idx}"

            with st.container(border=True):

                checked = st.checkbox(
                    f"Pagina {t['page']}",
                    value=st.session_state["checkboxes"][flavor][idx],
                    key=key
                )

                st.dataframe(t["df"], width='stretch')#, use_container_width=True)

            st.session_state["checkboxes"][flavor][idx] = checked


    # -----------------------------------------
    # Step 4: Rimuovere tabelle e sostituire
    # -----------------------------------------
    def remove_wrong_tables(self):
        

        # Rimuove le tabelle selezionate dal PDF e prepara elements reworked
        self.elements_reworked = remove_tables_and_substitute(
            elements=self.elements,
            camelot_data=self.camelot_data,
            pdf_path=self.pdf_path,
            min_iou=self.table_overlap
        )


    # -----------------------------------------
    # Step 5: Trasformazione tabelle in testo
    # -----------------------------------------
    def transform_tables(self):
        if self.header_footer_data:
                y_number = self.header_footer_data['footer']['y_number']
                num_tolerance = self.header_footer_data['footer']['tolerance']
                y_header = self.header_footer_data['header']['y_number']
                header_tolerance = self.header_footer_data['header']['tolerance']
        else:
            y_number = None
            num_tolerance = 0.001
            y_header = None
            header_tolerance = 0.05

        self.changes_list = substitute_tables_only(
            elements=self.elements_reworked,
            api_key=self.api_key,
            pdf_path=self.pdf_path,
            gemini=self.gemini,
            model=self.model,
            y_number=y_number,
            num_tolerance=num_tolerance,
            y_header=y_header,
            header_tolerance=header_tolerance
        )
        if not self.streamlit:
            print(f"Completata sostituzione di {len(self.changes_list)} tabelle con testo semantico.")

    # -----------------------------------------
    # Step 6: Visualizzazione tabelle trasformate
    # -----------------------------------------

    def review_transformed_tables_streamlit(self):

        if self.elements_reworked is None:
            self.elements_reworked = self.elements

        if not self.camelot_data:
            st.success("Nessuna tabella disponibile ---> Già a posto.")
            unlock_next_page()
            st.rerun()
            return
        
        
        # --- Trasformazione iniziale (solo se non già in cache) ---
        if "changes_list" not in st.session_state.data_cache:
            with st.spinner("Trasformazione iniziale delle tabelle..."):
                self.transform_tables()
            st.session_state.data_cache["changes_list"] = self.changes_list

        # Carica dalla cache
        self.changes_list = st.session_state.data_cache["changes_list"]

        if not self.camelot_data or not self.changes_list:
            st.success("Nessuna tabella disponibile ---> Già a posto.")
            unlock_next_page()
            st.rerun()
            return
        
        if "last_instructions_input" not in st.session_state:
            st.session_state["last_instructions_input"] = ""
        
        dark_mode = st.session_state.dark_mode
        
        bg_color = "#3a3a3a" if dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if dark_mode else "black"

        # --- Preparazione mapping tabelle ---
        camelot_data_idx = {}
        idx = 1
        for i in self.changes_list:
            if len(i[0]) == 1:
                camelot_data_idx[f"Tabella {idx}"] = range(idx-1, idx)
            else:
                camelot_data_idx[f"Tabelle {idx}-{idx+len(i[0])-1}"] = range(idx-1, idx+len(i[0])-1)
            idx += len(i[0])

        # --- Modelli disponibili ---
        if self.gemini:
            available_models = [
                "gemma-3-27b-it",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ]
        else:
            available_models = [
                "gpt-4.1-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-5-nano",
                "gpt-5-mini"
            ]

        reasoning_models = ["gpt-5-nano", "gpt-5-mini"]

        # --- Sidebar / Controlli ---
        st.sidebar.title("Controlli")
        table_label = st.sidebar.selectbox("Seleziona tabella", list(camelot_data_idx.keys()))
        show_df = st.sidebar.checkbox("Mostra DataFrame originale", value=False)
        model = st.sidebar.selectbox("Modello", available_models, index=available_models.index(self.model) if self.model in available_models else 0)

        reasoning_effort = "low"
        if model in reasoning_models:
            reasoning_effort = st.sidebar.selectbox("Reasoning:", ["minimal", "low", "medium", "high"], index=1)

        instructions = st.text_area(
            "Istruzioni per la trasformazione",
            height=150,
            key="instructions_input"
        )

        # --- Pulsanti ---
        retransform_button = st.button("Ritrasforma tabella")
        confirm_button = st.button("Conferma e chiudi")

        # --- Identificazione tabelle ---
        idx = list(camelot_data_idx.keys()).index(table_label)
        df_range = camelot_data_idx[table_label]


        def render_panel_dataframe(df, title, bg_color="#1e1e1e", fg_color="#ffffff"):
            st.markdown(" ")
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.dataframe(df, width='stretch')#use_container_width=True)

        def render_panel_text(text, title, bg_color="#1e1e1e", fg_color="#ffffff"):

            unique_id = str(uuid.uuid4()).replace("-", "")

            # Escape per evitare problemi HTML
            safe_text = html.escape(str(text)).replace("\n", "<br>")

            st.markdown(
                f"""
                <style>
                .panel-{unique_id} {{
                    background-color: {bg_color};
                    color: {fg_color};
                    padding: 12px;
                    margin-bottom: 12px;
                    border-radius: 8px;
                    white-space: normal;
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='panel-{unique_id}'>
                    <b>{title}</b><br><br>
                    {safe_text}
                </div>
                """,
                unsafe_allow_html=True
            )


        # --- Mostra output ---
        def display_tables():
            assert self.camelot_data is not None
            assert self.changes_list is not None
            if show_df:
                for df_idx in df_range:
                    table = self.camelot_data[df_idx]

                    render_panel_dataframe(
                        df=table["df"],
                        title=f"DataFrame {df_idx+1} originale (pagina {table['page']})",
                        bg_color=bg_color,
                        fg_color=fg_color
                    )
            else:
                render_panel_text(
                    text=self.changes_list[idx][1].text,
                    title="Testo trasformato",
                    bg_color=bg_color,
                    fg_color=fg_color
                )
        display_tables()

        # --- Funzione di ritrasformazione ---
        def do_retransform():
            assert self.changes_list is not None
            result_container = {"result": None, "error": None}

            def call_llm():
                try:
                    assert self.camelot_data is not None
                    dfs = [self.camelot_data[df_idx]['df'] for df_idx in df_range]
                    result_container["result"] = substitute_single_table(
                        df=pd.concat(dfs),
                        instructions=instructions,
                        api_key=self.api_key,
                        gemini=self.gemini,
                        model=model,
                        reasoning_effort=reasoning_effort if model in reasoning_models else None
                    )
                except Exception as e:
                    result_container["error"] = e # pyright: ignore[reportArgumentType]

            thread = threading.Thread(target=call_llm)
            with st.spinner("Trasformazione in corso..."):
                thread.start()
                thread.join(timeout=180)

            # --- Gestione risultati ---
            if thread.is_alive():
                st.error("Timeout: il modello ha impiegato troppo tempo.")
            elif result_container["error"]:
                st.error(f"❌ Errore durante la ritrasformazione:\n{result_container['error']}")
                st.text(traceback.format_exc())
            else:
                new_text = result_container["result"]
                if not new_text or not isinstance(new_text, str):
                    st.error("Risposta LLM non valida o vuota.")
                elif len(new_text) > 50000:
                    st.error("Output troppo lungo — possibile hallucination.")
                else:
                    self.changes_list[idx][1].text = new_text
                    st.session_state.data_cache["changes_list"] = self.changes_list
                    st.success(f"✅ Tabella {idx+1} ritrasformata con successo!")
                    st.rerun()

        # Pulsante “Ritrasforma tabella”
        if retransform_button:
            st.session_state["last_instructions_input"] = instructions
            do_retransform()

        if retransform_button or (instructions != st.session_state["last_instructions_input"]):
            st.session_state["last_instructions_input"] = instructions
            do_retransform()

        if confirm_button:
            unlock_next_page()
            st.rerun()


    # -----------------------------------------
    # Step 7: Inserimento tabelle e pulizia
    # -----------------------------------------
    def cleaning_and_table_insertion(self):

        if self.header_footer_data:
            y_number = self.header_footer_data['footer']['y_number']
            num_tolerance = self.header_footer_data['footer']['tolerance']

            y_header = self.header_footer_data['header']['y_number']
            header_tolerance = self.header_footer_data['header']['tolerance']
        else:
            y_number = None
            num_tolerance = 0.0001

            y_header = None
            header_tolerance = 0.0001
        
        if self.use_image is not None:
            use_image = self.use_image
        else:
            use_image = False
        

        clean_elements = clean_extraction(elements= self.elements_reworked,changes_list= self.changes_list, y_number= y_number, num_tolerance= num_tolerance, y_header= y_header, header_tolerance= header_tolerance, use_image= use_image)

        self.clean_elements_not_final = clean_elements

    # -----------------------------------------
    # Step 8: Rimozione righe duplicate
    # -----------------------------------------

    def duplicate_review_streamlit(self):

        if "clean_elements_with_duplicates" not in st.session_state.data_cache:
            self.cleaning_and_table_insertion()
            st.session_state.data_cache["clean_elements_with_duplicates"] = self.clean_elements_not_final

        # Carica dalla cache
        self.clean_elements_not_final = st.session_state.data_cache["clean_elements_with_duplicates"]


        assert self.clean_elements_not_final is not None

        if "duplicated_phrases" not in st.session_state.data_cache:

            problems = []

            for i, el in enumerate(self.clean_elements_not_final):
                if not hasattr(el, "text"): 
                    continue

                curr = el.text.strip()
                if not curr:
                    continue

                curr_n = normalize_text(normalize(curr), remove_accents=False)
                curr_variants = [curr_n] + progressive_chunks(curr_n, MIN_LEN)
                curr_variants = list(set(curr_variants))

                if i == 0:
                    continue

                prev_el = self.clean_elements_not_final[i-1]
                prev_n = normalize_text(normalize(prev_el.text), remove_accents=False)

                for variant in curr_variants:
                    if prev_n.endswith(variant) or (len(variant) >= MIN_LEN and variant in prev_n):
                        info_match = {
                            'idx': i,
                            'original': prev_n,
                            'matched': variant,
                            'tipo': 'curr_in_prev'
                        }
                        problems.append(info_match)
                        break

                    elif curr_n.startswith(prev_n) or (len(prev_n) >= MIN_LEN and prev_n in curr_n):
                        problems.append({
                            'idx': i,
                            'original': curr_n,
                            'matched': prev_n,
                            'tipo': 'prev_in_curr'
                        })
                        break

            st.session_state.data_cache["duplicated_phrases"] = problems
        
        problems = st.session_state.data_cache["duplicated_phrases"]

            

        if not problems:
            st.success("Nessuna duplicazione trovata")
            self.clean_elements = self.clean_elements_not_final
            unlock_next_page()
            st.rerun()
            return

        if "remaining" not in st.session_state:
            st.session_state.remaining = problems.copy()
            st.session_state.decisions = {}
            st.session_state.current_problem_index = 0

        remaining = st.session_state.remaining


        # Colori

        def get_colors():
            if st.session_state.dark_mode:
                return {
                    "bg": "#1e1e1e",
                    "fg": "#e6e6e6",
                    "matched": "#8D2800",
                    "original": "#3A5F0B"
                }
            else:
                return {
                    "bg": "#e6e6e6a2",
                    "fg": "black",
                    "matched": "orange",
                    "original": "#90CF3D"
                }

        colors = get_colors()


        if not remaining:
            st.success("Revisione completata ✅")

            self.duplicate_list = problems
            self.duplicate_decisions = st.session_state.decisions

            el_to_remove = []
            el_to_replace = {}

            for prob in problems:
                idx = prob['idx']
                
                # Se l'utente NON ha deciso di rimuovere → salta
                if self.duplicate_decisions.get(idx, True):  # default True = tieni
                    continue
                
                # Se esiste remainder → sostituisci
                remainder = prob.get("remainder")
                if remainder:
                    el_to_replace[idx] = remainder
                else:
                    el_to_remove.append(idx)

            assert self.clean_elements_not_final is not None

            self.clean_elements = []

            for i, el in enumerate(self.clean_elements_not_final):
                if i in el_to_replace:
                    new_el = copy.copy(el)
                    new_el.text = el_to_replace[i]
                    self.clean_elements.append(new_el)
                elif i not in el_to_remove:
                    self.clean_elements.append(el)

            unlock_next_page()
            st.rerun()

            return


        # Selezione frase

        options = [f"Frase {p['idx']+1}" for p in remaining]
        selected = st.selectbox("Seleziona frase:", options, index=st.session_state.current_problem_index)
        problem_idx = options.index(selected)
        st.session_state.current_problem_index = problem_idx

        p = remaining[problem_idx]
        idx = p['idx']
        matched = p.get('matched')
        original = p.get('original')

        # Highlight

        def highlight_text(full_text, highlight_word, color):
            if not highlight_word:
                return html.escape(full_text)

            pattern = re.escape(highlight_word)
            return re.sub(
                pattern,
                rf'<span style="background-color:{color}; padding:2px; border-radius:3px;">\g<0></span>',
                html.escape(full_text),
                flags=re.IGNORECASE
            )

        context_range = range(max(0, idx-2), min(len(self.clean_elements_not_final), idx+3))

        html_block = f"""
        <div style='
            background-color:{colors["bg"]};
            color:{colors["fg"]};
            padding:15px;
            border-radius:8px;
            font-family:monospace;
        '>
        """

        for i in context_range:
            text = self.clean_elements_not_final[i].text

            if i == idx:
                text = highlight_text(text, matched, colors["matched"])
                html_block += f"<p><b>➤ {text}</b></p>"

            elif i == idx-1 and matched:
                text = highlight_text(text, original, colors["original"])
                html_block += f"<p>{text}</p>"

            else:
                html_block += f"<p>{html.escape(text)}</p>"

        html_block += "</div>"

        st.markdown(html_block, unsafe_allow_html=True)


        # Bottoni decisione

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Tieni"):
                real_idx = p['idx']
                st.session_state.decisions[real_idx] = True
                st.session_state.remaining.pop(problem_idx)
                st.session_state.current_problem_index = 0
                st.rerun()

        with col2:
            if st.button("❌ Rimuovi"):
                real_idx = p['idx']
                st.session_state.decisions[real_idx] = False
                st.session_state.remaining.pop(problem_idx)
                st.session_state.current_problem_index = 0
                st.rerun()

    
    # -----------------------
    # Step 9: Salvataggio clean_text
    # -----------------------
    def save_clean_elements(self):
        pdf_path_clean = path_for_extraction_cached(self.pdf_path, category="clean")
        

        if not pdf_path_clean.exists():
            message = "Salvare il testo estratto pulito?"
        else:
            message = "Sovrascrivere il testo pulito estratto in precedenza?"

        st.markdown(f"### {message}")

        bg_color = "#3a3a3a" if st.session_state.dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if st.session_state.dark_mode else "black"

        # --- Unico riquadro scrollabile con tutto il testo ---
        assert self.clean_elements is not None

        pages_dict = defaultdict(list)
        for el in self.clean_elements:
            page = el.metadata.page_number
            pages_dict[page].append(el.text)

        options = ["Testo completo"] + sorted(pages_dict.keys())
        selected_page = st.selectbox("Seleziona la pagina da visualizzare", options)

        # --- Determina quale testo mostrare ---
        if selected_page == "Testo completo":
            # Tutte le pagine concatenate
            header_html = f"<b>Testo completo</b><br><br>"
            page_text = ""
            for page_num in sorted(pages_dict.keys()):
                page_text += f"--- Pagina {page_num} ---\n"
                page_text += "\n".join(pages_dict[page_num])
                page_text += "\n\n"

            if len(page_text)>4:
                page_text = page_text[:-4]
        else:
            header_html = f"<b>Pagina {selected_page}</b><br><br>"
            page_text = "\n".join(pages_dict[selected_page])

        st.markdown(
            f"""
            <div style="
                background-color:{bg_color}; 
                color:{fg_color};
                padding:12px; 
                border-radius:8px; 
                max-height:400px; 
                overflow-y:auto;
                white-space: pre-wrap;
                font-family: monospace;
                margin-bottom:10px;
            ">
                {header_html}
                {page_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Bottone per salvare
        if st.button("Conferma salvataggio"):

            with open(pdf_path_clean, "wb") as f:
                pickle.dump(self.clean_elements, f)

            st.success(f"Testo salvato ✅")
            unlock_next_page()
            st.rerun()


    # -----------------------
    # Step 10: Titolo del documento
    # -----------------------

    def create_title(self):
        # Genera titolo automatico se non già presente
        if "title" not in st.session_state.data_cache:
            with st.spinner("Generazione titolo..."):
                st.session_state.data_cache["title"] = get_title_document(
                    self.clean_elements,
                    api_key=self.api_key,
                    gemini=self.gemini,
                    model=self.model,
                    page_start=self.page_start
                ).get('title', '')

        self.title = st.session_state.data_cache["title"]

        dark_mode = getattr(st.session_state, "dark_mode", True)
        bg_color = "#3a3a3a" if dark_mode else "#e6e6e6a2"
        fg_color = "#e6e6e6" if dark_mode else "black"

        st.subheader("Scelta del titolo del documento")

        # Mostra il titolo generato
        st.markdown(
            f"<div style='background-color:{bg_color}; color:{fg_color}; "
            f"padding:10px; font-family:monospace; border-radius:6px;'>"
            f"{html.escape(self.title)}"
            f"</div>",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        # Pulsante "Va bene"
        with col1:
            if st.button("Conferma titolo"):
                st.success("Il titolo è stato confermato con successo.")
                unlock_next_page()
                st.rerun()

        # Pulsante "Non va bene"
        with col2:
            if st.button("Rifiuta titolo"):
                st.warning("Titolo rifiutato. Puoi inserire manualmente un titolo.")
                st.session_state.manual_title_active = True

        with col3:
            if st.button("Ricrea default"):
                del st.session_state.data_cache["title"]
                st.session_state.manual_title_active = False
                st.rerun()

        # Mostra input per titolo manuale se necessario
        if st.session_state.get("manual_title_active", False):
            manual_title = st.text_input("Inserisci titolo manuale", value="")
            if manual_title:
                st.session_state.data_cache["title"] = manual_title
                self.title = manual_title
                st.success(f"Titolo manuale salvato ✅: {manual_title}")
                # Una volta salvato, puoi disattivare il campo
                st.session_state.manual_title_active = False

                unlock_next_page()
                st.rerun()
    
    # -----------------------
    # Step 11: Scelta del metodo di suddivisione
    # -----------------------

    def create_tree(self):

        options = ["Normativa","Sommario","Manuale"]

        st.selectbox("Seleziona la modalità per dividere in paragrafi:",options)