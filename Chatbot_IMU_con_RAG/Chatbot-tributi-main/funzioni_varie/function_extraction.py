import re
import os
import time
import numpy as np
import math
import pandas as pd
import json
import pickle
from pathlib import Path
import pdfplumber
import unicodedata
import tiktoken
import camelot
import copy
from collections import defaultdict
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text
from camelot import utils
from unstructured.documents.coordinates import RelativeCoordinateSystem
from unstructured.documents.elements import CoordinatesMetadata
from unstructured.cleaners.core import replace_unicode_quotes

#camelot-py[cv]

from openai import OpenAI

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*max_size.*deprecated.*",
    category=UserWarning
)

from typing import Callable,Optional, Union




# mappa dei sinonimi: normalizza varianti
synonyms = {
    "parte": "parte",
    "titolo": "titolo",
    "capo": "capo",
    "sezione": "sezione",
    "art.": "articolo",
    "articolo": "articolo",
    "art": "articolo",
    "comma": "comma",
    "aggiornamento": "aggiornamento"
}

# pattern regex
all_patterns = {
    "parte": re.compile(r'^PARTE\s+([IVXLCDM]+)', re.IGNORECASE),
    "titolo": re.compile(r'^Titolo\s+([IVXLCDM]+)', re.IGNORECASE),
    "capo": re.compile(r'^Capo\s+([IVXLCDM]+)', re.IGNORECASE),
    "sezione": re.compile(r'^Sezione\s+([IVXLCDM]+)', re.IGNORECASE),
    "articolo": re.compile(r'^(Art\.|Articolo)\s+(\d+[a-z\-]*)', re.IGNORECASE),
    "comma": re.compile(
            r'\(*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|duodevicies|undevicies|vicies|vices semel|vices bis|vices ter))?\.)'
            r'(?:\s*|\(\()',
            re.IGNORECASE
        ),
    "aggiornamento": re.compile(r'^AGGIORNAMENTO\s*\(\s*\d+\s*\)$')
}

SUFFIXES = [
    "bis", "ter", "quater", "quinquies", "sexies", "septies",
    "octies", "novies", "decies", "undecies", "duodecies",
    "terdecies", "quaterdecies", "quinquiesdecies",
    "sexiesdecies", "septiesdecies", "duodevicies",
    "undevicies", "vicies", "vices semel", "vices bis", "vices ter"
]

SUFFIX_RANK = {s: i + 1 for i, s in enumerate(SUFFIXES)}






def path_for_extraction(pdf_path, category = "raw"):
    """
    Definizione Path per salvataggio files
    """

    output_root = Path(f"Extracted/{category}")

    pdf_path_extracted = output_root / Path(pdf_path).with_suffix(".pkl")

    # crea le cartelle se non esistono
    pdf_path_extracted.parent.mkdir(parents=True, exist_ok=True)

    return pdf_path_extracted


def extract_text_unstructured(pdf_path, page_start = None, page_end = None):

    """
    Estrazione del testo da PDF usando Unstructured.
    
    Args:
        pdf_path (str): Percorso del file PDF.
        page_start (int, optional): Numero della prima pagina da estrarre (partendo da 1). Default = 1.
        page_end (int, optional): Numero dell'ultima pagina da estrarre (partendo da 1). Default = ultima pagina.
    
    Returns:
        List[Element]: Lista di elementi estratti dal PDF.
    """

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    if page_start is None:
        page_start = 1
    
    if page_end is None or page_end > total_pages:
        page_end = total_pages

    if page_start > page_end:
        raise ValueError("La prima pagina data è oltre l'ultima pagina considerata")
    

    pages = list(range(page_start, page_end+1))
        


    elements = partition_pdf(filename= pdf_path,
                            strategy="hi_res",
                            languages=['ita','eng'],
                            infer_table_structure= True,
                            )
    
    filtered = [el for el in elements if getattr(el.metadata, "page_number", None) in pages]

    for el in filtered:
        if hasattr(el, "metadata") and getattr(el.metadata, "coordinates", None):
            el.convert_coordinates_to_new_system(
                RelativeCoordinateSystem(), in_place=True
            )

    def y_max(el):
        return max(y for _, y in el.metadata.coordinates.points)

    elements_sorted = sorted(
        filtered,
        key=lambda el: (
            el.metadata.page_number,
            -y_max(el)
        )
    )

    return elements_sorted

def extract_tables(pdf_path, page_start=None, page_end=None, flavor = "lattice"):
    """
    Estrazione delle tabelle da PDF usando Camelot.
    
    Args:
        pdf_path (str): Percorso del file PDF.
        page_start (int, optional): Prima pagina da estrarre (partendo da 1). Default = 1.
        page_end (int, optional): Ultima pagina da estrarre (partendo da 1). Default = ultima pagina.
    
    Returns:
        List[dict]: Lista di tabelle con informazioni su pagina, bbox e DataFrame.
    """
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    if page_start is None:
        page_start = 1
    if page_end is None or page_end > total_pages:
        page_end = total_pages
    if page_start > page_end:
        raise ValueError("La prima pagina indicata è oltre l'ultima pagina disponibile.")

    # Camelot usa stringhe tipo "2-5" o "1,3,5"
    pages_str = f"{page_start}-{page_end}"


    if flavor == "lattice":
        camelot_tables = camelot.read_pdf(  # type: ignore
            pdf_path,
            pages=pages_str,
            flavor= flavor,
            line_scale=40,
        )

    else:
        camelot_tables = camelot.read_pdf(  # type: ignore
            pdf_path,
            pages=pages_str,
            flavor= flavor,
        )

    camelot_data = []
    for t in camelot_tables:
        df = t.df
        if df.map(lambda x: str(x).strip() == "").all().all():
            # Tabella vuota, saltala
            continue
        x1, y1, x2, y2 = t._bbox  # type: ignore
        camelot_data.append({
            "page": t.page,
            "bbox": (x1, y1, x2, y2),
            "y_top": y2,
            "y_bottom": y1,
            "df": t.df
        })
    
    return camelot_data

def normalize_punctuation_spaces(text, punctuation = True, spaces = True, e = True):
    # rimuove spazio PRIMA della punteggiatura
    text = re.sub(r'\s+([,.;:])', r'\1', text)

    # assicura UN solo spazio DOPO la punteggiatura (se non fine stringa)
    text = re.sub(r'([,.;:\)\]\}])\s*', r'\1 ', text)

    if not punctuation:
        text = re.sub(r'([,.;:])', '', text)
    
    if not e:
        text = re.sub(r'\s+e\s+ ', ' ', text)

    # pulizia spazi multipli
    text = re.sub(r'\s+', ' ', text)

    if not spaces:
        text = re.sub(r'\s+',"",text)


    return text.strip()

def normalize_text(testo: str | None, remove_accents: bool = True, remove_new_line: bool = True) -> str:
    if not testo:
        return ""

    if remove_accents:
        testo = unicodedata.normalize("NFKD", testo)
        testo = "".join(c for c in testo if not unicodedata.combining(c))
    else:
        testo = unicodedata.normalize("NFC", testo)

    testo = testo.replace('\uf0b7', ' ')
    testo = testo.replace('\xa0', ' ').replace("–", "-").replace("’", "'")
    
    if remove_new_line:
        testo = re.sub(r'\s+', ' ', testo)
    else:
        testo = '\n'.join(
            re.sub(r'\s+', ' ', riga)
            for riga in testo.splitlines()
        )
    
    testo = normalize_punctuation_spaces(testo)

    return testo.lower().strip()

# ------------------------
# Normalizzazione
# ------------------------

def normalize_camelot_bbox(ct, page_width, page_height):
    x1, y1, x2, y2 = ct['bbox']
    return (x1 / page_width, y1 / page_height, x2 / page_width, y2 / page_height)

def normalize_unstructured_bbox(el):
    coords = getattr(el.metadata, "coordinates", None)
    if not coords or not coords.points:
        return None
    xs = [p[0] for p in coords.points]
    ys = [p[1] for p in coords.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (x_min, y_min, x_max, y_max)

# ------------------------
# IoU verticale
# ------------------------

def bbox_iou_y_only(b1, b2):
    """
    Calcola Intersection over Union tra due bbox usando solo le coordinate Y (verticale)
    """
    y_min, y_max = b1[1], b1[3]
    y_min_tab, y_max_tab = b2[1], b2[3]

    height_b1 = y_max - y_min
    if height_b1 <= 0:
        return 0

    intersection = max(
        0,
        min(y_max, y_max_tab) - max(y_min, y_min_tab)
    )

    return intersection / height_b1

# ------------------------
# Trova match per una tabella
# ------------------------

def find_matching_camelot_table_y(el, camelot_data, page_width, page_height, min_iou=0.7):
    el_bbox = normalize_unstructured_bbox(el)
    if not el_bbox:
        return None

    best_match = None
    best_iou = 0

    for ct in camelot_data:
        if int(ct["page"]) != getattr(el.metadata, "page_number", 1):
            continue
        ct_bbox = normalize_camelot_bbox(ct, page_width, page_height)
        iou = bbox_iou_y_only(el_bbox, ct_bbox)
        if iou >= min_iou and iou > best_iou:
            best_iou = iou
            best_match = ct

    return best_match

# ------------------------
# Funzione principale: sostituzione + inserimento
# ------------------------

def remove_tables_and_substitute(elements, camelot_data, pdf_path, min_iou=0.7):
    """
    Rimuove gli elementi Unstructured che hanno un match e inserisce le tabelle
    Camelot nella posizione verticale corretta (anche senza match).
    """
    _, page_dim = utils.get_page_layout(pdf_path)
    page_width, page_height = page_dim

    # Converti tutte le coordinate Unstructured in RelativeCoordinateSystem
    for el in elements:
        if hasattr(el, "metadata") and getattr(el.metadata, "coordinates", None):
            el.convert_coordinates_to_new_system(
                RelativeCoordinateSystem(), in_place=True
            )

    new_elements = []

    # Per evitare di aggiungere la stessa tabella più volte
    added_tables = set()

    # Raggruppiamo elementi e tabelle per pagina
    pages = {}
    for el in elements:
        page = getattr(el.metadata, "page_number", 1)
        pages.setdefault(page, {"elements": [], "tables": []})
        pages[page]["elements"].append(el)

    for i, ct in enumerate(camelot_data):
        page = int(ct["page"])
        ct["_table_id"] = i  # id univoco
        pages.setdefault(page, {"elements": [], "tables": []})
        pages[page]["tables"].append(ct)

    # Per ogni pagina, match e inserimento
    for page, content in pages.items():
        els = content["elements"]
        tables = content["tables"]

        for el in els:
            match = find_matching_camelot_table_y(
                el,
                tables,
                page_width,
                page_height,
                min_iou=min_iou
            )

            if match:
                table_id = match["_table_id"]
                if table_id not in added_tables:
                    new_elements.append(match)
                    added_tables.add(table_id)
            else:
                # Mantieni elemento originale se non è match
                new_elements.append(el)

        # Aggiungi eventuali tabelle non matchate
        for ct in tables:
            table_id = ct["_table_id"]
            if table_id not in added_tables:
                new_elements.append(ct)
                added_tables.add(table_id)

    # Riordina tutti gli elementi globalmente per pagina e y_min
    def get_y_max(el):
        if hasattr(el, "metadata"):
            bbox = normalize_unstructured_bbox(el)
            if bbox is None:
                return getattr(el.metadata, "page_number", 1), float("-inf")
            return getattr(el.metadata, "page_number", 1), -bbox[3]
        else:
            # Camelot table
            ct_bbox = normalize_camelot_bbox(el, page_width, page_height)
            y_max = ct_bbox[3]
            return int(el["page"]), -y_max

    new_elements.sort(key=get_y_max)

    return new_elements

def bbox_to_coordinates_metadata(bbox, pdf_path):

    _, page_dim = utils.get_page_layout(pdf_path)
    page_width, page_height = page_dim
    
    x1, y1, x2, y2 = bbox
    # normalize to 0-1
    x1_n, y1_n = x1 / page_width, y1 / page_height
    x2_n, y2_n = x2 / page_width, y2 / page_height

    # define points in clockwise order
    points = (
        (np.float64(x1_n), np.float64(y2_n)),  # top-left
        (np.float64(x1_n), np.float64(y1_n)),  # bottom-left
        (np.float64(x2_n), np.float64(y1_n)),  # bottom-right
        (np.float64(x2_n), np.float64(y2_n)),  # top-right
    )

    # create CoordinatesMetadata object
    coords = CoordinatesMetadata(
        points=points,
        system=RelativeCoordinateSystem()
    )
    return coords

def substitute_tables(elements, api_key, pdf_path, gemini = True, model = "gemma-3-27b-it", y_number = None, num_tolerance = 0.001,
                      y_header = None, header_tolerance = 0.05, keep_header_intro = True, use_image = False, ask_for_duplicates = True):

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
        Se la tabella è vuota rispondi semplicemente con una stringa vuota ("").

        Obiettivo principale:
        - Rappresentare in modo completo il contenuto informativo della tabella.
        - Riportare tutti i valori presenti, senza omissioni.
        - Esplicitare chiaramente le macro-categorie e le sotto-categorie che contestualizzano i valori.

        Linee guida:
        - La struttura del testo è libera (frasi, paragrafi o elenchi).
        - Includi i valori di tutte le colonne e righe.
        - Mantieni le relazioni logiche tra colonne.
        - Se una cella è vuota o manca un valore in una colonna, interpreta il dato come invariato rispetto all’ultimo valore non vuoto precedente nella stessa colonna, salvo indicazioni esplicite contrarie.
        - Se presenti, conserva simboli speciali (*, †), note e riferimenti così come compaiono nella tabella.
        - Evita introduzioni, conclusioni o commenti esterni ai dati.
        - Usa un linguaggio naturale chiaro e leggibile.

        Esempio di output atteso (a scopo illustrativo):
        Alta stagione (1 febbraio – 31 dicembre) — Tipologia: ALBERGHI.
        Per Venezia (Centro Storico, Giudecca e Isole con principale vocazione ricettiva), la tariffa base è pari a € 1,00 intero e € 0,50 ridotto (50%) per 1 stella, e € 2,00 intero e € 1,00 ridotto (50%) per 2 stelle.
        Per Lido e Isole, si applica una tariffa ridotta del 20%, pari a € 0,80 / € 0,40 (ridotto 50%) per 1 stella e € 1,60 / € 0,80 (ridotto 50%) per 2 stelle, con la nota che *la riduzione è del 10% per alberghi a 5 stelle*.
        Per la Terraferma, si applica una tariffa ridotta del 30%, pari a € 0,70 / € 0,30 (ridotto 50%) per 1 stella e € 1,40 / € 0,70 (ridotto 50%) per 2 stelle.

        Tabella:
        {df.to_markdown(index=False)}
        """

        response = client.chat.completions.create(
            model= model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

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

    
        response = client.chat.completions.create(
            model= model,
            messages=[{"role": "user", "content": prompt + df_txt}],
            temperature=0
        )
    
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
        
    
    def is_header(element, header_tolerance = header_tolerance, keep_intro = keep_header_intro):
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

    def next_table_index(start, clean_elements):
        j = start + 1
        while j < len(clean_elements) and (is_page_number(clean_elements[j]) or 
                                           is_header(clean_elements[j])):
            j += 1
        return j if j < len(clean_elements) and id(clean_elements[j]) in table_map else None


    i = 0

    while i < len(clean_elements):

        el = clean_elements[i]

        if id(el) not in table_map:
            i += 1
            continue

        df_list = [table_map[id(el)]]
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
                final_i = next_i
            else:
                break
        
        semantic_text = dataframe_to_semantic_text(pd.concat(df_list), model = model)
        new_text_element = Text(
            text=semantic_text,
        )
        new_text_element.metadata.page_number = el['page']
        new_text_element.metadata.coordinates = bbox_to_coordinates_metadata(el['bbox'], pdf_path)

        clean_elements[i:final_i + 1] = [new_text_element]
        i += 1



    clean_elements = [
        el for el in clean_elements
        if not (is_page_number(el) or is_header(el))
    ]

    if not use_image:
        clean_elements = [
            el for el in clean_elements if type(el).__name__ != "Image"
        ]

    client.close()

    def normalize(s):
        return " ".join(s.strip().split()).rstrip(".").lower()

    def chiedi_conferma(da_eliminare, contenitore, motivo, parte_specifica = None):
        print("\n⚠️ POSSIBILE DUPLICATO")
        print(f"Motivo: {motivo}")
        print("-" * 40)
        print("Riga CONTENITORE:")
        print(contenitore)
        print("\nRiga CANDIDATA:")
        print(da_eliminare)
        if parte_specifica is not None:
            print("\n Parte della Riga CANDIDATA:")
            print(parte_specifica)
        print("-" * 40)


        while True:
            scelta = input("Eliminare la riga candidata? (s/n): ").strip().lower()
            if scelta in ("s", "n"):
                return scelta == "s"

    MIN_LEN = 30

    PUNCT_RE = re.compile(r'([,.:;!?])')

    def progressive_chunks(text, min_len):
        """
        Ritorna una lista di sottostringhe progressive,
        dalla più lunga alla più corta, spezzate per punteggiatura.
        """
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

        return chunks[::-1]  # dalla più lunga alla più corta
    
    def get_remainder(full, matched):
        if not full.startswith(matched):
            return None
        remainder = full[len(matched):]
        return remainder.lstrip(" ,.:;")
    

    clean_elements_final = []


    for el in clean_elements:
        if not hasattr(el, "text"):
            continue

        curr = el.text.strip()
        if not curr:
            continue

        curr_n = normalize_text(normalize(curr),remove_accents= False)

        curr_variants = [curr_n] + progressive_chunks(curr_n, MIN_LEN)
        curr_variants = list(set(curr_variants))


        if clean_elements_final:
            prev_el = clean_elements_final[-1]
            prev = prev_el.text.strip()
            prev_n = normalize_text(normalize(prev), remove_accents=False)


            matched_variant = None

            aggiungi = True

            for variant in curr_variants:

                if prev_n.endswith(variant):
                    matched_variant = variant
                    if not ask_for_duplicates:
                        aggiungi = False
                        break
                    elif chiedi_conferma(curr, prev, "la riga è già finale della precedente", variant):
                        aggiungi = False
                        break

                elif len(variant) >= MIN_LEN and variant in prev_n:
                    matched_variant = variant
                    if not ask_for_duplicates:
                        aggiungi = False
                        break
                    elif chiedi_conferma(curr, prev, "contenuta arbitrariamente nella precedente", variant):
                        aggiungi = False
                        break

                elif curr_n.startswith(prev_n):
                    if not ask_for_duplicates:
                        clean_elements_final.pop()
                        break
                    elif chiedi_conferma(prev, curr, "la riga precedente è contenuta in questa"):
                        clean_elements_final.pop()
                        break

                elif len(prev_n) >= MIN_LEN and prev_n in curr_n:
                    if not ask_for_duplicates:
                        clean_elements_final.pop()
                        break
                    elif chiedi_conferma(prev, curr, "la precedente è contenuta arbitrariamente in questa"):
                        clean_elements_final.pop()
                        break

            if aggiungi:
                clean_elements_final.append(el)


            # se siamo qui → c'è stato un match
            if matched_variant and matched_variant != curr_n:
                remainder = get_remainder(curr_n, matched_variant)
                if remainder:
                    new_el = copy.copy(el)
                    new_el.text = remainder
                    clean_elements_final.append(new_el)
        
        else:
            clean_elements_final.append(el)
    
    import html

    def fully_unescape(text, max_iter=10):
        prev = text
        curr = prev
        for _ in range(max_iter):
            curr = html.unescape(prev)
            if curr == prev:
                break
            prev = curr
        return curr # pyright: ignore[reportPossiblyUnboundVariable]

    for el in clean_elements_final:
        el.apply(replace_unicode_quotes)
        if hasattr(el, "text") and el.text:
            el.text = fully_unescape(el.text)


    return clean_elements_final


def check_y_number(elements, tolerance=0.001, header=False, extract_text = False, print_elements=True):

    text_extracted = []

    # Inizializziamo variabili
    if header:
        extreme_y_min = float("-inf")  # per header, cerchiamo il più grande
    else:
        extreme_y_min = float("inf")   # per default, cerchiamo il più piccolo

    element_with_extreme_y = None
    y_number: tuple[float, float] | None = None

    # Iteriamo tutti gli elementi
    for el in elements:
        if hasattr(el, "text") and el.text.strip():
            if not hasattr(el.metadata, "coordinates") or el.metadata.coordinates is None:
                continue
            coords = el.metadata.coordinates
            if not hasattr(coords, "points") or coords.points is None:
                continue

            y_values = [y for x, y in coords.points]
            y_min = min(y_values)
            y_max = max(y_values)

            # Aggiorniamo a seconda del flag header
            if (not header and y_max < extreme_y_min) or (header and y_min > extreme_y_min):
                extreme_y_min = y_max
                element_with_extreme_y = el
                y_number = (y_min, y_max)

    all_elements = []

    # Iteriamo di nuovo per selezionare tutti gli elementi vicino all'estremo
    for el in elements:
        if hasattr(el, "text") and el.text.strip():
            if not hasattr(el.metadata, "coordinates") or el.metadata.coordinates is None:
                continue
            coords = el.metadata.coordinates
            if not hasattr(coords, "points") or coords.points is None:
                continue
            y_values = [y for x, y in coords.points]
            y_min = min(y_values)
            y_max = max(y_values)

            if y_number is not None:
                if (not header and y_min <= (y_number[0] + tolerance) and y_max <= (y_number[1] + tolerance)) or \
                   (header and y_min >= (y_number[0] - tolerance) and y_max >= (y_number[1] - tolerance)):
                    all_elements.append(el)

    if element_with_extreme_y:
        # Creiamo un dizionario per pagine
        page_elements = defaultdict(list)
        for el in all_elements:
            if hasattr(el, "text") and el.text.strip():
                page_num = getattr(el.metadata, "page_number", None)
                if page_num is None:
                    page_num = 0  # pagina sconosciuta
                page_elements[page_num].append(el)

        # Stampa ordinata per pagina
        for page_num in sorted(page_elements.keys()):
            text_extracted.append(f"\n--- Pagina {page_num} ---")
            if print_elements:
                print(f"\n--- Pagina {page_num} ---")
            for el in page_elements[page_num]:
                text_extracted.append(el.text)
                if print_elements:
                    print(el.text)
            text_extracted.append("          ")

        if print_elements:
            print(f"\ny_number = {y_number}")

    else:
        text_extracted.append("Nessun elemento con testo trovato.")
        if print_elements:
            print("Nessun elemento con testo trovato.")


    if extract_text:
        return y_number, text_extracted
    else:
        return y_number


def extraction_tot(pdf_path, api_key, gemini = True, model = "gemma-3-27b-it", table_overlap = 0.7,
                   page_start = None, page_end = None, number_at_end = True, num_tolerance = 0.001,
                   remove_header = False, header_tolerance = 0.05, keep_header_intro = True, use_image = False, ask_for_duplicates = True):

    pdf_path_raw = path_for_extraction(pdf_path, category = "raw")

    if os.path.exists(pdf_path_raw):
        with open(pdf_path_raw, "rb") as f:
            elements = pickle.load(f)
    
    else:
        try:
            elements = extract_text_unstructured(pdf_path, page_start= page_start, page_end= page_end)
        except Exception as e:
            raise RuntimeError("Error on extraction with unstructured:") from e
        with open(pdf_path_raw, "wb") as f:
            pickle.dump(elements, f)
    
    try:
        camelot_data = extract_tables(pdf_path, page_start= page_start, page_end= page_end)
    except Exception as e:
        raise RuntimeError("Error on extraction of Tables with camelot:") from e
    
    try:
        elements_reworked = remove_tables_and_substitute(elements=elements, camelot_data= camelot_data, pdf_path = pdf_path, min_iou= table_overlap)
    except Exception as e:
        raise RuntimeError("Error on extraction of Tables with camelot:") from e

    try:
        if number_at_end:
            y_number = check_y_number(elements= elements_reworked, print_elements= False, tolerance= num_tolerance)
        else:
            y_number = None

        if remove_header:
            y_header = check_y_number(elements= elements_reworked, print_elements= False, tolerance= header_tolerance, header= True)
        else:
            y_header = None

        clean_elements = substitute_tables(elements_reworked ,pdf_path = pdf_path, api_key = api_key, gemini = gemini, model = model,
                                           y_number = y_number, num_tolerance= num_tolerance, y_header = y_header,
                                           header_tolerance = header_tolerance, keep_header_intro = keep_header_intro,
                                           use_image = use_image, ask_for_duplicates= ask_for_duplicates)
    except Exception as e:
        raise RuntimeError("Error on cleaning of the extraction:") from e

    return clean_elements


def get_text_by_page(elements, page, delta=0):
    pages = {page + d for d in range(-delta, delta + 1)}

    candidates = []
    for i, el in enumerate(elements):
        el_page = getattr(el.metadata, "page_number", None)
        if el_page in pages and hasattr(el, "text"):
            candidates.append({
                "text": el.text.strip(),
                "page": el_page,
                "index": i
            })
    return candidates

def parse_llm_json(raw: str):
    # Rimuove ```json e ```
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON non valido:\n{cleaned}") from e

def get_title_document(elements, api_key, gemini = True, model = "gemma-3-27b-it", delta = 0, page_start = None):
    if page_start is None:
        page_start = 1
    first_page = get_text_by_page(elements=elements, page = page_start, delta = delta)

    if gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    prompt = f"""
    Sei un assistente specializzato nell'estrazione di titoli dai documenti. 
    Ti verrà fornita la **prima pagina del documento** come lista di dizionari Python, ciascuno con il formato: {{"text": "...", "page": 1, "index": ...}}.

    Il tuo compito è identificare **solo il titolo del documento**.

    Regole:

    1. Usa **solo il contenuto del campo "text"** di ciascun dizionario.
    2. Non aggiungere commenti, spiegazioni o informazioni non presenti nel testo.
    3. Il titolo deve essere conciso e completo, come appare ufficialmente.
    4. Se il titolo è diviso su più righe, uniscilo in una sola riga.
    6. Se non riesci a identificare un titolo, rispondi con una stringa vuota.
    7. **Rispondi esclusivamente con un JSON valido** del tipo:
    {{"title": "Il titolo identificato"}}
    Nessun testo fuori dal JSON.

    Prima pagina del documento:

    {first_page}
    """

    response = client.chat.completions.create(
        model= model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content

    client.close()

    if raw:
        return parse_llm_json(raw)
    
    else:
        client.close()

        raise ValueError("Errore del LLM nell' assegnazione")





"""
Functions for extracting tree using its summary
"""






def extract_first_pages(pdf_path, max_pages = 7, page_start = None):

    if page_start is None:
        page_start = 1

    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages[(page_start-1):(page_start-1)+max_pages]):
            # Raggruppa caratteri per linea (stesso top)
            y_lines = {}
            full_text.append(f"Pagina n.: {page_index+page_start}")
            for char in page.chars:
                y = round(char['top'], 1)
                if y not in y_lines:
                    y_lines[y] = []
                y_lines[y].append(char)

            # Ricostruisci linee dalla pagina
            for y, chars in sorted(y_lines.items()):
                text = ''.join(c['text'] for c in chars).strip()
                full_text.append({'text': text, "indent": chars[0]['x0']})

    return full_text

def get_summary(full_text, api_key, gemini = True, model = "gemini-2.5-flash-lite"):

    if gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    prompt = f"""
    Sei un assistente estremamente preciso incaricato di estrarre il **sommario/indice** da un documento testuale fornito come stringa `full_text`.

    Regole **rigorose e vincolanti**:

    1. Cerca **esclusivamente** la sezione che rappresenta un indice/sommario (es. "INDICE", "SOMMARIO"), dove compaiono titoli con **numeri di pagina espliciti**.
    2. Il sommario può estendersi su **più pagine consecutive**: devi individuarle e includerle tutte.
    3. **Non dedurre** titoli o pagine dal corpo del testo: usa solo ciò che è presente nell’indice.
    4. Ogni voce dell’indice deve essere classificata in modo **gerarchico**, creando una struttura ad albero.
    5. Se presente un titolo dell'indice (es. "INDICE", "SOMMARIO") non metterlo nell'albero.

    ### Regole di gerarchia

    - Stesso indent indica probabile stesso livello nell'albero. 
    - Titoli di livello superiore (es. "CAPO", "TITOLO", "SEZIONE") sono nodi principali.
    - Sottolivelli (es. "PARTE", "Art.", "Allegato") devono essere inseriti nel campo `children` del nodo padre corretto.
    - Se un titolo non ha sottotitoli, `children` deve essere una lista vuota.
    - Mantieni **l’ordine originale** del sommario.
    - Non saltare livelli e non creare gerarchie non presenti esplicitamente.
    
    ### Continuità della gerarchia tra pagine

    - Un cambio di pagina **NON interrompe mai** la struttura gerarchica.
    - Se un titolo appare su una pagina diversa ma:
    - ha lo stesso livello logico del precedente (es. "Art.", "PARTE", "CAPO"), oppure
    - è visivamente indentato come sottolivello, oppure
    - il nome del titolo indica chiaramente che è un sottolivello del precedente
    allora deve essere inserito come `children` del nodo corretto.

    - La gerarchia deve essere determinata **solo** da:
    1. indentazione o allineamento nel sommario
    2. parole chiave del titolo (CAPO > PARTE > Art. > comma / lettera)
    3. continuità strutturale dell’indice

    - **Non creare nuovi nodi di livello superiore solo perché cambia il numero di pagina.**

    ### Pulizia dei titoli

    - Se titolo e pagina sono uniti (es. "Art. 1 - Oggetto 3"), separali correttamente:
    - title: "Art. 1 - Oggetto"
    - page: 3

    ### Struttura di output (OBBLIGATORIA)

    Restituisci **solo JSON valido**, senza testo aggiuntivo, con questa struttura:

    ```json
    {{
    "sommario_page_start": <numero pagina in cui inizia il sommario>,
    "sommario": [
        {{
        "title": "<titolo>",
        "page": <numero pagina>,
        "children": [
            {{
            "title": "<sottotitolo>",
            "page": <numero pagina>,
            "children": [ ... ]
            }}
        ]
        }}
    ]
    }}


    7. Se non trovi un indice o il documento non contiene titoli, rispondi solo con "Nessun sommario".
    8. Non aggiungere testo libero, spiegazioni o note extra.
    9. Assicurati di includere **tutti i titoli presenti nel sommario**, anche quelli che appaiono dopo la prima pagina.

    Documento da analizzare:
    {full_text}
    """





    response = client.chat.completions.create(
        model= model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content

    client.close()

    if raw:
        if normalize_text(raw) == "nessun sommario":
            return None
        else:
            return parse_llm_json(raw)
    
    else:
        raise ValueError("Errore del LLM nell' assegnazione")

def add_document_field(nodes, document_name):
    for node in nodes:
        node["document"] = normalize_text(document_name,remove_accents= False, remove_new_line= False)
        if "children" in node and isinstance(node["children"], list):
            add_document_field(node["children"], document_name)

def create_tree_summary(pdf_path, elements, title, summary):
    """
    Estrae la leggenda con indent preciso e crea un nodo introduttivo
    con il testo delle pagine precedenti la legenda e, dalla pagina
    della legenda, solo il testo prima della prima keyword.
    """
    intro_text_lines = []
    intro_text_lines_summary = []

    found_match = False

    summary_page = summary['sommario_page_start']
    legend_lines = summary['sommario']

    # Parole chiave da cercare nella pagina della legenda
    keywords = ["sommario", "summary", "indice", "index"]

    if elements:
        for el in elements:
            el_page = getattr(el.metadata, "page_number", 1)
            text_line = el.text.strip()

            if el_page < summary_page:
                # Pagine precedenti: prendi sempre il testo
                intro_text_lines.append(text_line)
            elif el_page == summary_page:
                # Pagina della legenda: prendi testo solo fino alla keyword
                if any(normalize_text(text_line).startswith(k) for k in keywords):
                    # fermati al primo match
                    found_match = True
                    break
                intro_text_lines_summary.append(text_line)
            else:
                # Pagine successive alla legenda: ignora
                continue

    if found_match:
        intro_text_lines.extend(intro_text_lines_summary)

    intro_text = "\n".join(intro_text_lines).strip()

    intro_node = {
        'title': f'introduzione_{os.path.basename(pdf_path)}',
        'page': 1,
        'text': intro_text,
        'children': []
    }

    # Inserisci nodo introduttivo all’inizio della legenda
    full_tree = [intro_node] + legend_lines


    add_document_field(full_tree,normalize_text(title, remove_accents = False, remove_new_line= False))

    return full_tree



def get_tree_titles(tree, pdf_path, skip_intro = True, page_offset = 0):

    list_titles = []

    for node in tree:
        if skip_intro and node['title'] == f'introduzione_{os.path.basename(pdf_path)}':
            continue

        page = node.get('page') + page_offset

        title_el = {'title': node['title'],
                    'page': page}
        list_titles.append(title_el)

        if node.get('children'):
            list_titles += get_tree_titles(node['children'], pdf_path = pdf_path, skip_intro = False, page_offset= page_offset)

    return list_titles

def get_titles_from_text(elements, type_chosen):
    list_titles = []

    for i,el in enumerate(elements):
        if type(el).__name__ == type_chosen:
            title_el = {'text': el.text.strip(),
                        'page': getattr(el.metadata, "page_number", 1),
                        'index': i}
            list_titles.append(title_el)

    return list_titles
    
def validate_indexes(tree_titles, matches, elements):
    missing = []

    # Controllo degli indici in matches
    for item in matches:
        idx = item.get("index")
        if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(elements):
            missing.append(item["title"])

    # Creiamo un set dei titoli presenti in matches
    match_titles = {item["title"] for item in matches}

    # Aggiungiamo a missing i titoli in tree_titles ma non in matches
    for item in tree_titles:
        if item["title"] not in match_titles:
            missing.append(item["title"])

    return missing


def build_failsafe_prompt(missing_titles, page_text):
    return f"""
    Assegna la posizione più plausibile ai titoli che ti presento.

    Regole per il match dei titoli:

    1. Per ciascun "title" nella lista dei titoli, trova il "text" più simile possibile negli elementi da matchare.
    - La similarità deve essere semantica: non serve che le parole siano esatte, ma il significato deve corrispondere.
    - Assicurati che ci sia un solo match per ogni titolo.

    2. Le due proprietà "page" devono corrispondere o essere molto vicine.

    3. Se il "title" è diviso in più parti consecutive nel "text", considera solo la **prima parte** come match principale.

    4. Rispondi SOLO con JSON valido.
    - Non aggiungere commenti, spiegazioni o testo extra.
    - Nessuna virgola finale dopo l’ultimo elemento della lista.

    5. Il JSON deve avere il formato esatto seguente:
    [
    {{
        "title": "<titolo considerato>",
        "text": "<testo del match trovato>",
        "index": <indice dell’elemento trovato nella lista originale>
    }}
    ]

    7. Se non trovi un match neanche con gli elementi, scrivi "text": null e "index": null.

    8. Mantieni l’ordine dei titoli come apparso nella lista originale.


    Titoli da matchare:
    {missing_titles}

    Testo disponibile:
    {page_text}
    """

def assign_title_position(elements, tree, pdf_path, api_key, gemini = True, model = "gemma-3-27b-it", skip_intro = True, page_offset = 0):

    """
    Trasformiamo le tabelle estratte in testo usando un LLM
    """

    if gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)
    
    tree_titles = get_tree_titles(tree = tree, pdf_path = pdf_path, skip_intro = skip_intro, page_offset= page_offset)
    actual_titles = get_titles_from_text(elements=elements, type_chosen = "Title")
    headers = get_titles_from_text(elements=elements, type_chosen = "Header")
    footers = get_titles_from_text(elements=elements, type_chosen = "Footer")

    prompt = f"""
    Assegna la posizione più plausibile ai titoli che ti presento.

    Regole per il match dei titoli:

    1. Per ciascun "title" nella lista dei titoli, trova il "text" più simile possibile negli elementi da matchare.
    - La similarità deve essere semantica: non serve che le parole siano esatte, ma il significato deve corrispondere.
    - Assicurati che ci sia un solo match per ogni titolo.

    2. Le due proprietà "page" devono corrispondere o essere molto vicine.

    3. Se il "title" è diviso in più parti consecutive nel "text", considera solo la **prima parte** come match principale.

    4. Rispondi SOLO con JSON valido.
    - Non aggiungere commenti, spiegazioni o testo extra.
    - Nessuna virgola finale dopo l’ultimo elemento della lista.

    5. Il JSON deve avere il formato esatto seguente:
    [
    {{
        "title": "<titolo considerato>",
        "text": "<testo del match trovato>",
        "index": <indice dell’elemento trovato nella lista originale>
    }}
    ]

    6. Se un titolo non ha un match diretto, controlla anche negli elementi extra forniti (ad esempio headers e footers).

    7. Se non trovi un match neanche con gli elementi extra, scrivi "text": null e "index": null.

    8. Mantieni l’ordine dei titoli come apparso nella lista originale.

    Titoli:
    {tree_titles}

    Elementi da matchare:
    {actual_titles}

    Se manca qualche match, controlla anche in questi elementi:
    {headers}

    {footers}
    """

    response = client.chat.completions.create(
        model= model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content

    if raw:
        cleaned = parse_llm_json(raw)

        missing = validate_indexes(tree_titles = tree_titles, matches = cleaned, elements = elements)

        if missing:

            if len(missing)>5:
                raise ValueError("Troppi titoli non assegnati")

            recovered = []

            for title in missing:
                # trova info del titolo
                title_info = next(t for t in tree_titles if t["title"] == title)
                page = title_info["page"]

                page_text = get_text_by_page(elements, page, delta=1)

                if not page_text:
                    continue

                prompt = build_failsafe_prompt(
                    missing_titles=[title_info],
                    page_text=page_text
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                raw = response.choices[0].message.content
                if raw:
                    # rimuovi eventuale match precedente (incompleto/errato) per questo titolo
                    cleaned = [c for c in cleaned if c.get("title") != title]

                    recovered.extend(parse_llm_json(raw))
            
            final = cleaned + recovered

            still_missing = validate_indexes(tree_titles = tree_titles, matches = final, elements = elements)

            if still_missing:
                raise ValueError(
                    f"Impossibile assegnare {len(still_missing)} titoli:\n"
                    + "\n".join(still_missing)
                )
            
            client.close()

            return final
        else:
            return cleaned

    else:
        client.close()

        raise ValueError("Errore del LLM nell' assegnazione")

def populate_text_between_node_and_first_child(
    tree,
    elements,
    pdf_path,
    summary_match,
    skip_intro=True,
    end_indexes=None,
    end_title=None,
    nested=False,
    title_occurrence=None
):
    """
    Aggiorna ricorsivamente il campo 'text' di ogni nodo,
    prendendo il testo dagli elementi compreso tra il nodo corrente
    e il prossimo titolo (o figlio).

    Gestisce titoli duplicati tramite contatore di occorrenze.
    """

    # Ordinamento di summary_match

    summary_match = sorted(summary_match, key=lambda x: x["index"])

    # --------------------------------------------------
    # strutture condivise
    # --------------------------------------------------
    if title_occurrence is None:
        title_occurrence = defaultdict(int)

    def build_index_map(matches):
        index_map = defaultdict(list)
        for item in matches:
            index_map[normalize_text(item["title"])].append(item["index"])
        return index_map

    def build_title_map(matches):
        title_map = defaultdict(list)
        for item in matches:
            title_map[normalize_text(item["title"])].append(item["text"])
        return title_map

    index_map = build_index_map(summary_match)
    title_map = build_title_map(summary_match)

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------
    def get_index(title):
        key = normalize_text(title)
        occ = title_occurrence[key]

        try:
            idx = index_map[key][occ]
        except IndexError:
            raise KeyError(
                f"Troppe occorrenze del titolo '{title}' "
                f"(richiesta {occ}, disponibili {len(index_map[key])})"
            )

        title_occurrence[key] += 1
        return idx

    def get_title_text(title, occ=0):
        key = normalize_text(title)
        try:
            return title_map[key][occ]
        except IndexError:
            raise KeyError(f"Testo non trovato per il titolo: {title}")

    # --------------------------------------------------
    # Valori default
    # --------------------------------------------------
    if end_indexes is None:
        end_indexes = len(elements)

    first_piece = summary_match[0]["index"]

    # --------------------------------------------------
    # Loop principale
    # --------------------------------------------------
    for i, node in enumerate(tree):

        # Ignora nodo introduttivo
        if skip_intro and node["title"] == f"introduzione_{os.path.basename(pdf_path)}":
            continue

        current_title = node["title"]
        current_index = get_index(current_title)

        if node.get("children"):
            next_title = node["children"][0]["title"]
            last_index = index_map[normalize_text(next_title)][
                title_occurrence[normalize_text(next_title)]
            ]
        else:
            if i + 1 < len(tree):
                next_title = tree[i + 1]["title"]
                last_index = index_map[normalize_text(next_title)][
                    title_occurrence[normalize_text(next_title)]
                ]
            else:
                next_title = end_title
                last_index = end_indexes


        # Raccolta testo

        node_text_lines = []

        for el in elements[current_index:last_index]:
            el_text = el.text.strip()
            if el_text:
                node_text_lines.append(el_text)

        node["text"] = "\n".join(node_text_lines).strip()

        # Rimuove titolo ripetuto
        if node["text"] in {current_title, get_title_text(current_title)}:
            node["text"] = ""


        # --------------------------------------------------
        # Ricorsione figli
        # --------------------------------------------------
        if node.get("children"):
            if i + 1 < len(tree):
                end_title_n = tree[i + 1]["title"]
                end_index_n = index_map[normalize_text(end_title_n)][
                    title_occurrence[normalize_text(end_title_n)]
                ]
            else:
                if nested:
                    end_title_n = end_title
                    end_index_n = end_indexes
                else:
                    end_title_n = None
                    end_index_n = len(elements)

            populate_text_between_node_and_first_child(
                node["children"],
                elements=elements,
                pdf_path=pdf_path,
                summary_match=summary_match,
                skip_intro=False,
                end_indexes=end_index_n,
                end_title=end_title_n,
                nested=True,
                title_occurrence=title_occurrence,
            )

    return tree, first_piece






"""
Functions for extracting tree using a custom summary
"""






def find_first_match_index(clean_elements, marker):
    last_text = ""
    for i, el in enumerate(clean_elements):
        if not hasattr(el, "text"):
            last_text = ""
            continue
        text = el.text.strip()
        if last_text:
            text_2 = last_text + " " + el.text.strip()
        else:
            text_2 = text
        
        if text.startswith(marker):
            return i
        elif text_2.startswith(marker):
            return i-1
        last_text = text
    return None

def build_introduction_node(clean_elements, first_match_idx, pdf_path, title):
    intro_text = []
    page = None

    for el in clean_elements[:first_match_idx]:
        if hasattr(el, "text") and el.text.strip():
            intro_text.append(el.text.strip())
            if page is None:
                page = getattr(el.metadata, "page_number", None)

    return {
        "title": f"introduzione_{os.path.basename(pdf_path)}",
        "page": page,
        "text": "\n".join(intro_text),
        "children": [],
        "document": normalize_text(title, remove_accents= False, remove_new_line= False)
    }

def build_structure(summary_list, title, parent=None):
    nodes = []

    if summary_list == []:
        node = {
            "marker": "",
            "title": "",
            "page": None,
            "children": [],
            "document": normalize_text(title, remove_accents= False, remove_new_line= False)
        }
        nodes.append(node)

    for item in summary_list:
        if isinstance(item, list):
            # attach children to last node
            if nodes:
                nodes[-1]["children"] = build_structure(item, title, nodes[-1])
        else:
            node = {
                "marker": item,
                "title": "",
                "page": None,
                "children": [],
                "document": normalize_text(title, remove_accents= False, remove_new_line= False)
            }
            nodes.append(node)

    return nodes


def flatten_nodes(nodes):
    flat = []
    for n in nodes:
        flat.append(n)
        flat.extend(flatten_nodes(n["children"]))
    return flat

def assign_titles_and_pages(tree, clean_elements, summary_list, verbose = False):

    flat_nodes = flatten_nodes(tree)

    el_index = 0  # ← posizione corrente nel documento
    last_text = ''
    el_int_index = 0

    for i, node in enumerate(flat_nodes):
        marker = node["marker"]
        found = False

        full_text = []
        if last_text:
            full_text.append(last_text)
        

        while el_index < len(clean_elements):
            el = clean_elements[el_index]

            if not hasattr(el, "text"):
                el_index += 1
                continue

            text = el.text.strip()
            if not text:
                el_index += 1
                continue

            text_divided = text.split("\n")

            while el_int_index < len(text_divided):
                text_analyzed = text_divided[el_int_index]
                el_int_index += 1

                if not text_analyzed:
                    continue

                else:
                    full_text.append(text_analyzed)


                # MATCH SOLO IN AVANTI
                if text_analyzed.startswith(marker):
                    found = True
                    if verbose:
                        print(text_analyzed)
                    node["title"] = node['marker']
                    node["page"] = getattr(el.metadata, "page_number", None)

                    last_text = text_analyzed

                    if i>0:
                        flat_nodes[i-1]['text'] = "\n".join(full_text[:-1])
                    break

                if len(full_text)>= 2:
                    text_analyzed = " ".join(full_text[-2:])
                    if text_analyzed.startswith(marker):
                        found = True
                        if verbose:
                            print(text_analyzed)
                        node["title"] = node['marker']
                        el_before = clean_elements[el_index-1] if el_int_index == 1 else el
                        node["page"] = getattr(el_before.metadata, "page_number", None)

                        last_text = text_analyzed

                        if i>0:
                            flat_nodes[i-1]['text'] = "\n".join(full_text[:-2])
                        break
            
            if el_int_index >= len(text_divided):
                el_int_index = 0
                el_index += 1

            if found:
                break



    full_text = [last_text]

    while el_index < len(clean_elements):
        el = clean_elements[el_index]

        if not hasattr(el, "text"):
            el_index += 1
            continue

        text = el.text.strip()

        if not text:
            el_index += 1
            continue

        text_divided = text.split("\n\n")

        while el_int_index < len(text_divided):
            text_analyzed = text_divided[el_int_index]
            el_int_index += 1

            if not text_analyzed:
                continue

            else:
                full_text.append(text_analyzed)
        
        if el_int_index >= len(text_divided):
            el_int_index = 0
            el_index += 1

    flat_nodes[-1]['text'] = "\n".join(full_text)

    if summary_list == []:
        tree[0]["title"] = ""


def clean_output(nodes):
    output = []

    for n in nodes:

        if n.get('marker'):

            if normalize_text(n['marker']) == normalize_text(n['text']):
                n['text'] = ""


        out = {
            "title": n["title"],
            "page": n["page"],
            "text": n["text"],
            "children": clean_output(n["children"]),
            "document": n["document"]
        }
        output.append(out)

    return output





"""
function to create tree from hierarchy
"""

def find_titles_with_hierarchy(hierarchy_titles, elements):

    max_len = max([len(el.text) for el in elements])

    # filtra solo i livelli attivi
    patterns = {}
    active_hierarchy = []
    for t in hierarchy_titles:
        key = t.lower().strip()
        if key in synonyms:
            internal_key = synonyms[key]
            patterns[internal_key] = all_patterns[internal_key]
            if internal_key not in active_hierarchy:
                active_hierarchy.append(internal_key)

    current = {level: None for level in active_hierarchy}
    indice = []
    last_comma = None
    MAX_GAP = 40

    def parse_comma(token: str):
        token = token.strip(".")
        if "-" not in token:
            return int(token), 0
        num, suff = token.split("-", 1)
        return int(num), SUFFIX_RANK.get(suff.lower(), -1)

    def append_path(key, pos):
        idx = active_hierarchy.index(key)
        for lower_level in active_hierarchy[idx + 1:]:
            current[lower_level] = None
        path = [v for k, v in current.items() if v]
        indice.append((path, pos))

    for i, el in enumerate(elements):
        if not hasattr(el, "text") or not el.text.strip():
            continue

        line = el.text.strip()

        for key, pattern in patterns.items():

            if key != "comma":
                match = pattern.match(line)
                if match:
                    if key in {"articolo", "sezione", "capo"}:
                        last_comma = None
                    current[key] = line
                    append_path(key, i)
                continue

            # gestione comma
            if current.get("articolo") is None:
                continue

            for m in pattern.finditer(line):
                num, suff_rank = parse_comma(m.group(1))

                if last_comma is None:
                    last_comma = (num, suff_rank)
                    current[key] = m.group(1)
                    append_path(key, i + m.start() / max_len)
                    continue

                last_num, last_suff = last_comma
                same_number = num == last_num and suff_rank > last_suff
                next_number = num == last_num + 1 and suff_rank == 0

                if same_number or next_number:
                    last_comma = (num, suff_rank)
                    current[key] = m.group(1)
                    append_path(key, i + m.start() / max_len)
                elif m.start() == 0 and last_num <= num <= last_num + MAX_GAP:
                    last_comma = (num, suff_rank)
                    current[key] = m.group(1)
                    append_path(key, i + m.start() / max_len)
                else:
                    continue

    return indice



def find_child(children, title):
    for child in children:
        if child["title"] == title:
            return child
    return None

def create_tree_from_hierarchy(indice):

    tree = []

    for path,idx in indice:
        tree_considered = tree
        for parts in path:
            parts = parts.strip()
            node = find_child(tree_considered,parts)
            if node is None:
                node = {
                        "title": parts,
                        "start": idx,
                        "children": []
                    }
                tree_considered.append(node)
            tree_considered = node['children']
    return tree


def assign_text_and_pages_from_idx(tree, clean_elements, title):

    max_len = max([len(el.text) for el in clean_elements])

    flat_nodes = sorted(flatten_nodes(tree), key=lambda x: x["start"])

    for i, node in enumerate(flat_nodes):
        start_idx = int(node["start"])
        start_frac = node["start"] % 1  # posizione frazionaria nella riga

        if i < len(flat_nodes) - 1:
            end_idx = int(flat_nodes[i + 1]["start"])
            end_frac = flat_nodes[i + 1]["start"] % 1
            if end_frac > 0:
                end_idx += 1
        else:
            end_idx = len(clean_elements)
            end_frac = 0

        node["page"] = getattr(clean_elements[start_idx].metadata, "page_number", None)
        node["document"] = normalize_text(title, remove_accents=False, remove_new_line=False)

        texts = []

        for idx in range(start_idx, end_idx):
            if not hasattr(clean_elements[idx], "text"):
                continue
            t = clean_elements[idx].text.strip()
            if not t:
                continue

            # gestisci frazione se siamo al primo o ultimo elemento
            if idx == start_idx and start_frac > 0:
                cut_point = int(start_frac * max_len)
                t = t[cut_point:]  # taglia l’inizio
            if idx == end_idx - 1 and end_frac > 0:
                cut_point = int(end_frac * max_len)
                t = t[:cut_point]  # taglia la fine

            if t:
                texts.append(t)

        if texts:
            node["text"] = "\n".join(texts)


def build_extended_title(node, elements, start, end, pattern_manual):
    lines = []

    max_len = max([len(el.text) for el in elements])

    start_idx = int(start)
    start_frac = start % 1  # posizione frazionaria nella riga

    end_idx = int(end)
    end_frac = end % 1
    if end_frac > 0:
        end_idx += 1

    for idx in range(start_idx, end_idx):
        el = elements[idx]
        if not hasattr(el, "text"):
            continue
        text = el.text.strip()
        if text:
            if idx == start_idx and start_frac > 0:
                cut_point = int(start_frac * max_len)
                text = text[cut_point:]  # taglia l’inizio
            if idx == end_idx - 1 and end_frac > 0:
                cut_point = int(end_frac * max_len)
                text = text[:cut_point]  # taglia la fine
            lines.append(text)

    if not lines:
        return node["title"]
    
    pattern = re.compile(
            r'\(*(\d+(?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|duodevicies|undevicies|vicies|vices semel|vices bis|vices ter))?)\.\s*',
            re.IGNORECASE
        )
    pattern_2 = re.compile(r'^AGGIORNAMENTO\s*\(\s*\d+\s*\)$')

    m = pattern.match(node["title"])
    m_2 = pattern_2.match(node['title'])

    # 🔹 CASO ARTICOLO
    if node["title"].startswith("Art"):
        title_lines = [lines[0]]  # Art. n

        # guardo al massimo le 4 righe successive
        for next_line in lines[1:5]:
            title_lines.append(next_line)
            if pattern_manual.search(next_line):
                break
        else:
            # se NON ho trovato il pattern → solo una riga dopo Art.
            title_lines = lines[:2]

        return "\n".join(title_lines)

    # CASO COMMA o AGGIORNAMENTO
    elif m:
        return f"Comma {m.group(1)}"
    
    elif m_2:
        return node['title']
        
    
    # 🔹 CASO NON ARTICOLO
    else:
        return "\n".join(lines)
    

def extend_titles(tree, clean_elements, pattern):

    flat_nodes = sorted(flatten_nodes(tree), key=lambda x: x["start"])

    for i, node in enumerate(flat_nodes):

        start = node["start"]

        if i < len(flat_nodes) - 1:
            end = flat_nodes[i + 1]["start"]
        else:
            end = len(clean_elements)

        node["title"] = build_extended_title(
            node, clean_elements, start, end, pattern
        )

def remove_text_equal_to_title(tree, pdf_path, skip_intro = True):
    flat_nodes = sorted(flatten_nodes(tree), key=lambda x: x["start"])

    for node in flat_nodes:

        if skip_intro and node["title"] == f"introduzione_{os.path.basename(pdf_path)}":
            continue

        else:
            if normalize_text(node['title']) == normalize_text(node['text']):
                node['text'] = ""




"""
Function to further clean the text
"""

def clean_text_tree(tree, pattern_function):
    flat_nodes = flatten_nodes(tree)
    for node in flat_nodes:
        node['text'] = pattern_function(node['text'])






"""
Functions to verify if the full text was extracted
"""





def extract_text_from_tree(tree, pdf_path, summary_match=None, use_auto_summary=True):

    if use_auto_summary and summary_match is None:
        raise ValueError("summary_match è obbligatorio quando use_auto_summary=True")

    title_map = None
    if use_auto_summary:
        assert summary_match is not None
        title_map = {
            normalize_text(item["title"]): item["text"]
            for item in summary_match
        }

    def get_title(title):
        norm = normalize_text(title)
        if use_auto_summary:
            assert title_map is not None
            try:
                return title_map[norm]
            except KeyError:
                raise KeyError(f"Titolo non trovato nel matching: {title}")
        return norm

    lines = []
    for node in tree:
        if node.get('title'):
            if node['title'] != f'introduzione_{os.path.basename(pdf_path)}' and not node.get('text'):
                lines.append(get_title(node['title']))

        if node.get('text'):
            lines.append(node['text'])
        
        if node.get('children'):
            lines.append(extract_text_from_tree(node['children'], pdf_path, summary_match= summary_match, use_auto_summary= use_auto_summary))
        
    return "\n".join(lines)


def remove_points(text: str) -> str:
    if not text:
        return ""

    # Normalizza unicode (accenti coerenti)
    text = unicodedata.normalize("NFKC", text)

    # Rimuove tutta la punteggiatura
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip().lower()

def verify_pages_against_tree_fuzzy(elements, tree_with_text, pdf_path, start_idx, summary_match = None, min_ratio=1, use_auto_summary = True):
    """
    Segnala una pagina come mancante se meno del min_ratio del testo
    è presente nel tree.
    """
    if use_auto_summary and summary_match is None:
        raise ValueError("summary_match è obbligatorio quando use_auto_summary=True")

    missing_pages = []

    full_text = remove_points(normalize_text(extract_text_from_tree(tree_with_text, pdf_path= pdf_path, summary_match= summary_match, use_auto_summary= use_auto_summary)))

    pages_dict = {}
    for el in elements[start_idx:]:
        page_num = getattr(el.metadata, "page_number", 1)
        pages_dict.setdefault(page_num, []).append(el.text.strip())

    for page_num, texts in pages_dict.items():
        found = 0
        total = 0

        for t in texts:
            norm_t = remove_points(normalize_text(t))
            if not norm_t:
                continue
            total += 1
            if norm_t in full_text:
                found += 1

        if total > 0 and found / total < min_ratio:
            missing_pages.append({
                "page": page_num,
                "coverage": round(found / total, 2),
                "missing_text": [
                    t for t in texts
                    if normalize_text(t) not in full_text
                ]
            })

    return missing_pages





"""
Unique function for creation of tree
"""





def build_tree_from_summary(pdf_path, elements, api_key, gemini = True, model = "gemma-3-27b-it", model_summary = "gemini-2.5-flash-lite",
                            max_pages=5, skip_intro = True, min_ratio = 1, title = None, hierarchy_list = None,
                            summary_list = None, use_auto_summary = True, page_start = None, title_in_pdf = True, page_offset = 0,
                            pattern_function = None):
    
    if title is None:
        try:
            title_document_json = get_title_document(elements= elements, api_key= api_key, gemini= gemini, model= model, delta = 0, page_start = page_start)
            title = title_document_json.get("title","")
        except Exception as e:
            raise RuntimeError("Error on extraction of title:") from e
    
    if use_auto_summary:
        try:
            full_text = extract_first_pages(pdf_path, max_pages= max_pages, page_start= page_start)
            summary = get_summary(full_text, api_key = api_key, gemini= gemini, model = model_summary)
        except Exception as e:
            raise RuntimeError("Error on extraction of summary:") from e

        try:
            tree = create_tree_summary(pdf_path, elements= elements, title= title, summary = summary)
        except Exception as e:
            raise RuntimeError("Error on creation of tree_summary from indent:") from e
        
        if len(tree) <= 1:
            raise ValueError("Error: couldn't find summary with current pattern")
        
        try:
            summary_match = assign_title_position(elements=elements, tree = tree, pdf_path= pdf_path, api_key = api_key, gemini = gemini,
                                                  model = model, skip_intro= skip_intro, page_offset= page_offset)
        except Exception as e:
            raise RuntimeError("Error on assigning title positions:") from e
        
        try:
            tree_with_text, start_idx = populate_text_between_node_and_first_child(tree, elements = elements, 
                                                                                skip_intro = skip_intro, pdf_path = pdf_path, summary_match= summary_match,
                                                                                    end_indexes=None, end_title=None, nested = False)
        except Exception as e:
            raise RuntimeError("Error on inserting text in tree:") from e
    else:

        summary_match = None

        if summary_list is not None:


            # 1. costruisci struttura
            tree = build_structure(summary_list = summary_list, title = title)

            # 2. flatten markers
            markers = [n["marker"] for n in flatten_nodes(nodes = tree)]

            # 3. trova primo match
            if summary_list == []:
                if title_in_pdf:
                    start_idx = find_first_match_index(clean_elements = elements, marker = title)
                    assert isinstance(start_idx, int)
                    start_idx += 1
                else:
                    start_idx = 0
            else:
                start_idx = find_first_match_index(clean_elements = elements, marker = markers[0])

            # 4. introduzione
            intro_node = build_introduction_node(clean_elements = elements, first_match_idx = start_idx, pdf_path = pdf_path, title = title)

            # 5. parsing normale MA partendo da first_match_idx
            assign_titles_and_pages(
                tree = tree,
                clean_elements= elements[start_idx:],
                summary_list= summary_list
            )
        
        elif hierarchy_list is not None:
            
            try:
                # Assegnazione indice per ogni titolo con gerarchia
                indice = find_titles_with_hierarchy(hierarchy_titles=hierarchy_list, elements= elements)
            except Exception as e:
                raise RuntimeError("Error on the creation of the hierarchy:") from e

            # Creazione albero partendo dall'indice
            tree = create_tree_from_hierarchy(indice)

            start_idx = indice[0][1]

            # introduzione
            intro_node = build_introduction_node(clean_elements= elements, first_match_idx= start_idx, pdf_path= pdf_path, title= title)

            # assegnazione del testo corrispondente ad ogni paragrafo
            assign_text_and_pages_from_idx(tree, clean_elements= elements, title= title)

            # Rimozione del testo se testo = titolo (chunk superfluo)
            remove_text_equal_to_title(tree, pdf_path)
        
        else:
            raise ValueError("nessuna lista di titoli dei paragrafi data")




        # output finale
        tree_with_text = [intro_node] + clean_output(nodes = tree)
    
    missing = verify_pages_against_tree_fuzzy(elements = elements, tree_with_text= tree_with_text,
                                              pdf_path= pdf_path, summary_match = summary_match, 
                                              start_idx= start_idx, min_ratio = min_ratio, use_auto_summary= use_auto_summary)
    
    if missing:
        raise RuntimeError("Missing text when creating tree (probably tree not built correctly)")
    
    if pattern_function is not None:
        clean_text_tree(tree_with_text, pattern_function)
    
    
    return tree_with_text
    
def build_base_chunks(tree, pdf_path, keep_intro = True):
    chunks = []

    def visit(node, parents):
        """
        node: nodo corrente
        parents: lista dei titoli dei genitori (dal più vicino al più lontano)
        """
        current_title = node.get("title")
        new_parents = [current_title] + parents if (current_title and current_title != f'introduzione_{os.path.basename(pdf_path)}') else parents

        if (not keep_intro) and current_title == f'introduzione_{os.path.basename(pdf_path)}':
            return

        text = node.get("text", "").strip()

        path = Path(pdf_path)

        # Se il nodo ha testo → crea chunk
        if text:
            # Contenuto del chunk
            chunk_text = text

            # Metadata
            metadata = {
                "document": node.get("document"),
                "title": current_title,
                "page": node.get("page")
            }

            # title_1, title_2, ...
            for i, parent_title in enumerate(parents, start=1):
                metadata[f"title_{i}"] = parent_title

            chunks.append({
                "text": normalize_text(chunk_text, remove_accents= False, remove_new_line= False),
                "metadata": metadata
            })

        # Continua nei children
        for child in node.get("children", []):
            visit(child, new_parents)

    for root in tree:
        visit(root, [])

    return chunks

def chunk_with_overlap_tiktoken(
    text: str,
    max_tokens: int = 500,
    overlap: int = 60,
    min_tail_tokens: int = 100,
    encoding_name: str = "cl100k_base" # Might add model name and do encoding = tiktoken.encoding_for_model('gpt-4o-mini')
):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    n = len(tokens)
    if n <= max_tokens:
        return [text]

    chunks = []
    start = 0

    while start < n:
        end = start + max_tokens

        # Se il residuo sarebbe troppo piccolo → accorpa
        remaining = n - end
        if 0 < remaining < min_tail_tokens:
            end = n

        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))

        if end >= n:
            break

        start = end - overlap

    return chunks

def build_chunks(tree, pdf_path,
                max_tokens: int = 500,
                overlap: int = 60,
                min_tail_tokens: int = 100,
                encoding_name: str = "cl100k_base",
                keep_intro = True
):
    
    try:
        final_chunks = []

        logical_chunks = build_base_chunks(tree, pdf_path, keep_intro= keep_intro)

        for chunk in logical_chunks:
            sub_chunks = chunk_with_overlap_tiktoken(
                text=chunk["text"],
                max_tokens=max_tokens,
                overlap=overlap,
                min_tail_tokens= min_tail_tokens,
                encoding_name= encoding_name
            )

            for i, sub_text in enumerate(sub_chunks):
                final_chunks.append({
                    "text": sub_text,
                    "metadata": {
                        **chunk["metadata"],
                        "sub_chunk": i
                    }
                })

    except Exception as e:
        raise RuntimeError("Error on creating chunks:") from e
    
    return final_chunks

def add_metadata(chunks, categoria: str = "", valenza: str = "",  tipologia: str = "", numero: str = "", eccezione: str = "",
                 data: str = "", periodo: tuple[str,str] = ("1/01/1900","31/12/2099"), anno: Optional[Union[int, list[int]]] = None):
    # if tipologia is None:
    #     tipologia= ""
    # if numero is None:
    #     numero= ""
    # if data is None:
    #     data = ""
    # if periodo is None:
    #     periodo = ("1/01/1900","31/12/2099")

    for chunk in chunks:
        if categoria:
            chunk['metadata']['categoria'] = categoria
        if valenza:
            chunk['metadata']['valenza'] = valenza
        if eccezione:
            chunk['metadata']['eccezione'] = eccezione
        if tipologia:
            chunk['metadata']['tipologia'] = tipologia
        if numero:
            chunk['metadata']['numero'] = numero
        if data:
            chunk['metadata']['data'] = data
        chunk['metadata']['periodo'] = periodo
        if anno is not None:
            if isinstance(anno, int):
                chunk['metadata']['anno'] = [anno]
            else:
                chunk['metadata']['anno'] = anno
    
    return chunks


def pagine_per_sommario_adattato(num_pag):
    """
    Numero di pagine da prendere per catturare il sommario.
    
    - Fino a 39 pagine: formula originale (ceil(0.1*num_pag)+1, max 10)
    - Dopo 39 pagine: aumento molto più lento, massimo sempre 10
    """
    if num_pag <= 39:
        pagine = math.ceil(0.1 * num_pag) + 1
    else:
        # crescita molto lenta oltre 39, ad esempio +1 ogni 20 pagine
        pagine = 5 + math.floor((num_pag - 39) / 20)
    
    return min(pagine, 10)



""""
pdf_path:                                           Path del file pdf da estrarre.
api_key:                                            api_key in formato stringa.
gemini:             Default: True                   True: se si usa un API di Gemini, False: per OpenAI.
model:              Default: "gemma-3-27b-it"       nome del modello principale da usare.               
model_summary:      Default: "gemini-2.5-flash"     nome del modello usato esclusivamente per il riconoscimento di un sommario.
                                                    (più complesso rispetto alle altre task).
table_overlap:      Default: 0.7                    Indica quale parte del testo estratto con uncostructed deve essere sostituito.
                    Valori da 0 a 1                 con la rispettiva tabella estratta con Camelot (più precisa) usando la posizione all'interno della pagina.
                                                    Indica la percentuale della sua posizione all'interno di una colonna.
                                                    (Nel caso = 1 viene tolto solamente il testo completamente all'interno della tabella,
                                                    nel caso = 0 viene tolto qualsiasi elemento che abbia qualsiasi punto all'interno della tabella).
page_start:         Default: None                   Se specificato, estrae solamente a partire da quella pagina compresa (pagine ordinate partendo dall'1).
                                                    In caso contrario parte da 1.
page_end:           Default: None                   Se specificato, estrae solamente fino a quella pagina compresa (pagine ordinate a partire dall'1 (1,2,3,4,...)).
                                                    In caso contrario arriva fino all'ultima pagina.
max_pages:          Default: None                   Se specificato, la ricerca del sommario avviene nelle prime "max_pages" pagine.
                                                    In caso contrario calcola un numero adatto in base al numero di pagine complessivo.
number_at_end:      Default: True                   Se attivo, rimuove gli elementi della pagina in posizione più bassa.
                                                    (Per eliminare eventuali numeri di pagina)
num_tolerance:      Default: 0.001                  Indica il range per la scelta di cosa rimuovere quando number_at_end è attivo.
                                                    (Prende fino a posizione minima + num_tolerance. 
                                                    Sistema di riferimento: RelativeCoordinateSystem)
remove_header:      Default: False                  Se attivo, rimuove gli elementi in cima alla pagina
header_tolerance:   Default: 0.05                   Indica il range per la scelta di cosa rimuovere quando remove_header è attivo.
                                                    (Prende fino a posizione massima - header_tolerance. 
                                                    Sistema di riferimento: RelativeCoordinateSystem)

keep_header_intro:  Default: True                   Se attivo e remove_header è settato a True, tiene comunque l'header della prima pagina.
ask_for_duplicates: Default: True                   Se attivo chiede prima di eliminare sospette righe duplicate durante l'estrazione.
use_image:          Default: False                  Se attivo, prende anche il testo estratto da immagini presenti nel pdf.
title:              Default: None                   Stringa che rappresenta il titolo del documento. Verrà salvato come metadata.
                                                    In caso non si fornisca manualmente il titolo, verrà estratto usando il model dalla prima pagina.
skip_intro:         Default: True                   Finchè attivo evita di lavorare sull'introduzione costruita (Non cambiare a meno di debug)
min_ratio:          Default: 1                      Proporzione del testo correttamente immagazzinato nell'albero costruito.
                                                    Serve a controllare la validità dell'albero.
max_tokens:         Default: 500                    Numero massimo di token per chunk
overlap:            Default: 60                     Numero di token su cui fare l'overlap per non perdere la semanticità del testo
min_tail_tokens:    Default: 100                    Numero di token minimi per procedere con la separazione in due chunk diversi.
                                                    Esempio: Se un articolo contenesse 530 chunk, dovrebbe essere diviso in un chunk da 500
                                                             e in un chunk da 90 chunk (60 da overlap + 30 rimanenti). Poichè meno di 100 si crea invece
                                                             un chunk unico da 530 token.
                                                    Numero massimmo di token per chunk è dunque: max_tokens + min_tail_tokens - overlap
encoding_name:      Default: "cl100k_base"          Nome dell’encoding usato per il conteggio e la suddivisione dei token tramite tiktoken.
                                                    "cl100k_base": encoding standard per GPT-4 / GPT-4-Turbo / GPT-4o-mini
use_auto_summary    Default: True                   Se attivo, uso il modello specificato con model_summary per cercare il sommario nelle prime max_pages
                                                    e succcessivamente, lo utilizza per la creazione dell'albero.
                                                    Se disattivato è necassario fornire un valore per summary_list o hierarchy_list.
summary_list:       Default: None                   Utilizzato solamente se use_auto_summary = False.
                                                    Usa la lista fornita per creare il summary.
                                                    La lista deve contenere il nome dei paragrafi (o almeno la prima parte se abbastanza specifica),
                                                    la gerarchia deve essere indicata con sottoliste (la lista dei sottoparagrafi deve seguire immediatamente
                                                    il nome del paragrafo genitore)

                                                    Esempio: ["Art-1_name",["Comma_1_name","Comma_2_name",...],"Art-2_name","Art-3_name"]

                                                    Nel caso si usi summary_list = [], verrà preso tutto il testo come un unico paragrafo.
hierarchy_list:     Default: None                   Utilizzato solamente se use_auto_summary = False e non si fornisce summary_list.
                                                    Lista nella forma ["Parte", "titolo", "CAPO", "sezione", "art."], dove ogni stringa
                                                    indica che esiste un paragrafo con quel nome e possibilmente dei sottopragrafi con nome il nome delle stringhe successive.
                                                    Al momento sono accettati solamente:    "parte"  --->  'Parte' + numero romano
                                                                                            "titolo" --->  'Titolo' + numero romano
                                                                                            "capo"   --->  'Capo' + numero romano
                                                                                            "sezione"--->  'Sezione' + numero romano
                                                                                "Art." o "Articolo"  --->  'Art.' + numero arabo + eventuali "-bis"
                                                                                                            o altre scritte
                                                                                            "comma"  --->   numero arabo seguito da punto e eventuale testo
                                                                                                            (es: 201. Per i processi dichiarati estinti ai sensi ...)
                                                                                    "Aggiornamento"  --->   "AGGIORNAMENTO + (numero arabo)
                                                    In tutti i casi i caratteri maiuscoli o minuscoli sono interscambiabili.
keep_intro:         Default: True                   Se disattivato non verrà creato il chunk per l'introduzione (tutto ciò che viene 
                                                    prima del primo paragrafo)
title_in_pdf:       Default: True                   Se attivo può cercare il title per creare il nodo "introduzione" (solo nel caso summary_list = [])
pattern_function:   Default: None                   Se specificato usa la funzione per effettuare un'ulteriore pulizia del testo attraverso la funzione.
                                                    (La pulizia avviene durante la creazione dell'albero)
categoria:          Default: ""                     Categoria del documento (Regolamento, Normativa, Aliquote, ...).
eccezione:          Default: ""                     Indica qual'è l'eccezione specifica di cui parla il documento.
valenza:            Default: ""                     Valenza del documento (Comunale, Regionale, Nazionale,...)
tipologia:          Default: ""                     Tipologia da aggiungere al metadata (es. Decreto Legge, Legge, Decreto Legislativo, Regio Decreto ...).
numero:             Default: ""                     Numero della Legge da aggiungere al metadata.
data:               Default: ""                     Data in cui la legge è stata emanata da aggiungere al metadata.
                                                    (Formato: "d/m/YYYY")
periodo:            Default:                        Periodo in cui il documento è "attivo".
                    ("1/01/1900","31/12/2099")
use_saved_chunks:   Default: True                   Se attivo, se esiste già un file con i chunk creati in precedenza, lo carica invece di rifare tutta la procedura.
"""



def build_chunks_embedding(pdf_path: str, api_key:str,
                            gemini: bool = True,
                            model: str = "gemma-3-27b-it",
                            model_summary: str = "gemini-2.5-flash",
                            table_overlap: float = 0.7,
                            page_start: Optional[int] = None,
                            page_end: Optional[int] = None,
                            max_pages: Optional[int] = None,
                            number_at_end: bool = True,
                            num_tolerance: float = 0.001,
                            remove_header: bool = False,
                            header_tolerance: float = 0.05,
                            keep_header_intro: bool = True,
                            ask_for_duplicates: bool = True,
                            use_image: bool = False,
                            title: Optional[str] = None,
                            skip_intro: bool = True,
                            min_ratio: int = 1,
                            max_tokens: int = 500,
                            overlap: int = 60,
                            min_tail_tokens: int = 100,
                            encoding_name: str = "cl100k_base",
                            hierarchy_list: Optional[list] = None,
                            summary_list: Optional[list] = None,
                            use_auto_summary: bool = True,
                            keep_intro: bool = True,
                            title_in_pdf: bool = True,
                            page_offset: int = 0,
                            pattern_function: Optional[Callable] = None,
                            categoria: str = "",
                            valenza: str = "",
                            tipologia: str = "",
                            eccezione: str = "",
                            numero: str = "", 
                            data: str = "", 
                            periodo: tuple[str,str] = ("1/01/1900","31/12/2099"),
                            anno: Optional[Union[int, list[int]]] = None,
                            use_saved_chunks: bool = True):
    
    
    pdf_path_chunk = path_for_extraction(pdf_path, category = "chunk")

    if os.path.exists(pdf_path_chunk) and use_saved_chunks:
        with open(pdf_path_chunk, "rb") as f:
            final_chunks = pickle.load(f)
    
    else:
        pdf_path_clean = path_for_extraction(pdf_path, category = "clean")

        if os.path.exists(pdf_path_clean):
            with open(pdf_path_clean, "rb") as f:
                clean_elements = pickle.load(f)
        
        else:
            print("Extracting text from pdf...")
            try:
                clean_elements = extraction_tot(pdf_path, api_key, gemini= gemini, model= model, table_overlap= table_overlap,
                                                page_start= page_start, page_end = page_end, number_at_end= number_at_end,
                                                num_tolerance= num_tolerance, remove_header = remove_header, header_tolerance = header_tolerance,
                                                keep_header_intro = keep_header_intro, use_image = use_image, ask_for_duplicates=ask_for_duplicates)
            except Exception as e:
                raise RuntimeError("Error during extraction:") from e
            with open(pdf_path_clean, "wb") as f:
                pickle.dump(clean_elements, f)
            
            print("Done")
            print()
        
        pdf_path_tree = path_for_extraction(pdf_path, category = "tree")

        if os.path.exists(pdf_path_tree):
            with open(pdf_path_tree, "rb") as f:
                tree_with_text = pickle.load(f)
        
        else:
            print("Extracting tree from summary...")

            try:
                if max_pages == None:
                    end_page = max(getattr(el.metadata, "page_number", 1) for el in clean_elements)
                    max_pages = pagine_per_sommario_adattato(end_page)

                tree_with_text = build_tree_from_summary(pdf_path, elements = clean_elements, api_key= api_key,
                                                        gemini= gemini, model = model,
                                                        model_summary= model_summary, max_pages= max_pages,
                                                        skip_intro = skip_intro, min_ratio = min_ratio, title = title,
                                                        hierarchy_list= hierarchy_list, summary_list= summary_list,
                                                        use_auto_summary= use_auto_summary, page_start = page_start, title_in_pdf = title_in_pdf,
                                                        page_offset= page_offset, pattern_function= pattern_function)
            except Exception as e:
                raise RuntimeError("Error during creation of tree from summary:") from e
            
            with open(pdf_path_tree, "wb") as f:
                pickle.dump(tree_with_text, f)

            print("Done")
            print()
            

        print("Creating chunks...")
        try:
            final_chunks = build_chunks(tree_with_text, pdf_path, max_tokens = max_tokens, overlap = overlap, 
                                        min_tail_tokens = min_tail_tokens, encoding_name= encoding_name, keep_intro= keep_intro)
            final_chunks = add_metadata(chunks= final_chunks, categoria = categoria, valenza = valenza, tipologia= tipologia,
                                        numero= numero, data = data, periodo = periodo, anno = anno, eccezione = eccezione)
        except Exception as e:
            raise RuntimeError("Error during creation of chunks:") from e
        with open(pdf_path_chunk, "wb") as f:
            pickle.dump(final_chunks, f)
        
        print("Done")
        
    return final_chunks

def get_clean_extraction(pdf_path, api_key, gemini: bool = True, model: str = "gemma-3-27b-it", table_overlap: float = 0.7,
                         page_start: Optional[int] = None, page_end: Optional[int] = None, ask_for_duplicates: bool = True,
                         number_at_end: bool = True, num_tolerance: float = 0.001, remove_header: bool = False,
                         header_tolerance: float = 0.05, keep_header_intro: bool = True, use_image: bool = False):
    pdf_path_clean = path_for_extraction(pdf_path, category = "clean")

    if os.path.exists(pdf_path_clean):
        with open(pdf_path_clean, "rb") as f:
            clean_elements = pickle.load(f)
    
    else:
        try:
            clean_elements = extraction_tot(pdf_path, api_key, gemini = gemini, model = model, table_overlap= table_overlap,
                                            page_start= page_start, page_end = page_end, number_at_end= number_at_end,
                                            num_tolerance= num_tolerance, remove_header = remove_header, header_tolerance = header_tolerance,
                                            keep_header_intro = keep_header_intro, use_image = use_image, ask_for_duplicates= ask_for_duplicates)
        except Exception as e:
            raise RuntimeError("Error during extraction:") from e
        with open(pdf_path_clean, "wb") as f:
            pickle.dump(clean_elements, f)
    
    return clean_elements

def get_tree(pdf_path, api_key, elements, gemini: bool = True, model: str = "gemma-3-27b-it",
             model_summary: str = "gemini-2.5-flash", max_pages: int = 5,
             page_start: Optional[int] = None, skip_intro: bool = True, min_ratio: int = 1,
             title: Optional[str] = None, hierarchy_list: Optional[list] = None,
            summary_list: Optional[list] = None, use_auto_summary: bool = True, title_in_pdf: bool = True,
            page_offset: int = 0, pattern_function: Optional[Callable] = None):

    pdf_path_tree = path_for_extraction(pdf_path, category = "tree")

    if os.path.exists(pdf_path_tree):
        with open(pdf_path_tree, "rb") as f:
            tree_with_text = pickle.load(f)
    
    else:

        try:
            if max_pages == None:
                    end_page = max(getattr(el.metadata, "page_number", 1) for el in elements)
                    max_pages = pagine_per_sommario_adattato(end_page)
            tree_with_text = build_tree_from_summary(pdf_path, elements = elements, api_key= api_key,
                                                    gemini= gemini, model = model,
                                                    model_summary= model_summary, max_pages= max_pages,
                                                    skip_intro = skip_intro, min_ratio = min_ratio, title = title, hierarchy_list = hierarchy_list,
                                                    summary_list= summary_list, use_auto_summary= use_auto_summary,
                                                    page_start = page_start, title_in_pdf= title_in_pdf, page_offset= page_offset, pattern_function= pattern_function)
        except Exception as e:
            raise RuntimeError("Error during creation of tree from summary:") from e
        
        with open(pdf_path_tree, "wb") as f:
            pickle.dump(tree_with_text, f)
    
    return tree_with_text











"""
Functions to reconstructs full_chunks
"""

def reconstruct_chunk(sub_chunks, overlap, encoding_name="cl100k_base"):
    """
    sub_chunks: list of dicts with keys:
        - "text"
        - "metadata" -> must include "sub_chunk"
    """
    enc = tiktoken.get_encoding(encoding_name)

    # Ensure correct order
    sub_chunks = sorted(sub_chunks, key=lambda x: x["metadata"]["sub_chunk"])

    reconstructed_tokens = []

    for i, chunk in enumerate(sub_chunks):
        tokens = enc.encode(chunk["text"])

        if i == 0:
            # keep everything from the first chunk
            reconstructed_tokens.extend(tokens)
        else:
            # drop overlapping prefix
            reconstructed_tokens.extend(tokens[overlap:])

    return enc.decode(reconstructed_tokens)

from itertools import groupby

def group_by_metadata(chunks, key):
    chunks = sorted(chunks, key=lambda x: x["metadata"][key])
    return {
        k: list(g)
        for k, g in groupby(chunks, key=lambda x: x["metadata"][key])
    }

def dict_title_to_full_text(final_chunks):
    grouped = group_by_metadata(final_chunks, key="title")

    full_chunks = {
        section: reconstruct_chunk(sub_chunks, overlap=60)
        for section, sub_chunks in grouped.items()
    }

    return full_chunks





#### Funzioni Aggiuntive per normative



def normalize_double_parentheses(text):

    def repl(match):
        content = match.group(1)

        # rimuovi se contiene solo spazi e punti
        if content.strip(" .") == "":
            return ""

        content = content.strip()

        if content.isdigit():
            return f"({content})"
        else:
            return content

    # ((testo))
    text = re.sub(r'\(\((.*?)\)\)', repl, text)

    # ((testo
    text = re.sub(r'\(\(([^)]*)', repl, text)

    # testo))
    text = re.sub(r'([^()]*)\)\)', repl, text)

    return text

def normalize_text_union(text, punctuation = True, spaces = True, e = True):
    text = normalize_double_parentheses(text)
    text = normalize_punctuation_spaces(text, punctuation = punctuation, spaces = spaces, e = e)
    text = text.rstrip()          # toglie spazi finali
    return text



def filter_commi_and_agg(tree, start=1, end=1, agg_list= [], include_suffix = True):
    filtered_tree = []
    pattern = re.compile(
        r'\(*Comma\s*(\d+)((?:-(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|quaterdecies|quinquiesdecies|sexiesdecies|septiesdecies|duodevicies|undevicies|vicies|vices semel|vices bis|vices ter))?)\s*',
        re.IGNORECASE
    )
    for node in tree:
        
        # Controlliamo se il nodo è un "Comma"
        if node['title'].startswith('Comma'):
            try:
                if include_suffix:
                    match = pattern.match(node['title'])
                    if match:
                        comma_number = int(match.group(1))
                    else:
                        raise ValueError("Formato del titolo del comma non riconosciuto")
                else:
                    comma_number = int(node['title'].split()[1])
            except ValueError:
                continue
            if start <= comma_number <= end:
                filtered_tree.append(node)
        elif node['title'].startswith('AGGIORNAMENTO'):
            try:
                match = re.search(r'\((\d+)\)', node['title'])
                agg_number = int(match.group(1))       # pyright: ignore[reportOptionalMemberAccess]
            except ValueError:
                continue
            if agg_number in agg_list:
                filtered_tree.append(node)

        else:
            # Se il nodo non è un comma, filtriamo ricorsivamente i suoi figli
            if node.get("children",[]):
                filtered_children = filter_commi_and_agg(node['children'], start, end, agg_list)
                if filtered_children:
                    # Manteniamo il nodo solo se ha figli validi
                    new_node = node.copy()
                    new_node['children'] = filtered_children
                    filtered_tree.append(new_node)
            else:
                # Nodo senza figli, non è un comma → ignoriamo
                continue
    return filtered_tree