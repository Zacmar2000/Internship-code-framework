# app.py
import streamlit as st
from funzioni_varie.pdf_wizard import *
import tempfile
from dotenv import load_dotenv

st.set_page_config(layout="wide")

# -------------------------------------------------
# Inizializza lo stato globale del wizard
# -------------------------------------------------
if "main_page" not in st.session_state:
    st.session_state.main_page = "Caricamento PDF"

if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if "wizard" not in st.session_state:
    st.session_state.wizard = None

if "api_key" not in st.session_state:
    load_dotenv() # Load environment variables from .env file
    st.session_state.api_key = os.getenv("OPENAI_API_KEY")

if "gemini" not in st.session_state:
    st.session_state.gemini = False

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 9

if "unlocked_pages" not in st.session_state:
    st.session_state.unlocked_pages = {0}

if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}  # salvataggio dati pesanti per step

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def reset_session():
    keys_to_keep = ["dark_mode","api_key","gemini", "total_pages"]
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

# -------------------------------------------------
# Lista step come funzioni
# -------------------------------------------------
def step_load_elements():

    # File uploader permette di scegliere direttamente un PDF
    uploaded_file = st.file_uploader("Seleziona un file PDF", type="pdf")

    if uploaded_file is not None:
        # Salva temporaneamente il file in una cartella locale per lavorarci
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Scrive il contenuto del file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Crea il wizard
        st.session_state.wizard = PDFExtractionWizard(temp_path, streamlit= True, api_key= st.session_state.api_key, gemini= st.session_state.gemini)


        # Carica elementi usando la cache
        st.session_state.wizard.load_elements()

        st.success(f"PDF '{uploaded_file.name}' caricato correttamente!")
        

def step_header_footer():
    wizard = st.session_state.wizard
    wizard.step_header_footer_streamlit()

def step_image_text():
    wizard = st.session_state.wizard
    wizard.check_image_text_streamlit()

def step_camelot():
    wizard = st.session_state.wizard
    wizard.step_camelot_tables_streamlit()

def step_review_transformed():
    wizard = st.session_state.wizard
    wizard.review_transformed_tables_streamlit()


def step_duplicate_review():
    wizard = st.session_state.wizard
    wizard.duplicate_review_streamlit()

def step_save_clean():
    wizard = st.session_state.wizard
    wizard.save_clean_elements()

def step_create_title():
    wizard = st.session_state.wizard
    wizard.create_title()

def step_create_tree():
    wizard = st.session_state.wizard
    wizard.create_tree()

# -------------------------------------------------
# Definizione sequenza step
# -------------------------------------------------
PAGES = [
    ("Step 1: Caricamento PDF", step_load_elements),
    ("Step 2: Header/Footer", step_header_footer),
    ("Step 3: Controllo testo immagini", step_image_text),
    ("Step 4: Selezione tabelle", step_camelot),
    ("Step 5: Revisione tabelle", step_review_transformed),
    ("Step 6: Revisione duplicati", step_duplicate_review),
    ("Step 7: Salvataggio testo pulito", step_save_clean),
    ("Step 1: Titolo del documento", step_create_title),
    ("Step 2: Scelta per la divisione", step_create_tree)
]

# -------------------------------------------------
# Layout principale
# -------------------------------------------------
st.session_state.dark_mode = st.sidebar.toggle(
    "Dark Mode",
    value=st.session_state.dark_mode
)

st.sidebar.divider()

if st.sidebar.button("New file",disabled= st.session_state.current_page == 0):
    st.session_state.confirm_reset = True

if st.session_state.confirm_reset:
    with st.sidebar.container(border=True):
        st.error("⚠️ Conferma reset")
        st.write("Tutti i dati verranno cancellati.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Conferma"):
                reset_session()
                st.rerun()

        with col2:
            if st.button("Annulla"):
                st.session_state.confirm_reset = False
                st.rerun()




st.sidebar.title("Flusso di Elaborazione")

# Se siamo nello step 1 → mostro solo Caricamento PDF
if st.session_state.current_page == 0:
    st.session_state.main_page = "Caricamento PDF"
elif st.session_state.current_page <= 6:
    st.session_state.main_page = "Pulizia testo"
else:
    st.session_state.main_page = "Divisione paragrafi"

caricamento_done = st.session_state.wizard.get("elements") is not None if st.session_state.wizard is not None else False
pulizia_done = st.session_state.wizard.get("clean_elements") is not None if st.session_state.wizard is not None else False
tree_done = False

def status_label(text, completed):
    current = st.session_state.main_page
    if completed:
        st.sidebar.markdown(
            f"<span style='color:#00C853;'>✔ {text}</span>",
            unsafe_allow_html=True
        )
    elif text == current:
        st.sidebar.markdown(f"{text}")

    else:
        st.sidebar.markdown(
            f"<span style='color:gray;'>{text}</span>",
            unsafe_allow_html=True
        )

status_label("Caricamento PDF", caricamento_done)
status_label("Pulizia testo", pulizia_done)
status_label("Divisione paragrafi", tree_done)

def get_sidebar_steps(nome_paragrafo, nome_sottoparagrafo, start = 0, end = st.session_state.total_pages):

    if st.session_state.main_page == nome_paragrafo:

        st.sidebar.divider()
        st.sidebar.subheader(nome_sottoparagrafo)

        for i, (title, page_func) in enumerate(PAGES[start:end], start=1):

            if (i+start-1) in st.session_state.unlocked_pages:
                if st.sidebar.button(title, key=f"page_{start+i-1}"):
                    st.session_state.current_page = start + i-1
            else:
                st.sidebar.button(title + " 🔒", disabled=True)

    st.sidebar.divider()

if st.session_state.main_page == "Pulizia testo":
    get_sidebar_steps("Pulizia testo","Step Pulizia",1,7)

elif st.session_state.main_page == "Divisione paragrafi":
    get_sidebar_steps("Divisione paragrafi","Step Paragrafi",7)


def apply_global_theme():
    if st.session_state.dark_mode:
        bg = "#0e1117"
        fg = "#e6e6e6"
    else:
        bg = "white"
        fg = "black"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg};
            color: {fg};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

apply_global_theme()


# Esegui step corrente

title, page_func = PAGES[st.session_state.current_page]
st.header(title)
page_func()