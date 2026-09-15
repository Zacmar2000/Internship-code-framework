import streamlit as st
from pathlib import Path

from RAG_system.config import USE_NEW

if USE_NEW:
    PROMPT_PATH = Path("Prompts/new_prompt.txt")

else:
    PROMPT_PATH = Path("Prompts/main_prompt.txt")

st.set_page_config(page_title="Editor Prompt")

st.markdown("""
<style>
.block-container {
    max-width: 1400px;
    padding-left: 3rem;
    padding-right: 3rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Editor Prompt Chatbot IMU")

st.sidebar.page_link('chatbot_IMU.py', label='Home')
st.sidebar.page_link('pages/prompt_editor.py', label='Prompt editor')

# Carica prompt
if "prompt" not in st.session_state:
    if PROMPT_PATH.exists():
        st.session_state.prompt = PROMPT_PATH.read_text(encoding="utf-8")
    else:
        st.session_state.prompt = ""

# Editor
if "editor_text" not in st.session_state:
    st.session_state.editor_text = st.session_state.prompt

if st.session_state.get("reset_editor", False):
    st.session_state.editor_text = st.session_state.prompt
    st.session_state.reset_editor = False

# Editor
st.text_area(
    "Modifica il prompt",
    key="editor_text",
    height=400
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Usa nuovo prompt"):
        st.session_state.prompt = st.session_state.editor_text
        st.success("Prompt salvato!")

with col2:
    if st.button("Ricarica da file"):
        # NON toccare editor_text qui
        st.session_state.prompt = PROMPT_PATH.read_text(encoding="utf-8")
        st.session_state.reset_editor = True
        st.rerun()

with col3:
    if st.button("Salva il nuovo prompt"):
        PROMPT_PATH.write_text(st.session_state.editor_text, encoding="utf-8")
        st.session_state.prompt = st.session_state.editor_text
        st.success("Prompt salvato!")


st.markdown("---")
st.markdown("### Prompt attualmente in uso")
st.code(st.session_state.prompt)