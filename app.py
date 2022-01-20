import awesome_streamlit as ast
import streamlit as st

from backend.utils import get_current_ram_usage, ga

import backend.aragpt
import backend.home
import backend.processor
import backend.sa
import backend.qa

st.set_page_config(
    page_title="TEST", page_icon="📖", initial_sidebar_state="expanded", layout="wide"
)

ga(st.__file__)

PAGES = {
    "Home": backend.home,
    "Arabic Text Preprocessor": backend.processor,
    "Arabic Language Generation": backend.aragpt,
    "Arabic Sentiment Analysis": backend.sa,
    "Arabic Sarcasm Detection": backend.sarcasm,
    "Arabic Question Answering": backend.qa,
}


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Pages", list(PAGES.keys()))

page = PAGES[selection]
# with st.spinner(f"Loading {selection} ..."):
ast.shared.components.write_page(page)

st.sidebar.header("Info")
st.sidebar.write("Made by [Wissam Antoun](https://twitter.com/wissam_antoun)")
st.sidebar.write(
    "Pre-trained models are available on [HF Hub](https://huggingface.co/aubmindlab)"
)
st.sidebar.write(
    "Models source code available on [GitHub](https://github.com/aub-mind/arabert)"
)
st.sidebar.write(
    "App source code available on [GitHub](https://github.com/WissamAntoun/Arabic-NLP-app)"
)
if st.sidebar.checkbox("Show RAM usage"):
    ram = get_current_ram_usage()
    st.sidebar.write("Ram usage: {:.2f}/{:.2f} GB".format(ram[0], ram[1]))
