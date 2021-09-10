import streamlit as st
import awesome_streamlit as ast
import pages.home
import pages.processor


st.set_page_config(
    page_title="TEST", page_icon="📖", initial_sidebar_state="expanded", layout="wide"
)

PAGES = {"Home": pages.home, "Arabic Text Preprocessor": pages.processor}


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Pages", list(PAGES.keys()))

page = PAGES[selection]
with st.spinner(f"Loading {selection} ..."):
    ast.shared.components.write_page(page)

st.sidebar.header("Info")
st.sidebar.write("Made by [Wissam Antoun](https://twitter.com/wissam_antoun)")
st.sidebar.write("[Models Repo](https://github.com/aub-mind/arabert)")
st.sidebar.write("Source Code [GitHub](https://github.com/WissamAntoun/Arabic-NLP-app)")
