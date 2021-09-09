import streamlit as st
import awesome_streamlit as ast
import pages.home
import pages.processor


st.set_page_config(
    page_title="TEST", page_icon="📖", initial_sidebar_state="expanded", layout="wide"
)

PAGES = {"Home": pages.home, "Arabic Text Preprocessor": pages.processor}


def main():
    """Main function."""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Pages", list(PAGES.keys()))

    page = PAGES[selection]
    ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()
