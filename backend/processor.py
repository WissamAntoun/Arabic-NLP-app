import streamlit as st
import awesome_streamlit as ast
from .preprocess import (
    ArabertPreprocessor,
    white_spaced_back_quotation_regex,
    white_spaced_double_quotation_regex,
    white_spaced_em_dash,
    white_spaced_single_quotation_regex,
    left_and_right_spaced_chars,
    left_spaced_chars,
    right_spaced_chars,
)
import re

MODELS_to_SELECT = [
    "None",
    "bert-base-arabertv01",
    "bert-base-arabert",
    "bert-base-arabertv02",
    "bert-base-arabertv2",
    "bert-large-arabertv02",
    "bert-large-arabertv2",
    "araelectra-base",
    "araelectra-base-discriminator",
    "araelectra-base-generator",
    "araelectra-base-artydiqa",
    "aragpt2-base",
    "aragpt2-medium",
    "aragpt2-large",
    "aragpt2-mega",
]


def unpreprocess(text: str) -> str:
    """Re-formats the text to a classic format where punctuations, brackets, parenthesis are not seperated by whitespaces.
    The objective is to make the generated text of any model appear natural and not preprocessed.

    Args:
        text (:obj:`str`): input text to be un-preprocessed
        desegment (:obj:`bool`, optional): [whether or not to remove farasa pre-segmentation before]..

    Returns:
        str: The unpreprocessed (and possibly Farasa-desegmented) text.
    """

    text = desegment(text)

    # removes the spaces around quotation marks ex: i " ate " an apple --> i "ate" an apple
    # https://stackoverflow.com/a/53436792/5381220
    text = re.sub(white_spaced_double_quotation_regex, '"' + r"\1" + '"', text)
    text = re.sub(white_spaced_single_quotation_regex, "'" + r"\1" + "'", text)
    text = re.sub(white_spaced_back_quotation_regex, "\`" + r"\1" + "\`", text)
    text = re.sub(white_spaced_back_quotation_regex, "\—" + r"\1" + "\—", text)

    # during generation, sometimes the models don't put a space after the dot, this handles it
    text = text.replace(".", " . ")
    text = " ".join(text.split())

    # handle decimals
    text = re.sub(r"(\d+) \. (\d+)", r"\1.\2", text)
    text = re.sub(r"(\d+) \, (\d+)", r"\1,\2", text)

    text = re.sub(left_and_right_spaced_chars, r"\1", text)
    text = re.sub(left_spaced_chars, r"\1", text)
    text = re.sub(right_spaced_chars, r"\1", text)

    return text


def desegment(text: str) -> str:
    """
    Use this function if sentence tokenization was done using
    `from arabert.preprocess_arabert import preprocess` with Farasa enabled
    AraBERT segmentation using Farasa adds a space after the '+' for prefixes,
    and after before the '+' for suffixes

    Example:
    >>> desegment('ال+ دراس +ات')
    الدراسات
    """
    text = text.replace("+ ", "+")
    text = text.replace(" +", "+")
    text = " ".join([_desegmentword(word) for word in text.split(" ")])
    return text


def _desegmentword(orig_word: str) -> str:
    """
    Word segmentor that takes a Farasa Segmented Word and removes the '+' signs

    Example:
    >>> _desegmentword("ال+يومي+ة")
    اليومية
    """
    word = orig_word.replace("ل+ال+", "لل")
    if "ال+ال" not in orig_word:
        word = word.replace("ل+ال", "لل")
    word = word.replace("+", "")
    word = word.replace("للل", "لل")
    return word


def write():

    st.markdown(
        """
        <h1 style="text-align:left;">Arabic Text Pre-Processor</h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        p, div, input, label {
        text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    input_text = st.text_input(
        "Text to Pre-Process",
        value="ولن نبالغ إذا قلنا: إن 'هاتف' أو 'كمبيوتر المكتب' في زمننا هذا ضروري",
    )

    st.sidebar.title("Model Selector")
    model_selector = st.sidebar.selectbox(
        """Select None to enable further filters""", options=MODELS_to_SELECT, index=3
    )
    if model_selector == "None":
        keep_emojis = st.sidebar.checkbox("Keep emojis", False)
        remove_html_markup = st.sidebar.checkbox("Remove html markup", True)
        strip_tashkeel = st.sidebar.checkbox("Strip tashkeel", True)
        replace_urls_emails_mentions = st.sidebar.checkbox(
            "Replace urls and emails", True
        )
        strip_tatweel = st.sidebar.checkbox("Strip tatweel", True)
        insert_white_spaces = st.sidebar.checkbox("Insert white spaces", True)
        remove_non_digit_repetition = st.sidebar.checkbox(
            "Remove non-digit repetition", True
        )
        replace_slash_with_dash = st.sidebar.checkbox("Replace slash with dash", None)
        map_hindi_numbers_to_arabic = st.sidebar.checkbox(
            "Map hindi numbers to arabic", None
        )
        apply_farasa_segmentation = st.sidebar.checkbox(
            "Apply farasa segmentation", None
        )

    run_preprocessor = st.button("Run Pre-Processor")

    prep_text = None
    if run_preprocessor:
        if model_selector == "None":
            arabert_preprocessor = ArabertPreprocessor(
                model_selector,
                keep_emojis,
                remove_html_markup,
                replace_urls_emails_mentions,
                strip_tashkeel,
                strip_tatweel,
                insert_white_spaces,
                remove_non_digit_repetition,
                replace_slash_with_dash,
                map_hindi_numbers_to_arabic,
                apply_farasa_segmentation,
            )
        else:
            arabert_preprocessor = ArabertPreprocessor(model_name=model_selector)
        prep_text = arabert_preprocessor._preprocess_v3(input_text)
        st.write(prep_text)

    st.write("-----")
    input_text_unprep = st.text_input(
        "Text to Undo the Pre-Processing",
        value=prep_text
        if prep_text
        else "و+ لن نبالغ إذا قل +نا : إن ' هاتف ' أو ' كمبيوتر ال+ مكتب ' في زمن +نا هذا ضروري",
    )
    run_unpreprocessor = st.button("Run Un-Pre-Processor")

    if run_unpreprocessor:
        st.write(unpreprocess(input_text_unprep))
