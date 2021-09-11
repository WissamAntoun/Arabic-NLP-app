import streamlit as st
from .services import SentimentAnalyzer
from functools import lru_cache

# @st.cache(allow_output_mutation=False, hash_funcs={Tokenizer: str})
@lru_cache(maxsize=1)
def load_text_generator():
    predictor = SentimentAnalyzer()
    return predictor


predictor = load_text_generator()


def write():
    input_text = st.text_input("Enter your text here:", key="Fuck you")
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction, score, all_score = predictor.predict([input_text])
            st.write(f"Prediction: {prediction}")
            st.write(f"Score: {score}")
            st.write(f"All scores: {all_score}")
