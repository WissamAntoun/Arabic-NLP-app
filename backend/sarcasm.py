import streamlit as st
from .sa import predictor


def write():
    st.markdown(
        """
        # Arabic Sarcasm Detection

        This is a simple sarcasm detection app that uses the [MARBERT](https://huggingface.co/UBC-NLP/MARBERT) model trained on [ArSarcasm](https://github.com/iabufarha/ArSarcasm)
        """
    )

    input_text = st.text_input(
        "Enter your text here:",
    )
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction, scores = predictor.get_preds_from_sarcasm([input_text])
            st.write(f"Result: {prediction[0]}")
            detailed_score = {
                "Sarcastic": scores[0][0],
                "Not_Sarcastic": scores[0][1],
            }
            st.write("All scores:")
            st.write(detailed_score)
