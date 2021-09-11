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
    st.markdown(
        """
        # Arabic Sentiment Analysis

        This is a simple sentiment analysis app that uses the prediction kernel from Wissam's (me) submission that won the [Arabic Senitment Analysis competition @ KAUST](https://www.kaggle.com/c/arabic-sentiment-analysis-2021-kaust)
        """
    )
    if st.checkbox("More info: "):
        st.markdown(
            """
            ### Submission Description:

            My submission is based on an ensemble of 5 models with varying preprocessing, and classifier design. All model variants are built over MARBERT [1], which is a BERT-based model pre-trained on 1B dialectal Arabic tweets.

            For preprocessing, all models shared the following steps:
            -	Replacing user mentions with “USER” and links with “URL”
            -	Replacing the “#” with “HASH”
            -	Removed the underscore character since it is missing the MARBERT vocabulary.
            -	Removed diacritics and elongations (tatweel)
            -	Spacing out emojis

            For classifier design, all models use a dense layer on top of MARBERT unless otherwise specified. Model training is done by hyperparameter grid-search with 5-fold cross-validation with the following search space:
            -	Learning rate: [2e-5,3e-5,4e-5]
            -	Batch size: 128
            -	Maximum sequence length: 64
            -	Epochs: 3 (we select the best epoch for the final prediction)
            -	Warmup ratio: [0,0.1]
            -	Seed: [1,25,42,123,666]

            Model I is a vanilla variant with only the preprocessing steps mention above applied. Model II enhances the emoji representation by replacing OOV emojis with ones that have similar meaning, for example 💊  😷.
            We noticed the repetitive use of “السلام عليكم” and “ورحمة الله وبركاته” in neutral tweets, especially when users were directing questions to business accounts. This could confuse the classifier, if it encountered these words in a for example a negative tweet, hence in Model III we removed variation of the phrase mentioned before using fuzzy matching algorithms.

            In Model IV, we tried to help the model by appending a sarcasm label to the input. We first trained a separate MARBERT on the ArSarcasm [2] dataset, and then used it to label the training and test sets.

            Model V uses the vanilla preprocessing approach, but instead of a dense layer built on top of MARBERT, we follow the approach detailed by Safaya et.al. [3] which uses a CNN-based classifier instead.

            For the final prediction, we first average the predictions of the 5 models from cross-validation (this is done for each model separately), we then average the results from the 5 model variants. We observed that the distribution of the predicted sentiment classes, doesn’t quite match the true distribution, this is due to the model preferring the neutral class over the positive class. To counter that, we apply what we call Label-Weighted average where during after the final averaging we rescale the score with the following weights 1.57,0.98 and 0.93 for positive, neutral, and negative (note that the weights were determined empirically).

            1- https://aclanthology.org/2021.acl-long.551/

            2- https://github.com/iabufarha/ArSarcasm

            3- https://github.com/alisafaya/OffensEval2020


            """
        )
    input_text = st.text_input(
        "Enter your text here:",
    )
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction, score, all_score = predictor.predict([input_text])
            st.write(f"Result: {prediction[0]}")
            detailed_score = {
                "Positive": all_score[0][0],
                "Neutral": all_score[0][1],
                "Negative": all_score[0][2],
            }
            st.write("All scores:")
            st.write(detailed_score)
