import streamlit as st
from .services import TextGeneration
from tokenizers import Tokenizer
from functools import lru_cache

# @st.cache(allow_output_mutation=False, hash_funcs={Tokenizer: str})
@lru_cache(maxsize=1)
def load_text_generator():
    generator = TextGeneration()
    generator.load()
    return generator


generator = load_text_generator()

qa_prompt = """
            أجب عن السؤال التالي:
            """
qa_prompt_post = """ الجواب هو  """
qa_prompt_post_year = """ في سنة: """


def write():
    st.markdown(
        """
        <h1 style="text-align:left;">Arabic Language Generation</h1>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar

    # Taken from https://huggingface.co/spaces/flax-community/spanish-gpt2/blob/main/app.py
    st.sidebar.subheader("Configurable parameters")

    model_name = st.sidebar.selectbox(
        "Model Selector",
        options=[
            "AraGPT2-Base",
            # "AraGPT2-Medium",
            # "Aragpt2-Large",
            "AraGPT2-Mega",
        ],
        index=0,
    )

    max_new_tokens = st.sidebar.number_input(
        "Maximum length",
        min_value=0,
        max_value=1024,
        value=100,
        help="The maximum length of the sequence to be generated.",
    )
    temp = st.sidebar.slider(
        "Temperature",
        value=1.0,
        min_value=0.1,
        max_value=100.0,
        help="The value used to module the next token probabilities.",
    )
    top_k = st.sidebar.number_input(
        "Top k",
        value=10,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    top_p = st.sidebar.number_input(
        "Top p",
        value=0.95,
        help=" If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    do_sample = st.sidebar.selectbox(
        "Sampling?",
        (True, False),
        help="Whether or not to use sampling; use greedy decoding otherwise.",
    )
    num_beams = st.sidebar.number_input(
        "Number of beams",
        min_value=1,
        max_value=10,
        value=3,
        help="The number of beams to use for beam search.",
    )
    repetition_penalty = st.sidebar.number_input(
        "Repetition Penalty",
        min_value=0.0,
        value=3.0,
        step=0.1,
        help="The parameter for repetition penalty. 1.0 means no penalty",
    )
    no_repeat_ngram_size = st.sidebar.number_input(
        "No Repeat N-Gram Size",
        min_value=0,
        value=3,
        help="If set to int > 0, all ngrams of that size can only occur once.",
    )

    st.write("#")

    col = st.columns(2)

    col[0].image("images/AraGPT2.png", width=200)

    st.markdown(
        """

        <h3 style="text-align:left;">AraGPT2 is GPT2 model trained from scratch on 77GB of Arabic text.</h3>
        <h4 style="text-align:left;"> More details in our <a href="https://github.com/aub-mind/arabert/tree/master/aragpt2">repo</a>.</h4>

        <p style="text-align:left;"><p>
        <p style="text-align:left;">Use the generation paramters on the sidebar to adjust generation quality.</p>
        <p style="text-align:right;"><p>
        """,
        unsafe_allow_html=True,
    )

    # col[0].write(
    #     "AraGPT2 is trained from screatch on 77GB of Arabic text. More details in our [repo](https://github.com/aub-mind/arabert/tree/master/aragpt2)."
    # )
    # st.write("## Generate Arabic Text")

    st.markdown(
        """
        <style>
        p, div, input, label, textarea{
        text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Prompt",
        "يحكى أن مزارعا مخادعا قام ببيع بئر الماء الموجود في أرضه لجاره مقابل مبلغ كبير من المال",
    )
    if st.button("Generate"):
        with st.spinner("Generating..."):
            generated_text = generator.generate(
                prompt=prompt,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            st.write(generated_text)

    st.markdown("---")
    st.subheader("")
    st.markdown(
        """
        <p style="text-align:left;"><p>
        <h2 style="text-align:left;">Zero-Shot Question Answering</h2>

        <p style="text-align:left;">Adjust the maximum length to closely match the expected output length. Setting the Sampling paramter to False is recommended</p>
        <p style="text-align:left;"><p>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_input(
        "Question", "من كان رئيس ألمانيا النازية في الحرب العالمية الثانية ؟"
    )
    is_date = st.checkbox("Help the model: Is the answer a date?")
    if st.button("Answer"):

        prompt2 = qa_prompt + question + qa_prompt_post
        if is_date:
            prompt2 += qa_prompt_post_year
        else:
            prompt2 += " : "
        with st.spinner("Thinking..."):
            answer = generator.generate(
                prompt=prompt2,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            st.write(answer)
