import streamlit as st

from qa_utils import annotate_answer, get_qa_answers

_, col1, _ = st.beta_columns(3)

with col1:
    st.image("is2alni_logo.png", width=200)
    st.title("إسألني أي شيء")

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

st.sidebar.header("Info")
st.sidebar.image("AraELECTRA.png", width=150)
st.sidebar.write("Powered by [AraELECTRA](https://github.com/aub-mind/arabert)")

st.sidebar.write("\n")
n_answers = st.sidebar.slider(
    "Max. number of answers", min_value=1, max_value=10, value=2, step=1
)

question = st.text_input("", value="من هو جو بايدن؟")
if "؟" not in question:
    question += "؟"

run_query = st.button("أجب")
if run_query:
    # https://discuss.streamlit.io/t/showing-a-gif-while-st-spinner-runs/5084
    with st.spinner("... جاري البحث "):
        results_dict = get_qa_answers(question)

    if len(results_dict) > 0:
        st.write("## :الأجابات هي")
        for result in results_dict["results"][:n_answers]:
            annotate_answer(result)
            f"[**المصدر**](<{result['link']}>)"
    else:
        st.write("## 😞 ليس لدي جواب")
