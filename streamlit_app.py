import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_model():

    qa_pipeline = pipeline(
        "question-answering",
        model="./models/qa_model_deploy",
        tokenizer="./models/qa_model_deploy"
    )

    return qa_pipeline


qa_pipeline = load_model()


st.set_page_config(
    page_title="Question Answering System",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 NLP Question Answering System")

st.markdown(
    """
    Enter a context passage and ask a question.
    The model will extract the answer from the given text.
    """
)

context = st.text_area(
    "Context",
    height=250,
    placeholder="Enter your passage here..."
)

question = st.text_input(
    "Question",
    placeholder="Ask a question..."
)

if st.button("Get Answer"):

    if context and question:

        result = qa_pipeline(
            question=question,
            context=context
        )

        st.success(
            result["answer"]
        )

        st.metric(
            "Confidence Score",
            f"{result['score']:.4f}"
        )

    else:

        st.warning(
            "Please enter both context and question."
        )