import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Page settings
st.set_page_config(
    page_title="AgriGPT",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    color: #2E8B57;
    font-size: 3rem;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #666666;
    margin-bottom: 2rem;
}

.answer-box {
    background-color: #f0f8f0;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #2E8B57;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🌾 AgriGPT")
    st.write("AI-Powered Agricultural Assistant")

    st.markdown("---")

    st.write("### Features")
    st.write("✅ Crop Advice")
    st.write("✅ Fertilizer Guidance")
    st.write("✅ Pest & Disease Support")
    st.write("✅ Irrigation Tips")
    st.write("✅ Farming Best Practices")

# Main content
st.markdown('<div class="title">🌾 AgriGPT</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Your AI-Powered Agricultural Assistant</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Ask any agriculture-related question:",
    placeholder="Example: What are the symptoms of nitrogen deficiency in rice?"
)

if question:
    with st.spinner("🌱 Analyzing..."):
        response = llm.invoke(question)

    st.markdown(
        f"""
        <div class="answer-box">
        {response.content}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit + LangChain + Groq
    </div>
    """,
    unsafe_allow_html=True
)