import streamlit as st
from langchain_groq import ChatGroq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgriGPT",
    page_icon="🌾",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🌾 AgriGPT")
    st.write("AI-Powered Agricultural Assistant")

    st.markdown("---")

    st.subheader("Features")
    st.write("✅ Crop Advice")
    st.write("✅ Fertilizer Guidance")
    st.write("✅ Pest & Disease Support")
    st.write("✅ Irrigation Tips")
    st.write("✅ Farming Best Practices")

# ---------------- API KEY ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------- MODEL ----------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ---------------- MAIN UI ----------------
st.markdown(
    """
    <h1 style='text-align:center;color:#4CAF50;'>
    🌾 AgriGPT
    </h1>
    <p style='text-align:center;font-size:20px;'>
    Your AI-Powered Agricultural Assistant
    </p>
    """,
    unsafe_allow_html=True
)

question = st.text_input(
    "Ask any agriculture-related question:",
    placeholder="Example: What are the symptoms of nitrogen deficiency in rice?"
)

if question:
    with st.spinner("Thinking..."):
        try:
            response = llm.invoke(question)

            st.success("Answer")

            st.markdown(
                f"""
                <div style="
                background-color:#E8F5E9;
                padding:20px;
                border-radius:10px;
                color:black;
                font-size:18px;">
                {response.content}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit + LangChain + Groq</center>",
    unsafe_allow_html=True
)