import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
groq_api_key=groq_api_key,
model_name="llama3-8b-8192"
)

st.set_page_config(
page_title="AgriGPT",
page_icon="🌾",
layout="wide"
)

# Sidebar

with st.sidebar:
st.title("🌾 AgriGPT")
st.write("AI-Powered Agricultural Assistant")

```
st.markdown("---")

st.subheader("Features")
st.write("✅ Crop Advice")
st.write("✅ Fertilizer Guidance")
st.write("✅ Pest & Disease Support")
st.write("✅ Irrigation Tips")
st.write("✅ Farming Best Practices")
```

# Main Header

st.markdown(
""" <div style='text-align:center'> <h1 style='color:#3FA34D;'>🌾 AgriGPT</h1> <p>Your AI-Powered Agricultural Assistant</p> </div>
""",
unsafe_allow_html=True
)

# Session state for chat history

if "messages" not in st.session_state:
st.session_state.messages = []

question = st.chat_input("Ask an agriculture-related question...")

if question:
st.session_state.messages.append(
{"role": "user", "content": question}
)

```
response = llm.invoke(question)

st.session_state.messages.append(
    {"role": "assistant", "content": response.content}
)
```

# Display chat history

for msg in st.session_state.messages:
with st.chat_message(msg["role"]):
st.write(msg["content"])

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + LangChain + Groq")
