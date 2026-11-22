"""AgriGPT - AI-Powered Agricultural Assistant & Plant Pathology Lab.
Centralized NLP, RAG, Groq Llama 3.1 8B Instant, and MobileNetV2 Plant Disease Detection.
"""

import os
import sys
import time
from PIL import Image
import streamlit as st

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import AgriGPTPipeline, default_pipeline
from src.vision.disease_classifier import default_disease_classifier
from src.nlp.vector_store import default_vector_store

# --- Page Configuration ---
st.set_page_config(
    page_title="AgriGPT | AI Agricultural Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Main Theme Variables */
    :root {
        --primary-color: #2e7d32;
        --secondary-color: #388e3c;
        --accent-color: #81c784;
        --bg-soft: #f1f8e9;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.3rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #e8f5e9;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 10px;
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }
    .diagnosis-card {
        background-color: #ffffff;
        border-left: 6px solid #2e7d32;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    .treatment-box {
        background-color: #f9fbe7;
        border: 1px solid #c0ca33;
        border-radius: 8px;
        padding: 14px;
        margin-top: 12px;
    }
    .nlp-chip {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        margin: 2px;
        font-weight: 500;
    }
    .chip-crop { background-color: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }
    .chip-disease { background-color: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a; }
    .chip-pest { background-color: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
    .chip-chem { background-color: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9; }
    .chip-qty { background-color: #f3e5f5; color: #4a148c; border: 1px solid #ce93d8; }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if "pipeline" not in st.session_state:
    st.session_state.pipeline = default_pipeline

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to AgriGPT!** I am your AI Agronomist and Crop Pathology assistant.\n\n"
                "- 💬 Ask me about **crop diseases, pest control, soil pH, fertilizers (NPK), or irrigation**.\n"
                "- 📸 Upload a leaf photo in the **Vision Lab** tab to diagnose plant diseases instantly.\n"
                "- 🧠 Explore the **NLP Inspector** to see real-time intent classification, NER, and RAG retrieval!"
            ),
            "source": "AgriGPT System",
            "meta": {}
        }
    ]

if "last_nlp_meta" not in st.session_state:
    st.session_state.last_nlp_meta = None

if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None


# --- Header Section ---
st.markdown("""
<div class="main-header">
    <h1>🌾 AgriGPT: AI Agricultural Advisory System</h1>
    <p>Domain-Specific NLP &bull; Multi-Turn RAG &bull; Groq Llama 3.1 8B Instant &bull; MobileNetV2 Plant Disease Vision</p>
    <div>
        <span class="badge-pill">🧠 Intent Classification (TF-IDF + ML)</span>
        <span class="badge-pill">🏷️ Agri-NER Extraction</span>
        <span class="badge-pill">⚡ FAISS Semantic Search</span>
        <span class="badge-pill">🦙 Groq Llama 3.1 8B</span>
        <span class="badge-pill">👁️ PyTorch MobileNetV2 (38 Classes)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.subheader("⚙️ System Configuration")

    # Groq API Key Input
    groq_key_env = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=groq_key_env,
        type="password",
        help="Enter your Groq API key for Llama 3.1 8B Instant inference. If left empty, AgriGPT runs on its built-in local knowledge engine."
    )

    if api_key_input:
        st.session_state.pipeline.set_groq_api_key(api_key_input)

    # Status indicators
    st.markdown("---")
    st.markdown("**System Health & Modules:**")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.success("✅ NLP Engine")
        st.success("✅ FAISS Vector")
    with col_s2:
        st.success("✅ MobileNetV2")
        if st.session_state.pipeline.llm_client.is_available():
            st.success("⚡ Groq Online")
        else:
            st.info("📦 Local RAG")

    # Active Context Tracker
    st.markdown("---")
    st.subheader("🎯 Active Session Context")
    context_info = st.session_state.pipeline.conversation.get_context_summary()
    st.write(f"**Focused Crop:** `{context_info.get('active_crop') or 'General'}`")
    st.write(f"**Focused Disease:** `{context_info.get('active_disease') or 'None'}`")
    st.write(f"**Total Turns:** `{context_info.get('total_messages')}`")

    if st.button("🔄 Reset Conversation Context", use_container_width=True):
        st.session_state.pipeline.reset_conversation()
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Conversation context reset. How can I help your crop today?",
                "source": "AgriGPT System",
                "meta": {}
            }
        ]
        st.session_state.last_nlp_meta = None
        st.session_state.last_diagnosis = None
        st.rerun()

    st.markdown("---")
    st.markdown("💡 **Sample Inquiries:**")
    samples = [
        "My tomato leaves have brown circular spots with yellow rings",
        "What chemical spray controls stem borer in rice?",
        "Recommended NPK fertilizer dose for wheat per acre?",
        "How to manage yellow leaf curl virus and whiteflies on chilli?",
        "How can I correct alkaline soil with pH 8.2 using gypsum?"
    ]
    for s in samples:
        if st.button(f"👉 {s[:36]}...", key=s, use_container_width=True):
            st.session_state.sample_query = s


# --- Main Application Tabs ---
tab_chat, tab_vision, tab_nlp, tab_kb = st.tabs([
    "💬 AgriChat (Conversational RAG)",
    "📸 Plant Disease Vision Lab",
    "🧠 NLP & RAG Explainability Inspector",
    "📚 Agricultural Knowledge Base"
])


# ==========================================
# TAB 1: Conversational RAG Chat
# ==========================================
with tab_chat:
    st.markdown("### 🌾 Multimodal Agricultural Chat")

    # Display prior messages
    for msg in st.session_state.chat_history:
        avatar = "👨‍🌾" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("source") and msg["role"] == "assistant":
                st.caption(f"Generated via: `{msg['source']}`")

    # Check if a sample query was clicked
    prefill = st.session_state.pop("sample_query", "")

    user_prompt = st.chat_input("Ask any agricultural or plant pathology question...", key="chat_input")
    actual_query = prefill if prefill else user_prompt

    if actual_query:
        # 1. Add user message to UI
        st.session_state.chat_history.append({
            "role": "user",
            "content": actual_query
        })
        with st.chat_message("user", avatar="👨‍🌾"):
            st.markdown(actual_query)

        # 2. Process query via unified pipeline
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing agricultural intent, extracting entities & retrieving knowledge..."):
                start_time = time.time()
                result = st.session_state.pipeline.process_query(actual_query)
                elapsed = time.time() - start_time

            st.markdown(result["response"])
            st.caption(f"⚡ Source: `{result['source']}` &bull; Latency: `{elapsed:.2f}s` &bull; Intent: `{result['nlp_analysis']['intent']['intent']}` ({result['nlp_analysis']['intent']['confidence']*100:.1f}%)")

            # Store for explainability tab
            st.session_state.last_nlp_meta = result

            # Append assistant message to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "source": result["source"],
                "meta": result["nlp_analysis"]
            })


# ==========================================
# TAB 2: Plant Disease Vision Lab
# ==========================================
with tab_vision:
    st.markdown("### 🔬 MobileNetV2 Foliar Disease Diagnosis")
    st.markdown("Upload a close-up photo of an affected plant leaf to classify the disease with deep learning.")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        uploaded_file = st.file_uploader(
            "Choose a leaf image (PNG, JPG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="High-resolution photos with clear view of lesions or spots yield best results."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)

    with col_v2:
        if uploaded_file is not None:
            with st.spinner("Executing MobileNetV2 CNN inference..."):
                diagnosis = st.session_state.pipeline.process_image(uploaded_file)
                st.session_state.last_diagnosis = diagnosis

            status_icon = "🟢" if diagnosis["is_healthy"] else "🔴"
            st.markdown(f"""
            <div class="diagnosis-card">
                <h3>{status_icon} Diagnosis: {diagnosis['crop']} - {diagnosis['disease']}</h3>
                <p><strong>Predicted Class:</strong> <code>{diagnosis['raw_class']}</code></p>
                <p><strong>Model Confidence:</strong> <span style="font-size: 1.2rem; font-weight: bold; color: #2e7d32;">{diagnosis['percentage']}%</span></p>
                <p><strong>Estimated Severity:</strong> <code>{diagnosis['severity']}</code></p>
            </div>
            """, unsafe_allow_html=True)

            # Top 5 distribution
            st.markdown("**Top Predictions Distribution:**")
            for pred in diagnosis["top_k"]:
                label = pred["raw_class"].replace("___", " - ").replace("_", " ")
                st.write(f"*{label}* ({pred['percentage']}%)")
                st.progress(float(pred["confidence"]))

            # Treatment box
            st.markdown(f"""
            <div class="treatment-box">
                <h4>🛡️ Prescribed Management:</h4>
                <p>{diagnosis['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Integrate with Chat button
            if st.button("💬 Consult AgriChat About This Diagnosis", use_container_width=True):
                question = (
                    f"My leaf photo was diagnosed as {diagnosis['crop']} with {diagnosis['disease']} "
                    f"({diagnosis['percentage']}% confidence). What immediate steps should I take to cure this and save my yield?"
                )
                st.session_state.sample_query = question
                st.success("Diagnosis injected! Switch to the AgriChat tab to continue.")
        else:
            st.info("👆 Upload a plant leaf image on the left to begin pathology analysis.")


# ==========================================
# TAB 3: NLP & RAG Explainability Inspector
# ==========================================
with tab_nlp:
    st.markdown("### 🧠 NLP Pipeline & RAG Explainability Engine")
    st.markdown("Inspect how AgriGPT breaks down natural language queries into machine-understandable linguistic signals.")

    meta = st.session_state.last_nlp_meta

    if meta is None:
        st.info("No queries processed in this session yet. Ask a question in the AgriChat tab to inspect real-time NLP extractions!")
    else:
        st.markdown(f"**Query Analyzed:** *\"{meta['query']}\"*")

        col_n1, col_n2 = st.columns(2)

        with col_n1:
            st.markdown("#### 1. Text Preprocessing & Tokenization")
            prep = meta["nlp_analysis"]["preprocessed"]
            st.write(f"**Normalized String:** `{prep['cleaned']}`")
            st.write("**Extracted Lemmas:**", prep["lemmas"])
            if prep["preserved_terms"]:
                st.write("**Preserved Units/Formulas:**", prep["preserved_terms"])

            st.markdown("#### 2. ML-Based Intent Classification")
            intent_res = meta["nlp_analysis"]["intent"]
            st.success(f"**Classified Intent:** `{intent_res['intent']}` ({intent_res['confidence']*100:.1f}%)")
            st.caption(f"Description: {intent_res['description']}")

            st.markdown("**Top Intent Candidates:**")
            for candidate, prob in intent_res.get("top_k", []):
                st.write(f"- `{candidate}`: {prob*100:.1f}%")

        with col_n2:
            st.markdown("#### 3. Named Entity Recognition (NER)")
            grouped_entities = meta["nlp_analysis"]["entities"]["grouped"]
            
            st.markdown("**Detected Crops:**")
            if grouped_entities.get("CROPS"):
                for c in grouped_entities["CROPS"]:
                    st.markdown(f"<span class='nlp-chip chip-crop'>🌱 CROP: {c}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

            st.markdown("**Detected Diseases:**")
            if grouped_entities.get("DISEASES"):
                for d in grouped_entities["DISEASES"]:
                    st.markdown(f"<span class='nlp-chip chip-disease'>🦠 DISEASE: {d}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

            st.markdown("**Detected Pests:**")
            if grouped_entities.get("PESTS"):
                for p in grouped_entities["PESTS"]:
                    st.markdown(f"<span class='nlp-chip chip-pest'>🐛 PEST: {p}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

            st.markdown("**Detected Fertilizers & Agrochemicals:**")
            if grouped_entities.get("FERTILIZERS_CHEMICALS"):
                for f in grouped_entities["FERTILIZERS_CHEMICALS"]:
                    st.markdown(f"<span class='nlp-chip chip-chem'>🧪 CHEMICAL: {f}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

            st.markdown("**Detected Quantities & Dosages:**")
            if grouped_entities.get("METRIC_QUANTITIES"):
                for q in grouped_entities["METRIC_QUANTITIES"]:
                    st.markdown(f"<span class='nlp-chip chip-qty'>📏 DOSAGE: {q}</span>", unsafe_allow_html=True)
            else:
                st.write("None")

        st.markdown("---")
        st.markdown("#### 4. Semantic Search & Retrieved RAG Chunks")
        docs = meta.get("retrieved_documents", [])
        if docs:
            for i, doc in enumerate(docs):
                with st.expander(f"📄 Match #{i+1}: {doc['title']} (Crop: {doc['crop']}, Score: {doc['score']})", expanded=(i==0)):
                    st.write(doc["content"])
                    st.caption(f"Category: `{doc['category']}` | Keywords: `{', '.join(doc.get('keywords', []))}`")
        else:
            st.warning("No knowledge base documents surpassed the semantic retrieval threshold.")


# ==========================================
# TAB 4: Agricultural Knowledge Base
# ==========================================
with tab_kb:
    st.markdown("### 📚 Integrated Agricultural Knowledge Repository")
    st.markdown("Browse curated agronomic facts, disease treatment protocols, and soil science guidelines.")

    kb_docs = default_vector_store.documents
    crop_filter = st.selectbox("Filter by Crop:", ["All"] + sorted(list({d.get("crop", "General") for d in kb_docs})))
    category_filter = st.selectbox("Filter by Category:", ["All", "disease", "pest", "fertilizer", "irrigation"])

    filtered_docs = [
        d for d in kb_docs
        if (crop_filter == "All" or d.get("crop") == crop_filter) and
           (category_filter == "All" or d.get("category") == category_filter)
    ]

    st.write(f"Showing **{len(filtered_docs)}** knowledge documents:")

    for doc in filtered_docs:
        with st.expander(f"🌾 [{doc.get('crop')}] {doc.get('title')}"):
            st.markdown(doc.get("content"))
            st.markdown(f"**Tags:** `{', '.join(doc.get('keywords', []))}`")
