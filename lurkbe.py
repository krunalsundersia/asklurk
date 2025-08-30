import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import asyncio
from mistralai import Mistral

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- INITIALIZE API CLIENTS ---
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
except Exception as e:
    groq_client = None
    st.warning(f"Groq client failed: {str(e)[:50]}")

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) if os.getenv("GOOGLE_API_KEY") else None
except Exception as e:
    genai = None
    st.warning(f"Google API failed: {str(e)[:50]}")

try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except Exception as e:
    openai_client = None
    st.warning(f"OpenAI client failed: {str(e)[:50]}")

try:
    deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com") if os.getenv("DEEPSEEK_API_KEY") else None
except Exception as e:
    deepseek_client = None
    st.warning(f"DeepSeek client failed: {str(e)[:50]}")

try:
    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY")) if os.getenv("MISTRAL_API_KEY") else None
except Exception as e:
    mistral_client = None
    st.warning(f"Mistral client failed: {str(e)[:50]}")

try:
    aiml_client = OpenAI(base_url="https://api.aimlapi.com/v1", api_key=os.getenv("AIML_API_KEY")) if os.getenv("AIML_API_KEY") else None
except Exception as e:
    aiml_client = None
    st.warning(f"AIML client failed: {str(e)[:50]}")

try:
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
except Exception as e:
    anthropic_client = None
    st.warning(f"Anthropic client failed: {str(e)[:50]}")

# --- DEFINE GENERATE_RESPONSE FUNCTION ---
async def generate_response(model_name, config, prompt):
    try:
        if config["provider"] == "gemini":
            if not genai:
                return None, "❌ Google API key missing"
            model = genai.GenerativeModel(config["model"])
            response = model.generate_content(prompt, request_options={"timeout": 15})
            return response.text, None
        elif config["provider"] == "groq":
            if not groq_client:
                return None, "❌ Groq API key missing"
            chat = groq_client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=15
            )
            return chat.choices[0].message.content, None
        elif config["provider"] == "openai":
            if not openai_client:
                return None, "❌ OpenAI API key missing"
            chat = openai_client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=15
            )
            return chat.choices[0].message.content, None
        elif config["provider"] == "deepseek":
            if not deepseek_client:
                return None, "❌ DeepSeek API key missing"
            chat = deepseek_client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=15
            )
            return chat.choices[0].message.content, None
        elif config["provider"] == "mistral":
            if not mistral_client:
                return None, "❌ Mistral API key missing"
            response = await mistral_client.chat.stream_async(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
            )
            full_response = ""
            async for chunk in response:
                if chunk.data.choices[0].delta.content is not None:
                    full_response += chunk.data.choices[0].delta.content
            return full_response, None
        elif config["provider"] == "aiml":
            if not aiml_client:
                return None, "❌ AIML API key missing"
            chat = aiml_client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "system", "content": "You are an AI assistant who knows everything."}, {"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=15
            )
            return chat.choices[0].message.content, None
        elif config["provider"] == "anthropic":
            if not anthropic_client:
                return None, "❌ Anthropic API key missing"
            chat = anthropic_client.messages.create(
                model=config["model"],
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
                timeout=15
            )
            return chat.content[0].text, None
        else:
            return None, f"⚠️ Not supported: {config['name']}"
    except Exception as e:
        return None, f"⚠️ Error: {str(e)[:100]}"

# --- SESSION STATE INIT ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_chat' not in st.session_state:
    st.session_state.current_chat = []
if 'selected_history' not in st.session_state:
    st.session_state.selected_history = None
if 'user_prompt' not in st.session_state:
    st.session_state.user_prompt = ""
if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'chat_input_key' not in st.session_state:
    st.session_state.chat_input_key = str(uuid.uuid4())  # Unique key for chat input
if 'previous_selected_names' not in st.session_state:
    st.session_state.previous_selected_names = []

# --- MODEL CONFIGURATION ---
AVAILABLE_MODELS = {
    "GPT-5": {"provider": "openai", "model": "gpt-5-2025-08-07", "logo": "🧠", "name": "GPT"},
    "Grok 4": {"provider": "groq", "model": "grok-beta", "logo": "⚡", "name": "Grok"},
    "Gemini 2.5 Pro": {"provider": "gemini", "model": "gemini-2.5-pro", "logo": "🔷", "name": "Gemini"},
    "Mistral Large 2": {"provider": "mistral", "model": "mistral-large-latest", "logo": "🌫️", "name": "Mistral"},
    "Qwen3 235B A22B": {"provider": "aiml", "model": "qwen3-235b-a22b-thinking-2507", "logo": "📊", "name": "Qwen"},
    "DeepSeek-R1": {"provider": "deepseek", "model": "deepseek-chat", "logo": "🔍", "name": "DeepSeek"},
    "Llama 4 Maverick": {"provider": "aiml", "model": "meta-llama/llama-4-maverick", "logo": "🐫", "name": "Llama"},
    "GPT-4.1": {"provider": "openai", "model": "gpt-4.1-2025-04-14", "logo": "🧠", "name": "GPT"},
    "Gemini 1.5 Pro": {"provider": "gemini", "model": "gemini-1.5-pro", "logo": "🔷", "name": "Gemini"},
    "Claude 3.5 Sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "logo": "🎶", "name": "Claude"}
}

# --- CUSTOM CSS TO REMOVE ANIMATION AND TITLE ---
st.markdown(
    """
    <style>
    /* Hide the loading animation */
    [data-testid="stSpinner"] {
        display: none !important;
    }
    /* Hide the default Streamlit title or branding */
    [data-testid="stAppViewContainer"] > div:first-child h1,
    [data-testid="stAppViewContainer"] > div:first-child .stApp > h1 {
        display: none !important;
    }
    /* Remove footer branding */
    footer [data-testid="stFooter"] {
        display: none !important;
    }
    /* Ensure main content loads without delay */
    .main .block-container {
        padding-top: 0rem;
        margin-top: 0rem;
    }
    /* Dark Theme (rest of your existing CSS) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@500;600&display=swap');
    body, .stApp, .main, [data-testid="stAppViewContainer"] {
        background: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
    }
    [data-testid="stSidebar"] {
        background: #1e293b;
        border-right: 1px solid #334155;
        padding: 1.5rem;
    }
    .sidebar-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        color: #ffffff;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .sidebar-buttons {
        display: flex;
        gap: 10px;
        margin-bottom: 1.5rem;
    }
    .sidebar-history {
        display: flex;
        flex-direction: column;
        gap: 12px;
        overflow-y: auto;
        max-height: calc(100vh - 250px);
    }
    .sidebar-history::-webkit-scrollbar {
        width: 6px;
    }
    .sidebar-history::-webkit-scrollbar-track {
        background: #0f172a;
    }
    .sidebar-history::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 1px;
    }
    .history-item {
        padding: 10px 12px;
        border-radius: 1px;
        cursor: pointer;
        background: #0f172a;
        border: 1px solid #334155;
        transition: all 0.2s ease;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .history-item:hover {
        background: #2a3448;
        border-color: #6366F1;
        animation: glow-blue-red 1.5s ease-in-out infinite alternate;
    }
    .history-item.active {
        background: #2a3448;
        border-color: #6366F1;
        animation: glow-blue-red 1.5s ease-in-out infinite alternate;
    }
    .history-timestamp {
        font-size: 0.75rem;
        color: #64748b;
    }
    .history-question {
        font-size: 0.9rem;
        color: #e2e8f0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    @keyframes glow-blue-red {
        0% { box-shadow: 0 0 10px #4f46e5, 0 0 20px rgba(79, 70, 229, 0.5); }
        100% { box-shadow: 0 0 10px #ef4444, 0 0 20px rgba(239, 68, 68, 0.5); }
    }
    [data-testid="stTextInput"] input,
    [data-testid="stMultiSelect"] > div:first-child {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 1px !important;
        padding: 1px 1px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 0 !important;
    }
    [data-testid="stTextInput"] input:hover,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stMultiSelect"] > div:hover,
    [data-testid="stMultiSelect"] > div:focus-within {
        border: 1px solid #334155 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stMultiSelect"] span {
        background-color: #6366F1 !important;
        color: white !important;
        border-radius: 0px !important;
        padding: 4px 10px !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #6366F1;
        color: white;
        border: none;
        border-radius: 1px;
        padding: 8px 16px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #5a52c7;
        transform: translateY(-1px);
    }
    .models-container {
        display: flex;
        flex-direction: row;
        gap: 10px;
        padding: 20px 0;
        overflow-x: auto;
        justify-content: flex-start;
    }
    .models-container::-webkit-scrollbar {
        height: 6px;
    }
    .models-container::-webkit-scrollbar-track {
        background: #0f172a;
    }
    .models-container::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 1px;
    }
    .card {
        width: 500px;
        max-width: 500px;
        min-height: auto;
        border-radius: 1px;
        padding: 10px;
        background: #1e293b;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        transition: all 0.3s ease;
    }
    .card.full-screen {
        width: 100%;
        max-width: 100%;
    }
    .card.two-column {
        width: 50%;
        max-width: 50%;
    }
    .card:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        border-color: #6366F1;
    }
    .card.zoomed {
        transform: scale(1.1);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4);
        border-color: #6366F1;
        z-index: 10;
    }
    .model-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 5px 0;
    }
    .assistant-message {
        background: #1e293b;
        border: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        width: 100%;
        height: auto;
        max-height: 400px;
        color: #e2e8f0;
        font-size: 0.95rem;
        overflow-wrap: break-word;
        word-break: break-all;
        overflow-y: auto;
    }
    .assistant-message p {
        margin: 0;
        padding: 10px;
    }
    .error-section {
        padding: 12px;
        background: #370617;
        border: 1px solid #7C1D1D;
        border-radius: 1px;
        color: #FCA5A5;
        font-size: 0.85rem;
    }
    .warning-section {
        padding: 12px;
        background: #412B0D;
        border: 1px solid #9A4C00;
        border-radius: 1px;
        color: #FBBF24;
        font-size: 0.85rem;
    }
    .card-footer {
        font-size: 0.75rem;
        color: #64748b;
        text-align: right;
        margin-top: 10px;
    }
    .user-message {
        background: #2a3448;
        border: 1px solid #334155;
        border-radius: 1px;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-end;
        color: #e2e8f0;
        font-size: 0.95rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .user-message p {
        margin: 0;
    }
    hr {
        border: none;
        height: 1px;
        background: #334155;
        margin: 20px 0;
    }
    .chatbox-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 300px;
        z-index: 1000;
    }
    .chat-input-wrapper {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 1px;
        padding: 10px;
        display: flex;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .chat-input-wrapper:hover {
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
    }
    .chat-input-wrapper textarea {
        flex-grow: 1;
        background: transparent;
        border: none;
        color: #e2e8f0;
        font-size: 0.95rem;
        outline: none;
        padding: 8px;
        resize: none;
        min-height: 40px;
        max-height: 100px;
        overflow-y: auto;
        font-family: 'Inter', sans-serif;
    }
    .chat-input-wrapper textarea::placeholder {
        color: #64748b;
    }
    .send-button {
        background: linear-gradient(45deg, #4f46e5, #ef4444);
        border: none;
        border-radius: 3px;
        padding: 8px 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        margin-left: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #ffffff;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .send-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 10px rgba(79, 70, 229, 0.5);
    }
    @keyframes fadeInScale {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    [data-testid="stMultiSelect"] label {
        display: none !important;
    }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.card');
            let activeCard = null;
            cards.forEach(card => {
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    if (activeCard === this) {
                        this.classList.remove('zoomed');
                        activeCard = null;
                    } else {
                        if (activeCard) activeCard.classList.remove('zoomed');
                        this.classList.add('zoomed');
                        activeCard = this;
                    }
                });
            });
        });
    </script>
    """,
    unsafe_allow_html=True
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Lurk AI",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SIDEBAR WITH LOGO AND CHAT HISTORY ---
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 1.5rem;'>
           Lurk AI
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("New Chat", key="new_chat"):
        st.session_state.current_chat = []
        st.session_state.selected_history = None
        st.session_state.user_prompt = ""
        st.session_state.chat_input_key = str(uuid.uuid4())
        st.session_state.previous_selected_names = []
        st.rerun()
    if st.button("Clear History", key="clear_history"):
        st.session_state.chat_history = []
        st.session_state.current_chat = []
        st.session_state.selected_history = None
        st.session_state.user_prompt = ""
        st.session_state.chat_input_key = str(uuid.uuid4())
        st.session_state.previous_selected_names = []
        st.rerun()
    if st.button("Clear Cache", key="clear_cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown("<div class='sidebar-history'>", unsafe_allow_html=True)
    if st.session_state.chat_history:
        for idx, history in enumerate(st.session_state.chat_history):
            is_active = st.session_state.selected_history == idx
            history_class = "history-item active" if is_active else "history-item"
            question_snippet = history["question"][:30] + "..." if len(history["question"]) > 30 else history["question"]
            st.markdown(
                f"""
                <div class="{history_class}" onclick="document.getElementById('history_{idx}').click()">
                    <div class="history-timestamp">{history['timestamp']}</div>
                    <div class="history-question">{question_snippet}</div>
                </div>
                <input type="radio" id="history_{idx}" name="history_select" style="display:none;" {"checked" if is_active else ""}>
                """,
                unsafe_allow_html=True
            )
            if st.session_state.get(f"history_select") == f"history_{idx}":
                st.session_state.selected_history = idx
                st.session_state.current_chat = [
                    {"role": "user", "content": history["question"]},
                    {"role": "assistant", "content": history["responses"]}
                ]
                st.session_state.user_prompt = ""
                st.session_state.chat_input_key = str(uuid.uuid4())
                st.rerun()
    else:
        st.markdown("<p style='color: #64748b; font-style: italic;'>No chat history yet.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- MODEL SELECTION ---
st.markdown("<h3 style='text-align: center; margin-bottom: 1rem; font-size: 1.5rem; color: #ffffff; animation: fadeInScale 1s ease-in-out;'>Chat with Your AI Crew</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-style: italic; margin-bottom: 1.5rem;'>Select your favorite AI models to chat with!</p>", unsafe_allow_html=True)
selected_names = st.multiselect(
    "",
    options=list(AVAILABLE_MODELS.keys()),
    default=[],
    max_selections=10,
    label_visibility="hidden",
    key="model_select"
)

if len(selected_names) < 1:
    st.info("Please select at least 1 model to compare.", icon="💡")
    st.stop()

# Check for model selection changes and regenerate all responses
if selected_names != st.session_state.previous_selected_names and st.session_state.chat_history:
    with st.spinner("🧠 Regenerating responses for all models..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for history in st.session_state.chat_history:
            prompt = history["question"]
            current_responses = {}
            future_to_model = {
                loop.run_until_complete(generate_response(name, AVAILABLE_MODELS[name], prompt)): name
                for name in selected_names
            }
            for future, model_name in future_to_model.items():
                try:
                    reply, error = future
                    if reply:
                        current_responses[model_name] = reply
                    elif error:
                        current_responses[model_name] = f"[Error] {error}"
                except Exception:
                    current_responses[model_name] = "Failed to respond"
            history["responses"] = current_responses
        loop.close()
    st.session_state.previous_selected_names = selected_names.copy()
    st.rerun()

# --- DISPLAY CONVERSATION HISTORY ---
st.markdown("<h3 style='text-align: center; margin: 1.5rem 0 0.8rem; font-size: 1.5rem; color: #ffffff;'>What's on your mind?</h3>", unsafe_allow_html=True)

if st.session_state.selected_history is not None:
    history = st.session_state.chat_history[st.session_state.selected_history]
    st.markdown(f"<div class='user-message'><p>{history['question']}</p></div>", unsafe_allow_html=True)
    responses = history["responses"]
    if len(selected_names) == 1:
        cards_html = '<div class="models-container" style="width: 100%;">'
        for model_name, config in {name: AVAILABLE_MODELS[name] for name in selected_names}.items():
            card = f"""
            <div class='card full-screen' data-card-id='{model_name}'>
                <h3 class='model-name'>{model_name}</h3>
            """
            reply = responses.get(model_name)
            if reply:
                safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
            else:
                card += "<div class='warning-section'>⚠️ No response available</div>"
            card += "</div>"
            cards_html += card
        cards_html += "</div>"
    elif len(selected_names) == 2:
        cards_html = '<div class="models-container" style="width: 100%;">'
        for i, (model_name, config) in enumerate({name: AVAILABLE_MODELS[name] for name in selected_names}.items()):
            card = f"""
            <div class='card two-column' data-card-id='{model_name}'>
                <h3 class='model-name'>{model_name}</h3>
            """
            reply = responses.get(model_name)
            if reply:
                safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
            else:
                card += "<div class='warning-section'>⚠️ No response available</div>"
            card += "</div>"
            cards_html += card
        cards_html += "</div>"
    else:
        cards_html = '<div class="models-container" style="width: 100%;">'
        for model_name, config in {name: AVAILABLE_MODELS[name] for name in selected_names}.items():
            card = f"""
            <div class='card' data-card-id='{model_name}'>
                <h3 class='model-name'>{model_name}</h3>
            """
            reply = responses.get(model_name)
            if reply:
                safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
            else:
                card += "<div class='warning-section'>⚠️ No response available</div>"
            card += "</div>"
            cards_html += card
        cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
else:
    for history in st.session_state.chat_history:
        st.markdown(f"<div class='user-message'><p>{history['question']}</p></div>", unsafe_allow_html=True)
        responses = history["responses"]
        if len(selected_names) == 1:
            cards_html = '<div class="models-container" style="width: 100%;">'
            for model_name, config in {name: AVAILABLE_MODELS[name] for name in selected_names}.items():
                card = f"""
                <div class='card full-screen' data-card-id='{model_name}'>
                    <h3 class='model-name'>{model_name}</h3>
                """
                reply = responses.get(model_name)
                if reply:
                    safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                    card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
                else:
                    card += "<div class='warning-section'>⚠️ No response available</div>"
                card += "</div>"
                cards_html += card
            cards_html += "</div>"
        elif len(selected_names) == 2:
            cards_html = '<div class="models-container" style="width: 100%;">'
            for i, (model_name, config) in enumerate({name: AVAILABLE_MODELS[name] for name in selected_names}.items()):
                card = f"""
                <div class='card two-column' data-card-id='{model_name}'>
                    <h3 class='model-name'>{model_name}</h3>
                """
                reply = responses.get(model_name)
                if reply:
                    safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                    card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
                else:
                    card += "<div class='warning-section'>⚠️ No response available</div>"
                card += "</div>"
                cards_html += card
            cards_html += "</div>"
        else:
            cards_html = '<div class="models-container" style="width: 100%;">'
            for model_name, config in {name: AVAILABLE_MODELS[name] for name in selected_names}.items():
                card = f"""
                <div class='card' data-card-id='{model_name}'>
                    <h3 class='model-name'>{model_name}</h3>
                """
                reply = responses.get(model_name)
                if reply:
                    safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                    card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
                else:
                    card += "<div class='warning-section'>⚠️ No response available</div>"
                card += "</div>"
                cards_html += card
            cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.submit and st.session_state.user_prompt.strip():
        prompt_to_process = st.session_state.user_prompt
        st.session_state.user_prompt = ""
        st.session_state.chat_input_key = str(uuid.uuid4())
        st.markdown(f"<div class='user-message'><p>{prompt_to_process}</p></div>", unsafe_allow_html=True)
        responses = {}
        with st.spinner("🧠 Generating responses..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            future_to_model = {
                loop.run_until_complete(generate_response(name, AVAILABLE_MODELS[name], prompt_to_process)): name
                for name in selected_names
            }
            if len(selected_names) == 1:
                cards_html = '<div class="models-container" style="width: 100%;">'
            elif len(selected_names) == 2:
                cards_html = '<div class="models-container" style="width: 100%;">'
            else:
                cards_html = '<div class="models-container" style="width: 100%;">'
            for future, model_name in future_to_model.items():
                config = AVAILABLE_MODELS[model_name]
                try:
                    reply, error = future
                    responses[model_name] = reply
                    if len(selected_names) == 1:
                        card_class = 'full-screen'
                    elif len(selected_names) == 2:
                        card_class = 'two-column'
                    else:
                        card_class = ''
                    card = f"""
                    <div class='card {card_class}' data-card-id='{model_name}'>
                        <h3 class='model-name'>{model_name}</h3>
                    """
                    if error:
                        card += f"<div class='error-section'>[Error] {error}</div>"
                    elif reply:
                        safe_reply = reply.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
                        card += f"<div class='assistant-message'><p>{safe_reply}</p></div>"
                    else:
                        card += "<div class='warning-section'>⚠️ No response</div>"
                    card += "</div>"
                    cards_html += card
                except Exception:
                    card = f"<div class='card'><div class='error-section'>Failed to respond</div></div>"
                    cards_html += card
            cards_html += "</div>"
            st.markdown(cards_html, unsafe_allow_html=True)
            loop.close()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append({
            "timestamp": timestamp,
            "question": prompt_to_process,
            "responses": responses.copy()
        })
        st.session_state.current_chat.append({"role": "user", "content": prompt_to_process})
        st.session_state.current_chat.append({"role": "assistant", "content": responses})
        st.session_state.previous_selected_names = selected_names.copy()
        st.session_state.submit = False
        st.rerun()

# --- CHATBOX AT BOTTOM-RIGHT ---
with st.container():
    st.markdown("<div class='chatbox-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([9, 1])
    with col1:
        user_prompt = st.text_area(
            "Type your question...",
            value=st.session_state.user_prompt,
            key=st.session_state.chat_input_key,
            height=40,
            max_chars=1000,
            label_visibility="collapsed"
        )
        st.session_state.user_prompt = user_prompt
    with col2:
        if st.button("Send", key="send_button", disabled=not user_prompt.strip()):
            st.session_state.submit = True
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- JAVASCRIPT FOR ENTER KEY SUBMISSION ---
st.markdown(
    """
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.card');
            let activeCard = null;
            cards.forEach(card => {
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    if (activeCard === this) {
                        this.classList.remove('zoomed');
                        activeCard = null;
                    } else {
                        if (activeCard) activeCard.classList.remove('zoomed');
                        this.classList.add('zoomed');
                        activeCard = this;
                    }
                });
            });
            // Handle Enter key for submission
            document.querySelector('textarea').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    document.querySelector('[data-testid="stButton"]').click();
                }
            });
        });
    </script>
    """,
    unsafe_allow_html=True
)