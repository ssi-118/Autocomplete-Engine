import re
import torch
import torch.nn.functional as F
import streamlit as st
from torch import nn
from pathlib import Path
from streamlit_searchbox import st_searchbox

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FlowType",
    page_icon="⚡",
    layout="centered"
)

# CUSTOM CSS
st.markdown("""
<style>

    /* App background */
    .stApp {
        background: #f8fafc;
    }

    /* Center everything */
    .block-container {
        max-width: 700px;
        padding-top: 6rem;
    }

    /* Title */
    h1 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        color: #0f172a;
    }

    /* Subtitle */
    .saas-subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 2.5rem;
        font-size: 1.05rem;
    }

    /* Search container */
    div[data-baseweb="select"] > div {
        border-radius: 16px !important;
        border: none !important;
        background: #ffffff !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
        padding: 6px !important;
        transition: all 0.2s ease;
    }

    /* Focus effect */
    div[data-baseweb="select"] > div:focus-within {
        box-shadow: 0 0 0 2px #2563eb33, 0 12px 35px rgba(0,0,0,0.12) !important;
    }

    /* Input text */
    div[data-baseweb="select"] input {
        font-size: 16px !important;
        padding: 8px !important;
        color: #0f172a !important;
    }

    /* Dropdown */
    div[data-baseweb="popover"] {
        border-radius: 14px !important;
        box-shadow: 0 18px 40px rgba(0,0,0,0.12) !important;
        overflow: hidden !important;
    }

    /* Options */
    div[role="option"] {
        padding: 12px 16px !important;
        font-size: 14px !important;
    }

    /* Hover */
    div[role="option"]:hover {
        background: #f1f5f9 !important;
    }

    /* Footer metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }

    /* Divider */
    hr {
        margin-top: 3rem;
        margin-bottom: 1.5rem;
    }

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL PATH ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "autocomplete_lstm.pth"


# ---------------- MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, lstm_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(lstm_size, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file missing at: {MODEL_PATH}")
        st.stop()

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = LSTMModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        lstm_size=checkpoint["lstm_size"],
        num_layers=checkpoint["num_layers"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# ---------------- TEXT HELPERS ----------------
def get_current_word(text):
    match = re.search(r"([A-Za-z]+)$", text)
    return match.group(1) if match else ""


def replace_current_word(text, new_word):
    return re.sub(r"([A-Za-z]+)$", new_word, text)


def get_prefix_suggestions(text, unique_words, limit=5):
    current_word = get_current_word(text)
    if len(current_word) < 1:
        return []

    prefix = current_word.lower()
    suggestions = []

    for word in unique_words:
        if word == "<UNK>":
            continue
        if word.startswith(prefix) and word != prefix:
            completed_text = replace_current_word(text, word)
            suggestions.append(completed_text)

        if len(suggestions) >= limit:
            break

    return suggestions


def predict_next_words(model, text, checkpoint, top_k=5):
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return []

    sequence_length = checkpoint["sequence_length"]
    word_to_index = checkpoint["word_to_index"]
    index_to_word = checkpoint["index_to_word"]

    words = words[-sequence_length:]
    indexes = [word_to_index.get(word, word_to_index["<UNK>"]) for word in words]

    input_tensor = torch.tensor([indexes], dtype=torch.long)

    with torch.no_grad():
        outputs, _ = model(input_tensor)
        last_logits = outputs[0, -1]
        probabilities = F.softmax(last_logits, dim=0)
        top_indexes = torch.topk(probabilities, top_k).indices.tolist()

    suggestions = []
    for index in top_indexes:
        next_word = index_to_word[index]
        if next_word != "<UNK>":
            suggestions.append(text.rstrip() + " " + next_word)

    return suggestions


# ---------------- SESSION STATE ----------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""


# ---------------- SEARCH LOGIC ----------------
def search_suggestions(searchterm: str):
    st.session_state.user_text = searchterm

    if not searchterm.strip():
        return []

    current_word = get_current_word(searchterm)

    if current_word:
        suggestions = get_prefix_suggestions(
            text=searchterm,
            unique_words=checkpoint["unique_words"],
            limit=5
        )
    else:
        suggestions = predict_next_words(
            model=model,
            text=searchterm,
            checkpoint=checkpoint,
            top_k=5
        )

    return suggestions


def submit_searchbox_selection(selected_text):
    if selected_text:
        st.session_state.user_text = selected_text + " "


# ---------------- UI ----------------
st.title("⚡ FlowType")
st.markdown(
    "<p class='saas-subtitle'>Next-gen text prediction powered by LSTM</p>",
    unsafe_allow_html=True
)

model, checkpoint = load_model()

selected_value = st_searchbox(
    search_suggestions,
    key="flowtype_searchbox",
    placeholder="Ask anything or start typing...",
    default_searchterm=st.session_state.user_text,
    default=st.session_state.user_text,
    clear_on_submit=False,
    edit_after_submit="option",
    submit_function=submit_searchbox_selection,
)

if selected_value:
    st.session_state.user_text = selected_value + " "


# ---------------- FOOTER ----------------
st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("Vocabulary", checkpoint.get("vocab_size", "—"))
col2.metric("Latency", "~2ms")
col3.metric("Context", f"{checkpoint.get('sequence_length', '—')} words")