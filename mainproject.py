import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import io
import re

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind AI",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f1117; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }

    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 4px;
        margin-bottom: 28px;
    }

    .stat-card {
        background: #1e1e2e;
        border: 1px solid #2e2e3e;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }

    .stat-val { font-size: 1.8rem; font-weight: 700; color: #06b6d4; }
    .stat-lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }

    .response-box {
        background: #1e1e2e;
        border: 1px solid #2e2e3e;
        border-left: 3px solid #7c3aed;
        border-radius: 12px;
        padding: 20px 24px;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e2e8f0;
        margin-top: 12px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124,58,237,0.4);
    }

    div[data-testid="stTextArea"] textarea {
        background: #1e1e2e !important;
        color: #e2e8f0 !important;
        border: 1px solid #2e2e3e !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_table" not in st.session_state:
    st.session_state.last_table = None


# ─── Helper Functions ────────────────────────────────────────────────────────
def load_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format.")


def build_data_context(df: pd.DataFrame, max_rows: int = 100) -> str:
    sample = df.head(max_rows)
    info = {
        "columns": list(df.columns),
        "total_rows": len(df),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "sample_data": sample.to_dict(orient="records"),
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include="number").empty else {},
    }
    return json.dumps(info, default=str)


def query_ai(api_key: str, user_query: str, data_context: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""You are an expert data analyst AI. The user has uploaded a dataset and wants to query it.

You receive data as JSON with: columns, total_rows, dtypes, sample_data (up to 100 rows), and numeric_summary.

ALWAYS respond in valid JSON only (no markdown, no code fences, no explanation outside JSON).

For table results use:
{{"type": "table", "answer": "Brief explanation", "rows": [...list of row objects...]}}

For text/stats results use:
{{"type": "text", "answer": "Your detailed answer here"}}

Guidelines:
- Perform real calculations (sum, avg, count, min, max) from the data
- For filtering questions, return matching rows as a table
- Reference actual column names and values
- Never invent data

Dataset:
{data_context}

Question: {user_query}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown fences if Gemini wraps response
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {"type": "text", "answer": raw}


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 DataMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload CSV or Excel → Ask anything in plain English &nbsp;|&nbsp; Powered by Google Gemini ✨ (Free)</div>', unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
    )
    st.markdown(
        "<div style='font-size:11px; color:#64748b; margin-top:-8px;'>🔑 Get free key → <a href='https://aistudio.google.com/app/apikey' target='_blank' style='color:#7c3aed;'>aistudio.google.com</a></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### 📂 Upload File")
    uploaded = st.file_uploader(
        "Drop CSV or Excel here",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded:
        try:
            df = load_file(uploaded)
            st.session_state.df = df
            st.success(f"✅ **{uploaded.name}** loaded!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    if st.session_state.history:
        st.markdown("### 🕑 History")
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            label = h[:40] + ("..." if len(h) > 40 else "")
            if st.button(f"↩ {label}", key=f"hist_{i}", use_container_width=True):
                st.session_state["prefill_query"] = h

# ─── Main Area ───────────────────────────────────────────────────────────────
df = st.session_state.df

if df is not None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-val">{len(df.columns)}</div><div class="stat-lbl">Columns</div></div>', unsafe_allow_html=True)
    with col3:
        n_numeric = df.select_dtypes(include="number").shape[1]
        st.markdown(f'<div class="stat-card"><div class="stat-val">{n_numeric}</div><div class="stat-lbl">Numeric</div></div>', unsafe_allow_html=True)
    with col4:
        missing = df.isnull().sum().sum()
        st.markdown(f'<div class="stat-card"><div class="stat-val">{missing}</div><div class="stat-lbl">Missing</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📊 Data Preview (first 50 rows)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True, height=280)

    with st.expander("📈 Column Summary", expanded=False):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown("---")

    # Example query chips
    st.markdown("**💡 Quick Queries:**")
    cols = list(df.columns)
    numeric_col = next((c for c in cols if pd.api.types.is_numeric_dtype(df[c])), cols[-1])

    examples = [
        "Show first 10 rows",
        "How many rows total?",
        "What are all the column names and their types?",
        f"What is the average of '{numeric_col}'?",
        f"Show rows where '{cols[0]}' is not empty",
        f"Find the maximum value in '{numeric_col}'",
    ]

    ecols = st.columns(3)
    for i, ex in enumerate(examples):
        with ecols[i % 3]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["prefill_query"] = ex

    st.markdown("<br>", unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill_query", "")
    query = st.text_area(
        "🔍 Ask anything about your data",
        value=prefill,
        placeholder="e.g. Filter rows where Age > 30 and show the top 5 by Salary",
        height=90,
    )

    ask_col, _ = st.columns([1, 4])
    with ask_col:
        ask_btn = st.button("🚀 Ask AI", use_container_width=True)

    if ask_btn:
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
        elif not query.strip():
            st.warning("⚠️ Please type a question.")
        else:
            with st.spinner("🧠 Gemini is analyzing your data..."):
                try:
                    context = build_data_context(df)
                    result = query_ai(api_key, query, context)
                    st.session_state.last_response = result.get("answer", "")
                    st.session_state.last_table = None

                    if result.get("type") == "table" and result.get("rows"):
                        st.session_state.last_table = pd.DataFrame(result["rows"])

                    if query not in st.session_state.history:
                        st.session_state.history.append(query)

                except Exception as e:
                    err = str(e)
                    if "api" in err.lower() and "key" in err.lower():
                        st.error("❌ Invalid API key. Get a free key at https://aistudio.google.com/app/apikey")
                    else:
                        st.error(f"❌ Error: {err}")

    # Show response
    if st.session_state.last_response:
        st.markdown("### 🤖 AI Response")
        st.markdown(
            f'<div class="response-box">{st.session_state.last_response}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.last_table is not None:
            st.markdown("**📋 Result Table:**")
            st.dataframe(st.session_state.last_table, use_container_width=True)

            csv_buffer = io.StringIO()
            st.session_state.last_table.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download Result as CSV",
                data=csv_buffer.getvalue(),
                file_name="query_result.csv",
                mime="text/csv",
            )

else:
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px; color: #64748b;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📂</div>
        <div style="font-size: 1.3rem; font-weight: 600; color: #94a3b8; margin-bottom: 10px;">No file uploaded yet</div>
        <div style="font-size: 0.9rem;">Upload a CSV or Excel file from the sidebar to get started</div>
    </div>
    """, unsafe_allow_html=True)
