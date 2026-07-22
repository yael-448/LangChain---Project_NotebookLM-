import uuid
import streamlit as st
from langgraph.types import Command
from agent import agent

st.set_page_config(page_title="אספן מקורות חכם", page_icon="📚", layout="centered")

# עיצוב RTL (ימין-לשמאל) לכל העמוד
st.markdown(
    """
    <style>
    .stApp { direction: rtl; text-align: right; }
    textarea, input { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 אספן מקורות חכם")
st.caption("חיקוי קטן ל-NotebookLM — תני נושא, וה-Agent יאסוף מקורות מהאינטרנט")

# --- אתחול הזיכרון של הממשק (session_state) ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_sources" not in st.session_state:
    st.session_state.pending_sources = None
if "round" not in st.session_state:
    st.session_state.round = 0

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- הצגת היסטוריית השיחה ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- מצב 1: ה-Agent עצר ומחכה לבחירת מקורות (HITL) ---
if st.session_state.pending_sources is not None:
    sources = st.session_state.pending_sources
    st.subheader(f"📋 מצאתי {len(sources)} מקורות. בחרי אילו לכלול:")

    with st.form("sources_form"):
        choices = []
        for i, s in enumerate(sources):
            checked = st.checkbox(
                f"**{s.get('title')}**",
                value=True,
                key=f"src_{st.session_state.round}_{i}",
            )
            st.caption(f"🔗 {s.get('url')}")
            st.caption(s.get("description", ""))
            choices.append(checked)
        submitted = st.form_submit_button("✅ אשר והמשך לסיכום")

    if submitted:
        kept = [s for s, c in zip(sources, choices) if c]
        decision = {
            "type": "edit",
            "edited_action": {"name": "present_sources", "args": {"sources": kept}},
        }
        st.session_state.messages.append(
            {"role": "user", "content": f"בחרתי {len(kept)} מקורות מתוך {len(sources)}."}
        )
        with st.spinner("📝 כותב סיכום על בסיס המקורות שבחרת..."):
            result = agent.invoke(Command(resume={"decisions": [decision]}), config)
        summary = result["messages"][-1].text
        st.session_state.messages.append({"role": "assistant", "content": summary})
        st.session_state.pending_sources = None
        st.session_state.round += 1
        st.rerun()

# --- מצב 2: שיחה רגילה — תיבת קלט לנושא ---
else:
    topic = st.chat_input("על איזה נושא לאסוף מקורות?")
    if topic:
        st.session_state.messages.append({"role": "user", "content": topic})
        with st.chat_message("user"):
            st.markdown(topic)
        with st.spinner("🔎 מחפש מקורות באינטרנט..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": topic}]}, config
            )
        interrupt = result.get("__interrupt__")
        if interrupt:
            st.session_state.pending_sources = (
                interrupt[0].value["action_requests"][0]["args"]["sources"]
            )
        else:
            answer = result["messages"][-1].text
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
