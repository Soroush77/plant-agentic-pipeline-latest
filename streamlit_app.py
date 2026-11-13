"""Chat UI for the plant diagnosis agent.

  streamlit run streamlit_app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

import plant_agent as pa

st.set_page_config(page_title="دستیار گیاه‌پزشک", page_icon=":material/potted_plant:", layout="centered")
# Persian UI: Streamlit has no right-to-left setting, so these CSS rules are needed (fonts are in .streamlit/config.toml).
st.html("""<style>
.stApp { direction: rtl; }
.stApp [data-testid="stElementContainer"], .stApp ul, .stApp ol { text-align: start; }
.stApp [data-testid="stExpanderStepConnector"] { left: auto; right: 11.5px; }  /* step timeline line sits under the icons */
</style>""")

CATEGORY = {
    "healthy": ("سالم", "green", ":material/check_circle:"),
    "pest": ("آفت", "orange", ":material/bug_report:"),
    "disease": ("بیماری", "red", ":material/coronavirus:"),
    "abiotic": ("مشکل نگهداری", "blue", ":material/water_drop:"),
    "uncertain": ("نامشخص", "gray", ":material/help:"),
}
TOOL = {
    "search_pest_book": ("جستجو در راهنمای آفات", ":material/search:"),
    "read_pest_entry": ("خواندن مدخل راهنما", ":material/menu_book:"),
    "view_pest_photos": ("مقایسه با عکس‌های مرجع آفت", ":material/compare:"),
    "list_conditions": ("فهرست بیماری‌های این گیاه", ":material/list:"),
    "find_reference_classes": ("جستجو در کلاس‌های مرجع", ":material/category:"),
    "view_condition_photos": ("مقایسه با عکس‌های مرجع بیماری", ":material/compare:"),
    "ask_user": ("پرسش از شما", ":material/forum:"),
    "submit_diagnosis": ("ثبت تشخیص", ":material/task_alt:"),
}


@st.cache_resource(show_spinner="در حال بارگذاری پایگاه دانش…")
def load_kb():
    return pa.PestKB(), pa.DiseaseKB(pa.DISEASE_SETS)


def new_conversation():
    return pa.Conversation(pa.Session(pest, dis), pause_on_ask=True)


def render_step(e: dict) -> None:
    """One tool call as a timeline step (used live while the agent runs, and when replaying history)."""
    label, icon = TOOL.get(e["tool"], (e["tool"], ":material/build:"))
    a = e["args"] if isinstance(e["args"], dict) else {}
    detail = a.get("query") or a.get("entry_id") or " · ".join(str(a[k]) for k in ("plant", "condition") if k in a) or a.get("question") or a.get("name") or ""
    with st.status(f"{icon} {label}", type="step", state="complete"):
        if detail:
            st.caption(detail)
        if e["photos"]:
            st.image(e["photos"], caption=e["captions"], width=150)
        elif e["tool"] == "submit_diagnosis" and "Not accepted" in e["result"]:
            st.caption("رد شد: پیش از تشخیص قطعی باید با عکس مرجع مقایسه شود.")


def bullet_list(title: str, items: list) -> None:
    if items:
        st.markdown(f"**{title}**\n" + "\n".join(f"- {x}" for x in items))


def render_diagnosis(d: dict, user_images: list, steps: list) -> None:
    label, color, icon = CATEGORY.get(d["category"], CATEGORY["uncertain"])
    with st.container(border=True):
        st.badge(label, icon=icon, color=color)
        sci = f" · *{d['scientific_name']}*" if d.get("scientific_name") else ""
        st.subheader(f"{d['name']}{sci}")
        with st.container(horizontal=True):
            st.metric("گیاه میزبان", d["host_plant"])
            st.metric("اطمینان", f"{float(d['confidence']):.0%}")
        st.markdown(d["summary"])
        refs = [(p, c) for e in steps if e["tool"].startswith("view_") for p, c in zip(e["photos"], e["captions"])]
        if refs:
            st.markdown("**عکس شما در کنار مرجع‌های مقایسه‌شده**")
            st.image(user_images[:1] + [p for p, _ in refs], caption=["عکس شما"][: len(user_images[:1])] + [c for _, c in refs], width=150)
        bullet_list("شواهد", d.get("evidence"))
        bullet_list("برای اطمینان بیشتر بررسی کنید", d.get("next_checks"))
        bullet_list("چه کار کنید", d.get("actions"))
        if d.get("differentials"):
            with st.expander("احتمال‌های دیگر", icon=":material/alt_route:"):
                for x in d["differentials"]:
                    st.markdown(f"**{x['name']}**: {x['why_less_likely']}")


def render_turn(t: dict) -> None:
    if t["role"] == "user":
        if t["images"]:
            st.image(t["images"], width=180)
        if t["text"]:
            st.markdown(t["text"])
        return
    if t["steps"]:
        with st.status(f"{len(t['steps'])} گام بررسی", type="compact", state="complete"):
            for e in t["steps"]:
                render_step(e)
    if t.get("diagnosis"):
        render_diagnosis(t["diagnosis"], t["user_images"], t["case_steps"])
    elif t.get("question"):
        st.markdown(f":material/forum: **پرسش:** {t['question']}")
    elif t.get("text"):
        st.markdown(t["text"])


try:
    pest, dis = load_kb()
except FileNotFoundError as e:
    st.error(f"پایگاه دانش پیدا نشد. درایو داده وصل است؟\n\n`{e}`", icon=":material/error:")
    st.stop()

ss = st.session_state
ss.setdefault("turns", [])
ss.setdefault("photos", [])  # the user's most recent photos, shown next to the references
ss.setdefault("tmp", tempfile.mkdtemp(prefix="plant_ui_"))
ss.setdefault("case_start", 0)  # trace index where the current diagnosis began (it can span several turns when the agent asks)
if "conv" not in ss:
    ss.conv = new_conversation()

with st.sidebar:
    if st.button("گفتگوی جدید", icon=":material/add:", width="stretch"):
        ss.turns, ss.photos, ss.conv, ss.case_start = [], [], new_conversation(), 0
    ss.conv.s.verified_only = st.toggle("فقط مرجع‌های تأییدشده", help="عکس‌های مرجعِ جمع‌آوری‌شده از وب که برچسبشان تأیید نشده کنار گذاشته می‌شوند.")
    st.caption(f"مدل: `{pa.MODEL}`")
    st.caption(f"{len(pest.chunks)} مدخل آفت · {len(dis.items)} کلاس بیماری روی {len(dis.plants)} گیاه")
    if not os.environ.get("OPENROUTER_API_KEY"):
        st.warning("کلید OpenRouter تنظیم نشده است. آن را در فایل `.env` کنار برنامه بگذارید.", icon=":material/key:")

st.title("دستیار گیاه‌پزشک")
st.caption("عکس گیاه را بفرستید. دستیار آن را با راهنمای آفات و عکس‌های مرجع مقایسه می‌کند و اگر لازم باشد از شما سؤال می‌پرسد.")

for t in ss.turns:
    with st.chat_message(t["role"]):
        render_turn(t)

prompt = st.chat_input("عکس گیاه را پیوست کنید و مشکل را بنویسید", accept_file="multiple",
                       file_type=["jpg", "jpeg", "png", "webp"], submit_mode="disable")
if prompt:
    paths = []
    for f in prompt.files:
        p = Path(ss.tmp) / f"{len(ss.turns)}_{Path(f.name).name}"
        p.write_bytes(f.getvalue())
        paths.append(str(p))
    if paths:
        ss.photos = paths
    user_turn = {"role": "user", "text": prompt.text or "", "images": paths}
    ss.turns.append(user_turn)
    with st.chat_message("user"):
        render_turn(user_turn)

    with st.chat_message("assistant"):
        conv, first = ss.conv, len(ss.conv.s.trace)
        out = error = None
        with st.status(":shimmer[در حال بررسی…]", type="compact") as status:
            conv.s.on_tool = render_step  # each finished tool call appears as a step while the agent works
            try:
                out = conv.send(prompt.text or "", paths)
            except RuntimeError as e:  # missing key or API failure
                error = str(e)
            conv.s.on_tool = None
            steps = conv.s.trace[first:]
            label = "خطا" if error else f"{len(steps)} گام بررسی" if steps else "پاسخ بدون نیاز به ابزار"
            status.update(label=label, state="error" if error else "complete")
        if error:
            st.error(error, icon=":material/error:")
        else:
            turn = {"role": "assistant", "steps": steps, "case_steps": conv.s.trace[ss.case_start:], "user_images": ss.photos,
                    "diagnosis": out["diagnosis"], "question": out["question"], "text": out["text"]}
            ss.turns.append(turn)
            if turn["diagnosis"]:
                ss.case_start = len(conv.s.trace)
                render_diagnosis(turn["diagnosis"], turn["user_images"], turn["case_steps"])
            elif turn["question"]:
                st.markdown(f":material/forum: **پرسش:** {turn['question']}")
            elif turn["text"]:
                st.markdown(turn["text"])
