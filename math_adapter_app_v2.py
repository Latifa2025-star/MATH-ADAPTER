
# ============================================================
# Math Adapter Web App (Improved Version)
# ============================================================
# Features
# - ASDiv dataset loader
# - ADHD / ELL / ID adapted problems
# - Emoji math visualizations
# - Robust operation detection (always shows visual)
# - Illustration generation (optional)
# - Professional UI labels
# ============================================================

import os
import re
import json
import base64
import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Math Problem Adapter",
    page_icon="🧮",
    layout="wide"
)


# ============================================================
# STYLES
# ============================================================
st.markdown("""
<style>

.hero {
background: linear-gradient(90deg,#ffeaa7,#dfe6e9,#c7ecee);
padding: 1.5rem;
border-radius:20px;
margin-bottom:20px;
}

.bigtext {font-size:1.1rem; line-height:1.6}
.emojiwrap {font-size:2rem}

.generate-btn button {
background:#ff4757;
color:white;
font-weight:700;
border-radius:12px;
height:3rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHE
# ============================================================
APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


# ============================================================
# DATASET PARSER
# ============================================================
@st.cache_data
def parse_asdiv(xml_bytes):

    import io

    tree = ET.parse(io.BytesIO(xml_bytes))
    root = tree.getroot()

    rows = []

    for item in root.iter():

        if item.tag.lower() not in ["problem","item"]:
            continue

        body=None
        question=None
        answer=None

        for child in item.iter():

            txt=(child.text or "").strip()

            tag=child.tag.lower()

            if tag in ["body","stem","text"]:
                body=txt

            if tag in ["question","ques"]:
                question=txt

            if tag in ["answer","final"]:
                answer=txt

        full=(body or "")+" "+(question or "")

        if full and answer:

            rows.append({
                "problem":full.strip(),
                "answer":answer
            })

    return pd.DataFrame(rows)


# ============================================================
# OPENAI
# ============================================================
@st.cache_resource
def get_client():

    key = os.environ.get("OPENAI_API_KEY")

    if not key or OpenAI is None:
        return None

    return OpenAI(api_key=key)


def call_llm(prompt):

    client=get_client()

    if client is None:
        return {}

    r=client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    txt=r.output_text

    try:
        return json.loads(txt)
    except:
        return {}


# ============================================================
# PROMPT
# ============================================================
def build_prompt(problem,answer):

    return f"""

Return JSON.

Rewrite this math problem for:

ADHD
ELL
ID

Do not change numbers.

Problem:
{problem}

Answer:
{answer}

JSON:

{{
"ADHD_problem":"",
"ELL_problem":"",
"ID_problem":"",
"teacher_solution":""
}}

"""


# ============================================================
# EMOJI
# ============================================================
EMOJI={
"apple":"🍎",
"banana":"🍌",
"orange":"🍊",
"pear":"🍐",
"cookie":"🍪",
"ball":"⚽",
"car":"🚗",
"dog":"🐶",
"cat":"🐱"
}


def detect_object(text):

    for k in EMOJI:
        if k in text.lower():
            return k

    return "apple"


def detect_numbers(text):

    nums=re.findall(r'\d+',text)

    return [int(x) for x in nums]


def emoji_grid(emoji,n):

    s=""

    for i in range(n):

        s+=emoji+" "

    return s


# ============================================================
# OPERATION DETECTION (ROBUST)
# ============================================================
def build_operation(problem,answer):

    nums=detect_numbers(problem)

    if len(nums)<2:
        return "🧠 Unable to detect numbers"

    a,b=nums[0],nums[1]

    ans=int(re.findall(r'\d+',answer)[0])

    obj=detect_object(problem)

    emoji=EMOJI.get(obj,"🍎")

    # detect operation
    if a+b==ans:
        op="+"
    elif a-b==ans:
        op="-"
    elif a*b==ans:
        op="×"
    elif b!=0 and a/b==ans:
        op="÷"
    else:
        op="+"

    return f"""

{emoji_grid(emoji,a)}

{op}

{emoji_grid(emoji,b)}

=

{emoji_grid(emoji,ans)}

"""


# ============================================================
# IMAGE
# ============================================================
def generate_image(problem):

    client=get_client()

    if client is None:
        return ""

    img=client.images.generate(
        model="gpt-image-1",
        prompt="children math book illustration "+problem,
        size="1024x1024"
    )

    return img.data[0].b64_json


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Controls")

mode=st.sidebar.radio(
"Mode",
["Adapter only","Adapter + Emojis + Illustration"]
)

file=st.sidebar.file_uploader("Upload ASDiv.xml",type=["xml"])

search=st.sidebar.text_input("Search problem")

shuffle=st.sidebar.button("Random Problem")


# ============================================================
# LOAD DATA
# ============================================================
if file is None:

    st.markdown("""
    <div class="hero">
    <h2>Upload ASDiv.xml to start</h2>
    </div>
    """,unsafe_allow_html=True)

    st.stop()


df=parse_asdiv(file.read())


if search:
    df=df[df.problem.str.contains(search,case=False)]


if shuffle:
    idx=random.randint(0,len(df)-1)
else:
    idx=0


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
<h1>🧮 Math Problem Adapter</h1>
Generate learner‑friendly math problems with emoji visuals.
</div>
""",unsafe_allow_html=True)


labels=[f"{i} — {p[:120]}" for i,p in enumerate(df.problem)]

sel=st.selectbox("Choose a problem",labels)

i=int(sel.split(" — ")[0])

problem=df.problem.iloc[i]
answer=df.answer.iloc[i]


# ============================================================
# GENERATE BUTTON
# ============================================================
generate=st.button("✨ Generate Adapted Problem",use_container_width=True)


# ============================================================
# DISPLAY
# ============================================================
st.subheader("Original Problem")

st.write(problem)

st.write("Correct Answer:",answer)


if generate:

    with st.spinner("Generating adaptations..."):

        bundle=call_llm(build_prompt(problem,answer))

    st.divider()

    col1,col2,col3=st.columns(3)

    col1.subheader("ADHD")
    col1.write(bundle.get("ADHD_problem",""))

    col2.subheader("ELL")
    col2.write(bundle.get("ELL_problem",""))

    col3.subheader("ID")
    col3.write(bundle.get("ID_problem",""))

    st.divider()

    st.subheader("Teacher Solution")

    st.write(bundle.get("teacher_solution",""))

    if mode=="Adapter + Emojis + Illustration":

        st.divider()

        st.subheader("Emoji Operation Visual")

        st.markdown(
        f"<div class='emojiwrap'>{build_operation(problem,answer)}</div>",
        unsafe_allow_html=True)

        st.divider()

        st.subheader("Illustration")

        try:

            b64=generate_image(problem)

            st.image(base64.b64decode(b64))

        except:

            st.write("Image generation unavailable")
