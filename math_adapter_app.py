
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

st.set_page_config(
    page_title="Math Problem Adapter",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Fredoka', sans-serif; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
.hero { background: radial-gradient(circle at top left, #fff7d6 0%, #e6f7ff 40%, #f5e6ff 75%, #e9fff1 100%);
    border: 3px solid rgba(255,255,255,0.95); border-radius: 28px; padding: 1.5rem 1.7rem;
    box-shadow: 0 14px 30px rgba(0,0,0,0.10); margin-bottom: 1rem; }
.hero h1 { margin: 0 0 0.35rem 0; font-size: 2.4rem; }
.hero p { margin: 0; font-size: 1.05rem; color: #3a3a3a; }
.card { background: rgba(255,255,255,0.98); border-radius: 22px; padding: 1rem 1.1rem;
    box-shadow: 0 10px 18px rgba(0,0,0,0.07); border: 2px solid rgba(0,0,0,0.04); margin-bottom: 1rem; }
.badge { display: inline-block; padding: 0.34rem 0.75rem; border-radius: 999px; font-size: 0.84rem; font-weight: 700; margin-bottom: 0.7rem; }
.badge-orig { background:#e8f3ff; color:#0b3d91; }
.badge-img  { background:#fff0e6; color:#8a3b00; }
.badge-exact{ background:#e9fff1; color:#126a2e; }
.badge-op   { background:#fff7e6; color:#9a5a00; }
.badge-adhd { background:#fff0e6; color:#8a3b00; }
.badge-ell  { background:#e9fff1; color:#126a2e; }
.badge-id   { background:#f3e8ff; color:#5b21b6; }
.badge-sol  { background:#ffe6f2; color:#9b005d; }
.bigtext { font-size: 1.35rem; line-height: 1.55; }
.smalltext { font-size: 0.98rem; color: #404040; }
.eqline { font-size: 2.1rem; font-weight: 700; margin: 0.35rem 0; }
.emojiwrap { font-size: 2.2rem; line-height: 1.35; word-wrap: break-word; }
.stButton>button { border-radius: 14px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def read_text(path: Path):
    return path.read_text(encoding="utf-8") if path.exists() else None

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON found in model output.")
        cand = re.sub(r",\s*([}\]])", r"\1", m.group(0).strip())
        return json.loads(cand)

def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def dataset_answer_num(answer: str):
    m = re.search(r"(\d+)", str(answer))
    return int(m.group(1)) if m else None

@st.cache_data(show_spinner=False)
def parse_asdiv_full(xml_bytes: bytes) -> pd.DataFrame:
    import io
    tree = ET.parse(io.BytesIO(xml_bytes))
    root = tree.getroot()
    rows = []
    for item in root.iter():
        if item.tag.lower() not in ["problem", "item"]:
            continue
        body = None
        question = None
        answer = None
        solution = None
        for child in item.iter():
            tag = child.tag.lower()
            txt = clean_spaces(child.text)
            if not txt:
                continue
            if tag in ["body", "stem", "text"]:
                body = txt if (body is None or len(txt) > len(body)) else body
            if tag in ["question", "ques"]:
                question = txt if (question is None or len(txt) > len(question)) else question
            if tag in ["answer", "ans", "final"]:
                answer = txt if (answer is None or len(txt) > len(answer)) else answer
            if tag in ["solution", "rationale", "explanation"]:
                solution = txt if (solution is None or len(txt) > len(solution)) else solution
        full = f"{body} {question}" if body and question else (body or question)
        ref = answer or solution
        if full and ref:
            rows.append({
                "split": "asdiv",
                "problem_text_full": full,
                "body_text": body,
                "question_text": question,
                "reference_answer": ref,
            })
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def get_client():
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

TEXT_MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "gpt-image-1"

def call_text(prompt: str) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets or environment.")
    resp = client.responses.create(model=TEXT_MODEL, input=prompt)
    return resp.output_text

def images_generate_b64(prompt: str, size="1024x1024") -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets or environment.")
    key = md5(prompt + size)
    p = CACHE_DIR / f"{key}_storyimg.b64.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    img = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=size)
    b64 = img.data[0].b64_json
    _ = base64.b64decode(b64)
    p.write_text(b64, encoding="utf-8")
    return b64

def prompt_all_text(problem: str, dataset_answer: str) -> str:
    return f'''
Return ONLY valid JSON.

You are adapting a Grade 1–2 math word problem for three learner profiles:
ADHD, ELL, and Intellectual Disability (ID).

Rules:
- Keep all numbers exactly the same.
- Keep the math meaning exactly the same.
- Do not add new information.
- The final answer must stay: {dataset_answer}

Return this exact JSON schema:
{{
  "ADHD_problem": "...",
  "ELL_problem": "...",
  "ID_problem": "...",
  "teacher_solution": "...",
  "ADHD_expl": "...",
  "ELL_expl": "...",
  "ID_expl": "..."
}}

Problem:
{problem}

Dataset answer:
{dataset_answer}
'''.strip()

def prompt_story_image(problem: str) -> str:
    return f'''
Create a beautiful, colorful children's workbook illustration for this Grade 1–2 word problem.
Use a warm classroom/storybook style.
No text in image.
Simple scene showing the story objects.
Do not show written numbers.

Problem:
{problem}
'''.strip()

def normalize_obj(word: str) -> str:
    w = (word or "").lower().strip()
    w = re.sub(r"[^a-z\s\-]", "", w)
    w = w.replace("-", " ")
    w = re.sub(r"\s+", " ", w).strip()
    if w.endswith("ies") and len(w) > 3:
        w = w[:-3] + "y"
    elif w.endswith("ves") and len(w) > 3:
        w = w[:-3] + "f"
    elif w.endswith("s") and len(w) > 2 and not w.endswith("ss"):
        w = w[:-1]
    return w

ALIASES = {
    "apples":"apple", "oranges":"orange", "pears":"pear", "bananas":"banana", "grapes":"grape",
    "peaches":"peach", "cherries":"cherry", "lemons":"lemon", "strawberries":"strawberry",
    "watermelons":"watermelon", "pineapples":"pineapple", "cookies":"cookie", "candies":"candy",
    "donuts":"donut", "cupcakes":"cupcake", "sandwiches":"sandwich", "eggs":"egg",
    "pencils":"pencil", "pens":"pen", "books":"book", "notebooks":"notebook", "crayons":"crayon",
    "erasers":"eraser", "rulers":"ruler", "tickets":"ticket", "stickers":"sticker",
    "balloons":"balloon", "flowers":"flower", "toys":"toy", "dolls":"doll", "blocks":"block",
    "balls":"ball", "marbles":"marble", "buttons":"button", "boxes":"box", "baskets":"basket",
    "bags":"bag", "cups":"cup", "bottles":"bottle", "coins":"coin", "dollars":"money",
    "pennies":"coin", "nickels":"coin", "dimes":"coin", "quarters":"coin", "dogs":"dog",
    "cats":"cat", "birds":"bird", "turtles":"turtle", "ducks":"duck", "rabbits":"rabbit",
    "cows":"cow", "pigs":"pig", "horses":"horse", "toucans":"toucan", "cars":"car",
    "buses":"bus", "bikes":"bike", "trains":"train", "trucks":"truck",
}

EMOJI_MAP = {
    "apple": {"red":"🍎", "green":"🍏", "":"🍎"},
    "orange": {"":"🍊"},
    "pear": {"":"🍐"},
    "banana": {"":"🍌"},
    "grape": {"":"🍇"},
    "peach": {"":"🍑"},
    "cherry": {"":"🍒"},
    "lemon": {"":"🍋"},
    "strawberry": {"":"🍓"},
    "watermelon": {"":"🍉"},
    "pineapple": {"":"🍍"},
    "cookie": {"":"🍪"},
    "cake": {"":"🍰"},
    "cupcake": {"":"🧁"},
    "donut": {"":"🍩"},
    "candy": {"":"🍬"},
    "sandwich": {"":"🥪"},
    "egg": {"":"🥚"},
    "pencil": {"":"✏️"},
    "pen": {"":"🖊️"},
    "book": {"":"📚"},
    "notebook": {"":"📓"},
    "crayon": {"":"🖍️"},
    "eraser": {"":"🧽"},
    "ruler": {"":"📏"},
    "ticket": {"":"🎟️"},
    "sticker": {"":"⭐"},
    "balloon": {"":"🎈"},
    "flower": {"":"🌸"},
    "toy": {"":"🧸"},
    "doll": {"":"🪆"},
    "block": {"":"🧱"},
    "ball": {"":"⚽"},
    "marble": {"":"🔵"},
    "button": {"":"🔘"},
    "box": {"":"📦"},
    "basket": {"":"🧺"},
    "bag": {"":"👜"},
    "cup": {"":"🥤"},
    "bottle": {"":"🍼"},
    "sock": {"":"🧦"},
    "shirt": {"":"👕"},
    "shoe": {"":"👟"},
    "hat": {"":"🧢"},
    "coin": {"":"🪙"},
    "money": {"":"💵"},
    "dog": {"":"🐶"},
    "cat": {"":"🐱"},
    "bird": {"":"🐦"},
    "turtle": {"":"🐢"},
    "fish": {"":"🐟"},
    "duck": {"":"🦆"},
    "rabbit": {"":"🐰"},
    "cow": {"":"🐄"},
    "pig": {"":"🐷"},
    "horse": {"":"🐴"},
    "toucan": {"":"🦜"},
    "car": {"":"🚗"},
    "bus": {"":"🚌"},
    "bike": {"":"🚲"},
    "train": {"":"🚆"},
    "truck": {"":"🚚"},
}

def choose_emoji(obj: str, color: str = "") -> str:
    obj = normalize_obj(obj)
    obj = ALIASES.get(obj, obj)
    color = (color or "").lower().strip()
    if obj in EMOJI_MAP:
        return EMOJI_MAP[obj].get(color, EMOJI_MAP[obj].get("", "🟦"))
    return "🟦"

def emoji_grid(emoji: str, n: int, cols: int = 10):
    parts = []
    for i in range(int(n)):
        if i % cols == 0:
            parts.append("\n")
        parts.append(emoji)
        parts.append(" ")
    return "".join(parts).strip()

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50
}

def word_to_num(s):
    s = str(s).lower().strip()
    if s.isdigit():
        return int(s)
    return NUM_WORDS.get(s)

def extract_local_count_plan(problem: str):
    text = problem.lower()
    items = []
    pattern = re.compile(
        r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\s+"
        r"(?:(red|green|blue|yellow|orange|purple|pink|black|white|brown)\s+)?"
        r"([a-zA-Z]+)\b"
    )
    for m in pattern.finditer(text):
        count = word_to_num(m.group(1))
        color = m.group(2) or ""
        obj = normalize_obj(m.group(3))
        obj = ALIASES.get(obj, obj)
        if obj in {"minute","hour","day","week","month","year","mile","meter","foot","inch","page","more","less","left","each","every","total","amount","group"}:
            continue
        if count is not None:
            items.append({"object": obj, "count": count, "color": color})
    return {"items": items}

def build_exact_visual(plan: dict) -> str:
    items = plan.get("items", [])
    if not items:
        return "<div class='smalltext'>No countable objects were detected automatically.</div>"
    html = []
    for it in items:
        obj = it.get("object","")
        color = it.get("color","")
        count = int(it.get("count", 0))
        emo = choose_emoji(obj, color)
        label = f"{count} {color} {obj}(s)".replace("  ", " ").strip()
        html.append(f"<div class='smalltext'><b>✅ {label}</b></div>")
        html.append(f"<div class='emojiwrap'>{emoji_grid(emo, count, cols=10)}</div>")
    return "\n".join(html)

def get_answer_object(dataset_answer: str):
    m = re.search(r"\(([^)]+)\)", str(dataset_answer))
    if not m:
        return ""
    obj = normalize_obj(m.group(1))
    obj = ALIASES.get(obj, obj)
    return obj

def build_operation_visual(problem: str, dataset_answer: str, plan: dict) -> str:
    text = problem.lower()
    items = plan.get("items", [])
    final_num = dataset_answer_num(dataset_answer)
    answer_obj = get_answer_object(dataset_answer)

    if final_num is None:
        return "<div class='smalltext'>Could not detect the final number automatically.</div>"

    def block(obj, color, n, label):
        emo = choose_emoji(obj, color)
        return f"<div class='smalltext'><b>{label}</b></div><div class='emojiwrap'>{emoji_grid(emo, n, cols=10)}</div>"

    if len(items) >= 2 and any(w in text for w in ["together", "altogether", "in all", "total", "sum", "basket", "together have"]):
        a, b = items[0], items[1]
        out_obj = answer_obj or a["object"]
        return f'''
        {block(a["object"], a.get("color",""), a["count"], f'{a["count"]} {a.get("color","")} {a["object"]}(s)')}
        <div class="eqline">+</div>
        {block(b["object"], b.get("color",""), b["count"], f'{b["count"]} {b.get("color","")} {b["object"]}(s)')}
        <div class="eqline">=</div>
        {block(out_obj, "", final_num, f'{final_num} {out_obj}(s)')}
        '''

    if any(w in text for w in ["fewer", "less", "left", "remain", "after", "gave away", "lost", "spent", "used", "minus"]):
        nums = [word_to_num(x) for x in re.findall(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b", text)]
        nums = [x for x in nums if x is not None]
        if len(nums) >= 2:
            a, b = max(nums[0], nums[1]), min(nums[0], nums[1])
            obj = answer_obj or (items[0]["object"] if items else "item")
            color = items[0].get("color","") if items else ""
            return f'''
            {block(obj, color, a, f'{a} {obj}(s)')}
            <div class="eqline">−</div>
            {block(obj, color, b, f'{b} {obj}(s)')}
            <div class="eqline">=</div>
            {block(obj, color, final_num, f'{final_num} {obj}(s)')}
            '''

    if any(w in text for w in ["each", "every", "times", "twice", "double", "triple"]):
        nums = [word_to_num(x) for x in re.findall(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b", text)]
        nums = [x for x in nums if x is not None]
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            obj = answer_obj or (items[0]["object"] if items else "item")
            color = items[0].get("color","") if items else ""
            return f'''
            {block(obj, color, a, f'{a} group(s)')}
            <div class="eqline">×</div>
            {block(obj, color, b, f'{b} each')}
            <div class="eqline">=</div>
            {block(obj, color, final_num, f'{final_num} {obj}(s)')}
            '''

    if any(w in text for w in ["split", "share equally", "equally", "divide", "among", "per"]):
        nums = [word_to_num(x) for x in re.findall(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b", text)]
        nums = [x for x in nums if x is not None]
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            obj = answer_obj or (items[0]["object"] if items else "item")
            color = items[0].get("color","") if items else ""
            return f'''
            {block(obj, color, a, f'{a} {obj}(s)')}
            <div class="eqline">÷</div>
            <div class='smalltext'><b>{b} groups</b></div>
            <div class="eqline">=</div>
            {block(obj, color, final_num, f'{final_num} {obj}(s)')}
            '''

    return "<div class='smalltext'>Could not detect the full emoji operation automatically for this problem.</div>"

def get_bundle(problem: str, dataset_answer: str):
    k = md5("bundle::" + problem + "::" + dataset_answer)
    p = CACHE_DIR / f"{k}_bundle.json"
    txt = read_text(p)
    if txt is None:
        txt = call_text(prompt_all_text(problem, dataset_answer))
        write_text(p, txt)
    return extract_json_object(txt)

def get_story_image(problem: str):
    return images_generate_b64(prompt_story_image(problem), size="1024x1024")

st.sidebar.markdown("## 🎛️ Controls")
mode = st.sidebar.radio("Choose experience", ["Adapter only", "Adapter + emojis + illustration"], index=0)
uploaded_xml = st.sidebar.file_uploader("Upload ASDiv.xml", type=["xml"])
query = st.sidebar.text_input("Search by word", placeholder="apple, peach, ticket...")
only_with_color = st.sidebar.checkbox("Show problems with color words", value=False)
shuffle_btn = st.sidebar.button("🎲 Surprise me")
st.sidebar.info("Adapter only is fastest. Illustration mode adds the image and emoji visuals.")

client_ok = get_client() is not None
if client_ok:
    st.sidebar.success("OpenAI key detected")
else:
    st.sidebar.warning("No OpenAI key detected")
default_xml_path = APP_DIR / "ASDiv.xml"
xml_bytes = None
if uploaded_xml is not None:
    xml_bytes = uploaded_xml.read()
elif default_xml_path.exists():
    xml_bytes = default_xml_path.read_bytes()

if xml_bytes is None:
    st.markdown("<div class='hero'><h1>🌈 Math Problem Adapter</h1><p>Upload <b>ASDiv.xml</b> in the sidebar to begin.</p></div>", unsafe_allow_html=True)
    st.stop()

asdiv_full = parse_asdiv_full(xml_bytes)
df = asdiv_full.copy()

if query.strip():
    q = query.strip().lower()
    df = df[df["problem_text_full"].str.lower().str.contains(q, na=False)]
if only_with_color:
    df = df[df["problem_text_full"].str.lower().str.contains(r"\b(red|green|blue|yellow|orange|purple|pink|brown|black|white)\b", regex=True, na=False)]

if len(df) == 0:
    st.error("No problems matched your filters.")
    st.stop()

if shuffle_btn:
    st.session_state["picked_index"] = random.randint(0, len(df)-1)

if "picked_index" not in st.session_state or st.session_state["picked_index"] >= len(df):
    st.session_state["picked_index"] = 0

st.markdown("""
<div class="hero">
  <h1>🌈 Math Problem Adapter (Grades 1–2)</h1>
  <p>Beautiful conference-ready demo for accessibility-focused math adaptation. Pick a problem, generate learner-friendly rewrites, and show emoji math visuals with optional illustration.</p>
</div>
""", unsafe_allow_html=True)

problem_labels = [f"{i} — {re.sub(r'\s+', ' ', str(t)).strip()}" for i, t in enumerate(df["problem_text_full"].tolist())]
selected_label = st.selectbox("📘 Pick a problem", problem_labels, index=st.session_state["picked_index"])
selected_idx = int(selected_label.split(" — ")[0])
st.session_state["picked_index"] = selected_idx

row = df.iloc[selected_idx]
problem = row["problem_text_full"]
dataset_answer = row["reference_answer"]

generate = st.button("✨ Generate Conference View", type="primary", use_container_width=True)

def card_start(badge_class: str, title: str):
    return f"<div class='card'><div class='badge {badge_class}'>{title}</div>"

def card_end():
    return "</div>"

def image_html_from_b64(b64: str):
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;'/>"

def render_full(bundle=None, story_img_html="", exact_html="", operation_html=""):
    st.markdown(card_start("badge-orig", "📖 Original Problem (Full)") + f"<div class='bigtext'>{problem}</div>" + f"<div class='smalltext'><b>Correct answer (dataset):</b> {dataset_answer}</div>" + card_end(), unsafe_allow_html=True)

    if mode == "Adapter + emojis + illustration":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(card_start("badge-img", "🖼️ Story Illustration") + story_img_html + card_end(), unsafe_allow_html=True)
        with c2:
            st.markdown(card_start("badge-exact", "🎯 Exact Count Visual") + exact_html + card_end(), unsafe_allow_html=True)

        st.markdown(card_start("badge-op", "🧮 Emoji Math Operation") + operation_html + card_end(), unsafe_allow_html=True)

    if bundle:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(card_start("badge-adhd", "⚡ ADHD Version") + f"<div class='bigtext'>{bundle.get('ADHD_problem','')}</div>" + card_end(), unsafe_allow_html=True)
        with c2:
            st.markdown(card_start("badge-ell", "🟢 ELL Version") + f"<div class='bigtext'>{bundle.get('ELL_problem','')}</div>" + card_end(), unsafe_allow_html=True)
        with c3:
            st.markdown(card_start("badge-id", "🧠 ID Version") + f"<div class='bigtext'>{bundle.get('ID_problem','')}</div>" + card_end(), unsafe_allow_html=True)

        st.markdown(card_start("badge-sol", "👩‍🏫 Teacher Solution") + f"<div class='bigtext'>{bundle.get('teacher_solution','')}</div>" + card_end(), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(card_start("badge-sol", "⚡ ADHD Explanation") + f"<div class='bigtext'>{bundle.get('ADHD_expl','')}</div>" + card_end(), unsafe_allow_html=True)
        with c2:
            st.markdown(card_start("badge-sol", "🟢 ELL Explanation") + f"<div class='bigtext'>{bundle.get('ELL_expl','')}</div>" + card_end(), unsafe_allow_html=True)
        with c3:
            st.markdown(card_start("badge-sol", "🧠 ID Explanation") + f"<div class='bigtext'>{bundle.get('ID_expl','')}</div>" + card_end(), unsafe_allow_html=True)

if not generate:
    preview_plan = extract_local_count_plan(problem)
    preview_exact = build_exact_visual(preview_plan) if mode == "Adapter + emojis + illustration" else ""
    preview_op = build_operation_visual(problem, dataset_answer, preview_plan) if mode == "Adapter + emojis + illustration" else ""
    render_full(bundle=None, story_img_html="<div class='smalltext'>Click Generate to create the illustration.</div>", exact_html=preview_exact, operation_html=preview_op)
    st.stop()

try:
    with st.spinner("Generating learner-friendly output..."):
        bundle = get_bundle(problem, dataset_answer)

    story_img_html = ""
    exact_html = ""
    operation_html = ""

    if mode == "Adapter + emojis + illustration":
        plan = extract_local_count_plan(problem)
        exact_html = build_exact_visual(plan)
        operation_html = build_operation_visual(problem, dataset_answer, plan)

        with st.spinner("Generating story illustration..."):
            try:
                b64 = get_story_image(problem)
                story_img_html = image_html_from_b64(b64)
            except Exception as e:
                story_img_html = f"<div class='smalltext'>Image generation failed: {e}</div>"

    render_full(bundle=bundle, story_img_html=story_img_html, exact_html=exact_html, operation_html=operation_html)

except Exception as e:
    st.error(f"Error: {e}")
