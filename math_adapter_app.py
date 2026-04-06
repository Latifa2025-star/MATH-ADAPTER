
import os
import re
import io
import json
import base64
import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(
    page_title="Math Adapter Studio",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Styling
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Fredoka', sans-serif; }
.main .block-container { max-width: 1450px; padding-top: 1rem; padding-bottom: 2rem; }

:root {
  --bg1: #fff7d6;
  --bg2: #e6f7ff;
  --bg3: #f5e6ff;
  --bg4: #e9fff1;
  --card: rgba(255,255,255,0.96);
  --border: rgba(0,0,0,0.06);
  --shadow: 0 16px 40px rgba(0,0,0,0.08);
}

.hero {
  background: radial-gradient(circle at top left, var(--bg1) 0%, var(--bg2) 35%, var(--bg3) 68%, var(--bg4) 100%);
  border: 3px solid rgba(255,255,255,0.95);
  border-radius: 30px;
  padding: 1.4rem 1.6rem;
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 0.4rem 0; font-size: 2.4rem; }
.hero p { margin: 0; color: #3c3c3c; font-size: 1.02rem; }

.kpi {
  background: rgba(255,255,255,0.82);
  border-radius: 20px;
  padding: 0.9rem 1rem;
  border: 1px solid rgba(255,255,255,0.7);
  box-shadow: 0 8px 24px rgba(0,0,0,0.05);
  text-align: center;
}
.kpi .num { font-size: 1.45rem; font-weight: 700; }
.kpi .lab { font-size: 0.9rem; color: #5a5a5a; }

.card {
  background: var(--card);
  border: 2px solid var(--border);
  border-radius: 24px;
  padding: 1rem 1.1rem;
  box-shadow: 0 10px 24px rgba(0,0,0,0.06);
  margin-bottom: 1rem;
}
.badge {
  display: inline-block;
  padding: 0.32rem 0.72rem;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
  margin-bottom: 0.7rem;
}
.badge-orig { background:#e8f3ff; color:#0b3d91; }
.badge-img  { background:#fff0e6; color:#8a3b00; }
.badge-exact{ background:#e9fff1; color:#126a2e; }
.badge-op   { background:#fff7e6; color:#9a5a00; }
.badge-adhd { background:#fff0e6; color:#8a3b00; }
.badge-ell  { background:#e9fff1; color:#126a2e; }
.badge-id   { background:#f3e8ff; color:#5b21b6; }
.badge-sol  { background:#ffe6f2; color:#9b005d; }
.badge-meta { background:#eef2ff; color:#3730a3; }

.bigtext { font-size: 1.23rem; line-height: 1.6; white-space: pre-wrap; }
.smalltext { font-size: 0.96rem; color: #474747; white-space: pre-wrap; }
.eqline { font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
.emojiwrap { font-size: 2rem; line-height: 1.35; word-wrap: break-word; white-space: pre-wrap; }
.muted { color: #666; }

.stButton > button,
.stDownloadButton > button {
  border-radius: 16px !important;
  font-weight: 700 !important;
}

.search-hint {
  font-size: 0.9rem;
  color: #5d5d5d;
  margin-top: -0.3rem;
  margin-bottom: 0.5rem;
}
</style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEXT_MODEL = "gpt-4.1-mini"
IMAGE_MODEL = "gpt-image-1"

# =========================
# Helpers
# =========================
def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.exists() else None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clean_spaces(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON found in model output.")
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0).strip())
        return json.loads(candidate)


def dataset_answer_num(answer: str) -> Optional[int]:
    match = re.search(r"(\d+)", str(answer))
    return int(match.group(1)) if match else None


def answer_object(answer: str) -> str:
    match = re.search(r"\(([^)]+)\)", str(answer))
    if not match:
        return ""
    return normalize_obj(match.group(1))


def image_html_from_b64(b64: str) -> str:
    return f"<img src='data:image/png;base64,{b64}' style='width:100%;border-radius:18px;'/>"


def compact_problem_label(row: pd.Series, idx: int) -> str:
    text = re.sub(r"\s+", " ", str(row["problem_text_full"])).strip()
    preview = text[:120] + ("…" if len(text) > 120 else "")
    return f"#{idx} • {preview}"


# =========================
# Data loaders
# =========================
@st.cache_data(show_spinner=False)
def parse_asdiv_full(xml_bytes: bytes) -> pd.DataFrame:
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
        grade = None
        category = None
        problem_id = None
        for child in item.iter():
            tag = child.tag.lower()
            txt = clean_spaces(child.text)
            if not txt:
                continue
            if tag in ["body", "stem", "text"]:
                body = txt if (body is None or len(txt) > len(body)) else body
            elif tag in ["question", "ques"]:
                question = txt if (question is None or len(txt) > len(question)) else question
            elif tag in ["answer", "ans", "final"]:
                answer = txt if (answer is None or len(txt) > len(answer)) else answer
            elif tag in ["solution", "rationale", "explanation"]:
                solution = txt if (solution is None or len(txt) > len(solution)) else solution
            elif tag in ["grade", "schoolgrade"]:
                grade = txt
            elif tag in ["type", "category"]:
                category = txt
            elif tag in ["id", "problemid"]:
                problem_id = txt
        full = f"{body} {question}" if body and question else (body or question)
        ref = answer or solution
        if full and ref:
            rows.append(
                {
                    "dataset_name": "ASDiv",
                    "problem_id": problem_id or f"ASDiv-{len(rows)+1}",
                    "grade": grade or "",
                    "category": category or "",
                    "problem_text_full": full,
                    "body_text": body or "",
                    "question_text": question or "",
                    "reference_answer": ref,
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def parse_json_dataset(file_bytes: bytes, dataset_name: str, filename: str) -> pd.DataFrame:
    text = file_bytes.decode("utf-8", errors="ignore")
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ["data", "items", "problems", "examples", "records"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{filename}: expected a JSON list or a dict containing a list.")

    problem_keys = ["problem_text_full", "problem", "question", "text", "body", "input", "prompt"]
    answer_keys = ["reference_answer", "answer", "final_answer", "output", "label", "target"]
    grade_keys = ["grade", "level"]
    category_keys = ["category", "type", "operation"]
    id_keys = ["id", "problem_id", "uid"]

    rows = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        problem = next((clean_spaces(item.get(k)) for k in problem_keys if clean_spaces(item.get(k))), "")
        answer = next((clean_spaces(item.get(k)) for k in answer_keys if clean_spaces(item.get(k))), "")
        grade = next((clean_spaces(item.get(k)) for k in grade_keys if clean_spaces(item.get(k))), "")
        category = next((clean_spaces(item.get(k)) for k in category_keys if clean_spaces(item.get(k))), "")
        pid = next((clean_spaces(item.get(k)) for k in id_keys if clean_spaces(item.get(k))), f"{dataset_name}-{i+1}")
        if problem and answer:
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "problem_id": pid,
                    "grade": grade,
                    "category": category,
                    "problem_text_full": problem,
                    "body_text": item.get("body", "") or "",
                    "question_text": item.get("question", "") or "",
                    "reference_answer": answer,
                }
            )
    if not rows:
        raise ValueError(f"{filename}: no usable problem/answer pairs found.")
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_default_datasets(xml_bytes: Optional[bytes]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    if xml_bytes:
        datasets["ASDiv"] = parse_asdiv_full(xml_bytes)
    return datasets


# =========================
# OpenAI client
# =========================
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


def call_text(prompt: str) -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets or environment.")
    response = client.responses.create(model=TEXT_MODEL, input=prompt)
    return response.output_text


def images_generate_b64(prompt: str, size: str = "1024x1024") -> str:
    client = get_client()
    if client is None:
        raise RuntimeError("OpenAI API key not found. Add OPENAI_API_KEY to Streamlit secrets or environment.")
    key = md5(prompt + size)
    cached = CACHE_DIR / f"{key}_storyimg.b64.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    img = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=size)
    b64 = img.data[0].b64_json
    _ = base64.b64decode(b64)
    cached.write_text(b64, encoding="utf-8")
    return b64


# =========================
# Prompts
# =========================
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
- Keep each learner version classroom-friendly and natural.

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


# =========================
# Emoji logic
# =========================
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
    "orange": {"":"🍊"}, "pear": {"":"🍐"}, "banana": {"":"🍌"}, "grape": {"":"🍇"},
    "peach": {"":"🍑"}, "cherry": {"":"🍒"}, "lemon": {"":"🍋"}, "strawberry": {"":"🍓"},
    "watermelon": {"":"🍉"}, "pineapple": {"":"🍍"}, "cookie": {"":"🍪"}, "cake": {"":"🍰"},
    "cupcake": {"":"🧁"}, "donut": {"":"🍩"}, "candy": {"":"🍬"}, "sandwich": {"":"🥪"},
    "egg": {"":"🥚"}, "pencil": {"":"✏️"}, "pen": {"":"🖊️"}, "book": {"":"📚"},
    "notebook": {"":"📓"}, "crayon": {"":"🖍️"}, "eraser": {"":"🧽"}, "ruler": {"":"📏"},
    "ticket": {"":"🎟️"}, "sticker": {"":"⭐"}, "balloon": {"":"🎈"}, "flower": {"":"🌸"},
    "toy": {"":"🧸"}, "doll": {"":"🪆"}, "block": {"":"🧱"}, "ball": {"":"⚽"},
    "marble": {"":"🔵"}, "button": {"":"🔘"}, "box": {"":"📦"}, "basket": {"":"🧺"},
    "bag": {"":"👜"}, "cup": {"":"🥤"}, "bottle": {"":"🍼"}, "sock": {"":"🧦"},
    "shirt": {"":"👕"}, "shoe": {"":"👟"}, "hat": {"":"🧢"}, "coin": {"":"🪙"},
    "money": {"":"💵"}, "dog": {"":"🐶"}, "cat": {"":"🐱"}, "bird": {"":"🐦"},
    "turtle": {"":"🐢"}, "fish": {"":"🐟"}, "duck": {"":"🦆"}, "rabbit": {"":"🐰"},
    "cow": {"":"🐄"}, "pig": {"":"🐷"}, "horse": {"":"🐴"}, "toucan": {"":"🦜"},
    "car": {"":"🚗"}, "bus": {"":"🚌"}, "bike": {"":"🚲"}, "train": {"":"🚆"},
    "truck": {"":"🚚"},
}

NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}


def word_to_num(s: str) -> Optional[int]:
    s = str(s).lower().strip()
    if s.isdigit():
        return int(s)
    return NUM_WORDS.get(s)


def choose_emoji(obj: str, color: str = "") -> str:
    obj = normalize_obj(obj)
    obj = ALIASES.get(obj, obj)
    color = (color or "").lower().strip()
    if obj in EMOJI_MAP:
        return EMOJI_MAP[obj].get(color, EMOJI_MAP[obj].get("", "🟦"))
    return "🟦"


def emoji_grid(emoji: str, n: int, cols: int = 10) -> str:
    parts: List[str] = []
    for i in range(int(n)):
        if i % cols == 0:
            parts.append("\n")
        parts.append(emoji)
        parts.append(" ")
    return "".join(parts).strip()


def extract_local_count_plan(problem: str) -> dict:
    text = problem.lower()
    items = []
    pattern = re.compile(
        r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\s+"
        r"(?:(red|green|blue|yellow|orange|purple|pink|black|white|brown)\s+)?"
        r"([a-zA-Z]+)\b"
    )
    ignore = {
        "minute", "hour", "day", "week", "month", "year", "mile", "meter", "foot", "inch",
        "page", "more", "less", "left", "each", "every", "total", "amount", "group"
    }
    for match in pattern.finditer(text):
        count = word_to_num(match.group(1))
        color = match.group(2) or ""
        obj = normalize_obj(match.group(3))
        obj = ALIASES.get(obj, obj)
        if obj in ignore:
            continue
        if count is not None:
            items.append({"object": obj, "count": count, "color": color})
    return {"items": items}


def build_exact_visual(plan: dict) -> str:
    items = plan.get("items", [])
    if not items:
        return "<div class='smalltext'>No countable objects were detected automatically.</div>"
    html = []
    for item in items:
        obj = item.get("object", "")
        color = item.get("color", "")
        count = int(item.get("count", 0))
        emoji = choose_emoji(obj, color)
        label = f"{count} {color} {obj}(s)".replace("  ", " ").strip()
        html.append(f"<div class='smalltext'><b>✅ {label}</b></div>")
        html.append(f"<div class='emojiwrap'>{emoji_grid(emoji, count, cols=10)}</div>")
    return "\n".join(html)


def build_operation_visual(problem: str, dataset_answer: str, plan: dict) -> str:
    text = problem.lower()
    items = plan.get("items", [])
    final_num = dataset_answer_num(dataset_answer)
    obj = answer_object(dataset_answer) or (items[0]["object"] if items else "item")
    color = items[0].get("color", "") if items else ""

    if final_num is None:
        return "<div class='smalltext'>Could not detect the final answer automatically.</div>"

    def block(label: str, n: int) -> str:
        emo = choose_emoji(obj, color)
        return f"<div class='smalltext'><b>{label}</b></div><div class='emojiwrap'>{emoji_grid(emo, n, cols=10)}</div>"

    nums = [word_to_num(x) for x in re.findall(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b", text)]
    nums = [x for x in nums if x is not None]

    if len(items) >= 2 and any(word in text for word in ["together", "altogether", "in all", "total", "combined"]):
        first = items[0]
        second = items[1]
        a = first["count"]
        b = second["count"]
        emo_a = choose_emoji(first["object"], first.get("color", ""))
        emo_b = choose_emoji(second["object"], second.get("color", ""))
        return f"""
        <div class='smalltext'><b>Group 1</b></div>
        <div class='emojiwrap'>{emoji_grid(emo_a, a, cols=10)}</div>
        <div class='eqline'>+</div>
        <div class='smalltext'><b>Group 2</b></div>
        <div class='emojiwrap'>{emoji_grid(emo_b, b, cols=10)}</div>
        <div class='eqline'>=</div>
        {block(f'{final_num} total {obj}(s)', final_num)}
        """

    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if any(word in text for word in ["left", "remain", "after", "gave away", "lost", "spent", "used", "less", "fewer"]):
            top, sub = max(a, b), min(a, b)
            return f"{block(f'{top} {obj}(s)', top)}<div class='eqline'>−</div>{block(f'{sub} {obj}(s)', sub)}<div class='eqline'>=</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if any(word in text for word in ["each", "every", "times", "double", "triple"]):
            return f"<div class='smalltext'><b>{a} groups of {b}</b></div><div class='eqline'>{a} × {b} = {final_num}</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if any(word in text for word in ["share equally", "equally", "split", "divide", "among"]):
            return f"{block(f'{a} {obj}(s)', a)}<div class='eqline'>÷</div><div class='smalltext'><b>{b} groups</b></div><div class='eqline'>=</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if a + b == final_num:
            return f"{block(f'{a} {obj}(s)', a)}<div class='eqline'>+</div>{block(f'{b} {obj}(s)', b)}<div class='eqline'>=</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if max(a, b) - min(a, b) == final_num:
            top, sub = max(a, b), min(a, b)
            return f"{block(f'{top} {obj}(s)', top)}<div class='eqline'>−</div>{block(f'{sub} {obj}(s)', sub)}<div class='eqline'>=</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if a * b == final_num:
            return f"<div class='eqline'>{a} × {b} = {final_num}</div>{block(f'{final_num} {obj}(s)', final_num)}"
        if b != 0 and a // b == final_num and a % b == 0:
            return f"<div class='eqline'>{a} ÷ {b} = {final_num}</div>{block(f'{final_num} {obj}(s)', final_num)}"

    return "<div class='smalltext'>Could not detect the full emoji operation automatically for this problem.</div>"


# =========================
# Caching bundles
# =========================
def get_bundle(problem: str, dataset_answer: str) -> dict:
    key = md5("bundle::" + problem + "::" + dataset_answer)
    path = CACHE_DIR / f"{key}_bundle.json"
    cached = read_text(path)
    if cached is None:
        cached = call_text(prompt_all_text(problem, dataset_answer))
        write_text(path, cached)
    return extract_json_object(cached)


def get_story_image(problem: str) -> str:
    return images_generate_b64(prompt_story_image(problem), size="1024x1024")


def export_bundle_json(dataset_name: str, row: pd.Series, bundle: dict) -> str:
    payload = {
        "dataset_name": dataset_name,
        "problem_id": row.get("problem_id", ""),
        "grade": row.get("grade", ""),
        "category": row.get("category", ""),
        "problem_text_full": row["problem_text_full"],
        "reference_answer": row["reference_answer"],
        **bundle,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# =========================
# Dataset registry
# =========================
def load_uploaded_datasets(files) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for f in files or []:
        ext = Path(f.name).suffix.lower()
        dataset_name = Path(f.name).stem
        file_bytes = f.read()
        if ext == ".xml":
            datasets[dataset_name] = parse_asdiv_full(file_bytes)
        elif ext == ".json":
            datasets[dataset_name] = parse_json_dataset(file_bytes, dataset_name, f.name)
    return datasets


# =========================
# Sidebar + top area
# =========================
def card_start(badge_class: str, title: str) -> str:
    return f"<div class='card'><div class='badge {badge_class}'>{title}</div>"


def card_end() -> str:
    return "</div>"


render_full(
    bundle=bundle,
    story_img_html=story_img_html,
    exact_html=exact_html,
    operation_html=operation_html,
)
