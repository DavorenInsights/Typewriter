
import io
import random
import textwrap
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------
# Fixed page calibration
# ------------------------------------------------------------
DPI = 300
A5_WIDTH_MM = 148
A5_HEIGHT_MM = 210

PAGE_W = round(A5_WIDTH_MM / 25.4 * DPI)
PAGE_H = round(A5_HEIGHT_MM / 25.4 * DPI)

LEFT_MM = 6.0
RIGHT_MM = 6.0
TOP_MM = 11.0
BOTTOM_MM = 12.0

# Elite-style typewriter proportions:
# 12 characters per inch = 2.1167 mm per keystroke.
CHAR_PITCH_MM = 25.4 / 12.0
LINE_PITCH_MM = 8.2

# A 10 pt monospaced scaffold gives a glyph width close to a 12-CPI cell.
FONT_PT = 10.0


def mm_to_px(mm: float) -> int:
    return round(mm / 25.4 * DPI)


def pt_to_px(pt: float) -> int:
    return round(pt / 72.0 * DPI)


LEFT = mm_to_px(LEFT_MM)
RIGHT = mm_to_px(RIGHT_MM)
TOP = mm_to_px(TOP_MM)
BOTTOM = mm_to_px(BOTTOM_MM)

CHAR_PITCH = mm_to_px(CHAR_PITCH_MM)
LINE_PITCH = mm_to_px(LINE_PITCH_MM)
FONT_PX = pt_to_px(FONT_PT)


# ------------------------------------------------------------
# Font
# ------------------------------------------------------------
def load_elite_font():
    """
    Use a standard monospaced system font available on Streamlit Cloud.
    No font upload, no external font files, one fixed Elite-style face.
    """
    candidates = [
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_PX)

    return ImageFont.load_default()


FONT = load_elite_font()


# ------------------------------------------------------------
# Wrapping
# ------------------------------------------------------------
def wrap_fixed_pitch(text: str, max_chars: int):
    """
    Wrap using a fixed typewriter carriage width.
    Explicit new lines and blank lines are preserved.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    out = []

    for raw_line in text.split("\n"):
        if raw_line == "":
            out.append("")
            continue

        wrapped = textwrap.wrap(
            raw_line,
            width=max_chars,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )

        out.extend(wrapped if wrapped else [""])

    return out


# ------------------------------------------------------------
# Rendering
# ------------------------------------------------------------
def ribbon_profile(age: str):
    if age == "Fresh":
        return {
            "base": 238,
            "spread": 10,
            "dark_prob": 0.05,
            "light_prob": 0.02,
        }

    if age == "Used":
        return {
            "base": 205,
            "spread": 24,
            "dark_prob": 0.14,
            "light_prob": 0.10,
        }

    return {
        "base": 160,
        "spread": 34,
        "dark_prob": 0.17,
        "light_prob": 0.18,
    }


def strike_alpha(rng: random.Random, ribbon, imperfection: float) -> int:
    """
    Each keystroke gets its own ink strength.
    Some random letters are distinctly darker, like the uploaded pages.
    """
    spread = ribbon["spread"] * (0.55 + imperfection * 0.85)
    alpha = rng.gauss(ribbon["base"], spread)

    if rng.random() < ribbon["dark_prob"] * (0.7 + imperfection * 0.6):
        alpha += rng.randint(25, 60)

    if rng.random() < ribbon["light_prob"] * (0.7 + imperfection * 0.8):
        alpha -= rng.randint(25, 65)

    return max(70, min(255, int(alpha)))


def draw_character(layer, char, x, y, alpha):
    draw = ImageDraw.Draw(layer)
    draw.text(
        (x, y),
        char,
        font=FONT,
        fill=(47, 45, 42, alpha),
    )


def render_page(lines, page_index, ribbon_age, imperfections, seed):
    rng = random.Random(seed + page_index * 10007)
    imp = imperfections / 100.0
    ribbon = ribbon_profile(ribbon_age)

    # Warm off-white paper.
    page = Image.new("RGB", (PAGE_W, PAGE_H), (248, 247, 241))

    # Very subtle paper grain, intentionally restrained.
    paper_draw = ImageDraw.Draw(page)
    grain_count = int(250 + imp * 350)
    for _ in range(grain_count):
        x = rng.randrange(PAGE_W)
        y = rng.randrange(PAGE_H)
        shade = rng.choice([(244, 243, 237), (251, 250, 245), (240, 239, 233)])
        paper_draw.point((x, y), fill=shade)

    overlay = Image.new("RGBA", page.size, (255, 255, 255, 0))

    for line_no, line in enumerate(lines):
        y = TOP + line_no * LINE_PITCH

        # Whole-line start variation only.
        # No character-by-character vertical wobble.
        line_offset = round(rng.uniform(-0.7, 0.7) * mm_to_px(1.0) * imp)
        x = LEFT + line_offset

        for i, char in enumerate(line):
            alpha = strike_alpha(rng, ribbon, imp)

            # Tiny horizontal registration error only.
            x_jitter = round(rng.uniform(-0.10, 0.10) * CHAR_PITCH * imp)

            if char != " ":
                draw_character(overlay, char, x + x_jitter, y, alpha)

                # Rare horizontal rebound / double strike.
                if rng.random() < 0.012 * imp:
                    ghost_alpha = max(25, int(alpha * 0.25))
                    rebound = rng.choice([-1, 1])
                    draw_character(overlay, char, x + x_jitter + rebound, y, ghost_alpha)

            advance = CHAR_PITCH

            # Occasional larger word space, one of the main human/mechanical quirks.
            if char == " " and rng.random() < 0.045 * imp:
                advance += round(CHAR_PITCH * rng.choice([0.45, 0.70, 1.0]))

            x += advance

    page = Image.alpha_composite(page.convert("RGBA"), overlay).convert("RGB")
    return page


def image_to_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG", dpi=(DPI, DPI))
    return buf.getvalue()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Field Notes Typewriter",
    page_icon="⌨️",
    layout="wide",
)

st.title("Field Notes Typewriter")
st.caption(
    "Elite only · true A5 · fixed type scale · ribbon age + one imperfections control"
)

with st.sidebar:
    st.header("Typewriter")

    st.markdown("**Font:** Elite")

    ribbon_age = st.selectbox(
        "Ribbon age",
        ["Fresh", "Used", "Fading"],
        index=1,
    )

    imperfections = st.slider(
        "Imperfections",
        min_value=0,
        max_value=100,
        value=28,
        step=1,
    )

    seed = st.number_input(
        "Page seed",
        min_value=0,
        max_value=9_999_999,
        value=650,
        step=1,
        help="Change this only if you want a different random pattern.",
    )

st.subheader("Text")

default_text = """WHAT IS DEBT

This is money you borrow and
are expected to pay back.

The lender does not normally own
part of your project. Instead, you agree
how and when the money will be repaid...
usually with interest.

TAKEAWAY
--------
Can you repay the money?
"""

text = st.text_area(
    "Paste your text",
    value=default_text,
    height=360,
    label_visibility="collapsed",
)

usable_width = PAGE_W - LEFT - RIGHT
max_chars = max(1, usable_width // CHAR_PITCH)

usable_height = PAGE_H - TOP - BOTTOM
lines_per_page = max(1, usable_height // LINE_PITCH)

lines = wrap_fixed_pitch(text, max_chars)
pages = [
    lines[i:i + lines_per_page]
    for i in range(0, len(lines), lines_per_page)
] or [[""]]

if st.button("Generate", type="primary", use_container_width=True):
    rendered = [
        render_page(
            page_lines,
            page_index=i,
            ribbon_age=ribbon_age,
            imperfections=imperfections,
            seed=int(seed),
        )
        for i, page_lines in enumerate(pages)
    ]

    st.session_state["pages"] = [image_to_png_bytes(img) for img in rendered]


if "pages" in st.session_state:
    page_bytes = st.session_state["pages"]

    st.success(
        f"{len(page_bytes)} A5 page(s) generated — "
        f"{PAGE_W} × {PAGE_H} px at {DPI} DPI."
    )

    for i, png in enumerate(page_bytes, start=1):
        st.image(png, caption=f"Page {i}", use_container_width=True)

        st.download_button(
            label=f"Download page {i}",
            data=png,
            file_name=f"field_note_{i:02d}.png",
            mime="image/png",
            key=f"download_{i}",
        )
