
import io
import math
import random
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageChops
import numpy as np


# -----------------------------
# Page / rendering configuration
# -----------------------------
A5_MM = (148, 210)

PRESETS = {
    "Field Notes (balanced)": {
        "jitter_x": 0.28,
        "jitter_y": 0.10,
        "rotation": 0.12,
        "opacity_min": 155,
        "opacity_max": 235,
        "double_strike_prob": 0.035,
        "double_strike_offset": 0.8,
        "dropout_prob": 0.020,
        "char_spacing_jitter": 0.10,
        "baseline_wander": 0.05,
        "paper_noise": 5.0,
        "blur": 0.16,
    },
    "Fresh ribbon": {
        "jitter_x": 0.18,
        "jitter_y": 0.06,
        "rotation": 0.08,
        "opacity_min": 205,
        "opacity_max": 250,
        "double_strike_prob": 0.012,
        "double_strike_offset": 0.55,
        "dropout_prob": 0.005,
        "char_spacing_jitter": 0.06,
        "baseline_wander": 0.03,
        "paper_noise": 2.5,
        "blur": 0.08,
    },
    "Used ribbon": {
        "jitter_x": 0.34,
        "jitter_y": 0.12,
        "rotation": 0.14,
        "opacity_min": 125,
        "opacity_max": 220,
        "double_strike_prob": 0.050,
        "double_strike_offset": 0.95,
        "dropout_prob": 0.032,
        "char_spacing_jitter": 0.14,
        "baseline_wander": 0.06,
        "paper_noise": 6.0,
        "blur": 0.20,
    },
    "Tired / fading ribbon": {
        "jitter_x": 0.40,
        "jitter_y": 0.16,
        "rotation": 0.18,
        "opacity_min": 80,
        "opacity_max": 195,
        "double_strike_prob": 0.070,
        "double_strike_offset": 1.15,
        "dropout_prob": 0.060,
        "char_spacing_jitter": 0.18,
        "baseline_wander": 0.08,
        "paper_noise": 7.5,
        "blur": 0.28,
    },
    "Cleaner digital typewriter": {
        "jitter_x": 0.16,
        "jitter_y": 0.18,
        "rotation": 0.12,
        "opacity_min": 225,
        "opacity_max": 250,
        "double_strike_prob": 0.004,
        "double_strike_offset": 0.4,
        "dropout_prob": 0.001,
        "char_spacing_jitter": 0.08,
        "baseline_wander": 0.10,
        "paper_noise": 1.5,
        "blur": 0.05,
    },
}


def mm_to_px(mm, dpi):
    return int(round(mm * dpi / 25.4))



def load_font(font_upload, font_size_px, face="Davoren Pica"):
    """
    Internal scaffold only. The visible 'machine face' is modified afterwards
    into a stable per-key typewriter impression.

    We avoid a distressed-font look: the same physical key gets the same fixed
    wear pattern every time it appears.
    """
    if font_upload is not None:
        data = font_upload.getvalue()
        return ImageFont.truetype(io.BytesIO(data), font_size_px)

    if face == "Davoren Elite":
        candidates = [
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ]

    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, font_size_px)
    return ImageFont.load_default()


def machine_glyph_mask(ch, font, face, machine_seed, px_scale=1.0):
    """
    Build a consistent glyph impression for one physical typebar/key.

    The important realism rule: fixed defects belong to the KEY, not to each
    occurrence. An 'e' with a worn upper-right edge repeats that wear whenever
    the same key strikes, while ribbon pressure can still vary per strike.
    """
    bbox = font.getbbox(ch)
    gw = max(16, bbox[2] - bbox[0] + 20)
    gh = max(20, bbox[3] - bbox[1] + 24)

    mask = Image.new("L", (gw, gh), 0)
    d = ImageDraw.Draw(mask)
    d.text((10 - bbox[0], 10 - bbox[1]), ch, font=font, fill=255)

    # Mechanical face shaping: slightly denser, more typebar-like impression.
    if face == "Davoren Pica":
        mask = mask.filter(ImageFilter.MaxFilter(3))
        # Pica: broader / heavier.
        new_w = max(1, int(mask.width * 1.035))
        mask = mask.resize((new_w, mask.height), Image.Resampling.LANCZOS)
    else:
        # Elite: narrower and a touch lighter.
        new_w = max(1, int(mask.width * 0.92))
        mask = mask.resize((new_w, mask.height), Image.Resampling.LANCZOS)
        mask = mask.filter(ImageFilter.MaxFilter(3))

    # Deterministic physical wear for the key.
    local_seed = (machine_seed * 1315423911 + ord(ch) * 2654435761) & 0xFFFFFFFF
    grng = np.random.default_rng(local_seed)

    # Very small fixed chips; these recur identically for the same character.
    arr = np.array(mask, dtype=np.uint8)
    ys, xs = np.where(arr > 70)
    if len(xs) > 0:
        chip_count = int(grng.integers(0, 3))
        for _ in range(chip_count):
            k = int(grng.integers(0, len(xs)))
            cx, cy = int(xs[k]), int(ys[k])
            rw = int(grng.integers(1, 3))
            rh = int(grng.integers(1, 3))
            x1, x2 = max(0, cx-rw), min(arr.shape[1], cx+rw+1)
            y1, y2 = max(0, cy-rh), min(arr.shape[0], cy+rh+1)
            arr[y1:y2, x1:x2] = (arr[y1:y2, x1:x2] * grng.uniform(0.25, 0.65)).astype(np.uint8)

    return Image.fromarray(arr, "L")

def add_paper_texture(img, rng, strength=5.0, warmth=2):
    arr = np.asarray(img).astype(np.int16)
    h, w, _ = arr.shape

    noise = rng.normal(0, strength, (h, w, 1))
    # Very subtle low-frequency paper variation
    small_h = max(2, h // 120)
    small_w = max(2, w // 120)
    low = rng.normal(0, strength * 0.8, (small_h, small_w))
    low_img = Image.fromarray(
        np.clip(low - low.min(), 0, 255).astype(np.uint8)
    ).resize((w, h), Image.Resampling.BILINEAR)
    low_arr = np.asarray(low_img).astype(np.float32)
    if low_arr.max() > 0:
        low_arr = (low_arr - low_arr.mean()) / max(low_arr.std(), 1e-6)
    low_arr = low_arr[:, :, None] * (strength * 0.7)

    arr = arr + noise + low_arr

    # Slight warm paper cast
    arr[:, :, 0] += warmth
    arr[:, :, 1] += max(0, warmth - 1)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def split_words_preserve_newlines(text):
    # Keep words/spaces/newlines so wrapping can respect intentional paragraph breaks.
    return re_tokenize(text)


def re_tokenize(text):
    import re
    return re.findall(r"\n|[^\S\n]+|[^\s]+", text)


def measure_char(font, ch):
    # Pillow textlength is not available directly on Font, so use bbox.
    bbox = font.getbbox(ch if ch else " ")
    return max(1, bbox[2] - bbox[0])


def text_width(font, text, letter_spacing=0):
    return sum(measure_char(font, ch) + letter_spacing for ch in text)


def wrap_text(text, font, max_width, letter_spacing=0):
    """
    Wrap while preserving explicit blank lines.
    Returns a list of lines.
    """
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = []

    for pi, para in enumerate(paragraphs):
        if para == "":
            lines.append("")
            continue

        words = para.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if text_width(font, candidate, letter_spacing) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                # Break very long words if needed
                if text_width(font, word, letter_spacing) > max_width:
                    part = ""
                    for ch in word:
                        cand = part + ch
                        if text_width(font, cand, letter_spacing) <= max_width:
                            part = cand
                        else:
                            if part:
                                lines.append(part)
                            part = ch
                    current = part
                else:
                    current = word
        if current:
            lines.append(current)

    return lines



def apply_layout_quirks(lines, rng, indent_variation=3, extra_space_prob=0.07, extra_space_amount=1.8):
    """
    Preserve straight typing baselines, but introduce realistic mechanical/human quirks:
    - slight line-start / paragraph indent variation
    - occasional extra spacing between words
    """
    out = []
    for i, line in enumerate(lines):
        if line == "":
            out.append({"text": "", "indent": 0, "space_boosts": {}})
            continue

        indent = int(rng.integers(-indent_variation, indent_variation + 1)) if indent_variation > 0 else 0

        # Larger indent variation after blank lines / paragraph starts
        if i == 0 or (i > 0 and lines[i-1] == ""):
            indent += int(rng.integers(0, max(1, indent_variation * 2 + 1)))

        boosts = {}
        for j, ch in enumerate(line):
            if ch == " " and rng.random() < extra_space_prob:
                boosts[j] = extra_space_amount

        out.append({"text": line, "indent": indent, "space_boosts": boosts})
    return out



def render_glyph(base, ch, font, x, y, rng, cfg, ink_rgb=(55, 52, 48)):
    """
    Strike one stable physical key impression.

    No visible up/down character wobble. Variation comes from ribbon pressure,
    fixed key wear, rare double strikes, and spacing / carriage behavior.
    """
    if ch == " ":
        return

    mask = machine_glyph_mask(
        ch,
        font,
        cfg["face"],
        cfg["machine_seed"],
    )

    # Per-strike ribbon pressure. This changes each occurrence, unlike key wear.
    alpha = int(rng.integers(cfg["opacity_min"], cfg["opacity_max"] + 1))
    strike_mask = mask.point(lambda p: int(p * alpha / 255))

    # Ribbon can fail to transfer evenly without moving the typebar baseline.
    if rng.random() < cfg["dropout_prob"]:
        a = np.array(strike_mask, dtype=np.uint8)
        h, w = a.shape
        band_y = int(rng.integers(0, max(1, h)))
        thickness = int(rng.integers(1, 3))
        a[max(0, band_y-thickness):min(h, band_y+thickness+1), :] = (
            a[max(0, band_y-thickness):min(h, band_y+thickness+1), :] * rng.uniform(0.25, 0.7)
        ).astype(np.uint8)
        strike_mask = Image.fromarray(a, "L")

    patch = Image.new("RGBA", strike_mask.size, (*ink_rgb, 0))
    patch.putalpha(strike_mask)

    if cfg["blur"] > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(cfg["blur"]))

    # Horizontal registration can vary slightly. Vertical stays locked.
    jx = float(rng.normal(0, cfg["jitter_x"]))
    px = int(round(x + jx))
    py = int(round(y))
    base.alpha_composite(patch, (px, py))

    # Rare carriage/typebar rebound. Mostly horizontal, almost never vertical.
    if rng.random() < cfg["double_strike_prob"]:
        ghost = patch.copy()
        a = ghost.getchannel("A").point(lambda p: int(p * 0.35))
        ghost.putalpha(a)
        off = max(1, int(round(cfg["double_strike_offset"])))
        gx = px + int(rng.choice([-1, 1]) * off)
        gy = py
        base.alpha_composite(ghost, (gx, gy))

def render_page(lines, page_index, settings, font_upload=None, reference_image=None):
    dpi = settings["dpi"]
    w = mm_to_px(A5_MM[0], dpi)
    h = mm_to_px(A5_MM[1], dpi)

    seed = settings["seed"] + page_index * 100003
    rng = np.random.default_rng(seed)

    paper = Image.new("RGB", (w, h), settings["paper_rgb"])
    paper = add_paper_texture(
        paper,
        rng,
        strength=settings["paper_noise"],
        warmth=settings["paper_warmth"]
    )

    # Optional very faint reference-paper character from a photo.
    # We deliberately DO NOT copy its typed content.
    if reference_image is not None and settings["reference_texture_strength"] > 0:
        ref = reference_image.convert("L")
        ref = ref.resize((w, h), Image.Resampling.BILINEAR)
        ref = ImageEnhance.Contrast(ref).enhance(0.35)
        ref = ImageEnhance.Brightness(ref).enhance(1.15)
        ref_rgb = Image.merge("RGB", (ref, ref, ref))
        alpha = settings["reference_texture_strength"] / 100.0
        paper = Image.blend(paper, ref_rgb, alpha=min(alpha, 0.12))

    base = paper.convert("RGBA")
    font = load_font(font_upload, settings["font_size_px"], settings["face"])

    margin_left = mm_to_px(settings["margin_left_mm"], dpi)
    margin_top = mm_to_px(settings["margin_top_mm"], dpi)

    nominal_char = measure_char(font, "M")
    nominal_line_h = settings["line_height_px"]

    quirky_lines = apply_layout_quirks(
        lines,
        rng,
        indent_variation=settings["indent_variation"],
        extra_space_prob=settings["extra_space_prob"],
        extra_space_amount=settings["extra_space_amount"],
    )

    for li, item in enumerate(quirky_lines):
        line = item["text"]
        y = margin_top + li * nominal_line_h

        # Mechanical carriage: the entire line shares one locked baseline.
        # We deliberately avoid per-character vertical movement.
        line_wander = 0.0
        x = float(margin_left + item["indent"] + rng.normal(0, settings["line_start_jitter"]))

        for ci, ch in enumerate(line):
            render_glyph(
                base,
                ch,
                font,
                x,
                y,
                rng,
                settings,
                ink_rgb=settings["ink_rgb"]
            )

            advance = measure_char(font, ch)
            advance += settings["tracking_px"]
            advance += float(rng.normal(0, settings["char_spacing_jitter"]))

            # More authentic error: extra word spacing, not vertical wobble.
            if ch == " " and ci in item["space_boosts"]:
                advance += measure_char(font, " ") * item["space_boosts"][ci]

            x += max(1.0, advance)

    return base.convert("RGB")


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


st.set_page_config(page_title="Field Notes Typewriter", page_icon="⌨️", layout="wide")

st.title("Field Notes Typewriter")
st.caption(
    "A mechanical typewriter renderer: straight carriage baselines, stable worn keys, uneven ribbon pressure and human spacing quirks."
)

with st.sidebar:
    st.header("Page")
    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name].copy()

    face = st.selectbox(
        "Machine face",
        ["Davoren Pica", "Davoren Elite"],
        index=0,
        help="Pica is broader/heavier; Elite is narrower and more compact."
    )

    dpi = st.select_slider("Output DPI", options=[150, 200, 240, 300], value=240)
    font_pt = st.slider("Type size", 9, 18, 12)
    font_size_px = max(10, int(font_pt * dpi / 72))

    margin_left = st.slider("Left margin (mm)", 8, 30, 12)
    margin_right = st.slider("Right margin (mm)", 8, 30, 12)
    margin_top = st.slider("Top margin (mm)", 10, 35, 18)
    margin_bottom = st.slider("Bottom margin (mm)", 10, 35, 18)

    line_spacing = st.slider("Line spacing", 1.0, 2.2, 1.55, 0.05)
    tracking = st.slider("Tracking", -1.0, 3.0, 0.15, 0.05)

    st.header("Layout quirks")
    indent_variation = st.slider("Line / paragraph start variation", 0, 10, 3)
    extra_space_prob = st.slider("Occasional double-space chance", 0.0, 0.20, 0.05, 0.01)
    extra_space_amount = st.slider("Extra-space strength", 1.0, 3.0, 1.8, 0.1)

    st.header("Imperfections")
    intensity = st.slider(
        "Mechanical imperfection",
        0.0, 2.0, 0.8, 0.05,
        help="Affects ink, horizontal registration and spacing — not wavy baselines."
    )
    ink_fade = st.slider("Ink fade", 0.0, 1.0, 0.08, 0.01)
    paper_noise_override = st.slider("Paper texture", 0.0, 12.0, float(preset["paper_noise"]), 0.5)
    seed = st.number_input("Random seed", min_value=0, max_value=9999999, value=650, step=1)

    st.header("Optional")
    font_upload = st.file_uploader(
        "Upload your own TTF/OTF font",
        type=["ttf", "otf"],
        help="For example, use a typewriter font you already have licensed."
    )

    ref_upload = st.file_uploader(
        "Upload one of your photographed typewriter pages",
        type=["png", "jpg", "jpeg"],
        help="Used only for a faint paper/lighting texture reference, not to copy the existing words."
    )
    ref_strength = st.slider("Reference texture influence", 0, 100, 0, 5)

st.subheader("Text")
default_text = """WHAT IS DEBT

This is money you borrow and
are expected to pay back.

The lender does not normally own
part of your project. Instead, you agree
how and when the money will be repaid...
usually with interest.

TAKEAWAY
---------
Can you repay the money?
"""
text = st.text_area("Paste your text", value=default_text, height=360, label_visibility="collapsed")

# Build effective settings
cfg = PRESETS[preset_name].copy()

# Scale imperfection-related values
for key in [
    "jitter_x",
    "double_strike_prob", "dropout_prob",
    "char_spacing_jitter", "blur"
]:
    cfg[key] *= intensity

# A typewriter carriage is mechanically straight within a line.
cfg["jitter_y"] = 0.0
cfg["rotation"] = 0.0
cfg["baseline_wander"] = 0.0

# Fade lowers alpha range
fade_amt = int(ink_fade * 90)
cfg["opacity_min"] = max(25, cfg["opacity_min"] - fade_amt)
cfg["opacity_max"] = max(cfg["opacity_min"] + 5, cfg["opacity_max"] - int(fade_amt * 0.6))
cfg["paper_noise"] = paper_noise_override

cfg.update({
    "dpi": dpi,
    "font_size_px": font_size_px,
    "margin_left_mm": margin_left,
    "margin_right_mm": margin_right,
    "margin_top_mm": margin_top,
    "margin_bottom_mm": margin_bottom,
    "line_height_px": int(font_size_px * line_spacing),
    "tracking_px": tracking * dpi / 72.0,
    "seed": int(seed),
    "paper_rgb": (246, 244, 236),
    "paper_warmth": 2,
    "ink_rgb": (52, 49, 45),
    "reference_texture_strength": ref_strength,
    "line_start_jitter": 0.22 * intensity,
    "micro_baseline_jitter": 0.0,
    "indent_variation": int(indent_variation * dpi / 240),
    "face": face,
    "machine_seed": 650,
    "extra_space_prob": extra_space_prob,
    "extra_space_amount": extra_space_amount,
})

reference_image = None
if ref_upload is not None:
    reference_image = Image.open(ref_upload).convert("RGB")

font_for_wrap = load_font(font_upload, font_size_px, face)
page_w = mm_to_px(A5_MM[0], dpi)
page_h = mm_to_px(A5_MM[1], dpi)
max_text_width = page_w - mm_to_px(margin_left + margin_right, dpi)
max_text_height = page_h - mm_to_px(margin_top + margin_bottom, dpi)

lines = wrap_text(text, font_for_wrap, max_text_width, cfg["tracking_px"])
lines_per_page = max(1, int(max_text_height // cfg["line_height_px"]))
pages_lines = [lines[i:i+lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[""]]

if st.button("Generate pages", type="primary", use_container_width=True):
    rendered = [
        render_page(
            plines,
            page_index=i,
            settings=cfg,
            font_upload=font_upload,
            reference_image=reference_image
        )
        for i, plines in enumerate(pages_lines)
    ]

    st.session_state["rendered"] = [png_bytes(im) for im in rendered]
    st.session_state["settings_summary"] = {
        "preset": preset_name,
        "seed": int(seed),
        "dpi": dpi,
        "font_pt": font_pt,
        "page_count": len(rendered),
    }

if "rendered" in st.session_state:
    rendered_bytes = st.session_state["rendered"]
    st.success(f"Generated {len(rendered_bytes)} A5 page(s).")

    cols = st.columns(min(3, len(rendered_bytes)))
    for i, b in enumerate(rendered_bytes):
        with cols[i % len(cols)]:
            st.image(b, caption=f"Page {i+1}", use_container_width=True)
            st.download_button(
                f"Download page {i+1}",
                data=b,
                file_name=f"field_note_{i+1:02d}.png",
                mime="image/png",
                key=f"download_{i}",
                use_container_width=True
            )

    # ZIP all pages
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for i, b in enumerate(rendered_bytes):
            z.writestr(f"field_note_{i+1:02d}.png", b)
    st.download_button(
        "Download all pages as ZIP",
        data=zbuf.getvalue(),
        file_name="field_notes_typewriter_pages.zip",
        mime="application/zip",
        use_container_width=True
    )

with st.expander("Why this is useful"):
    st.write(
        """
        This does not ask an image model to invent a page. It renders your actual text
        character by character using a stable simulated machine face. Each physical
        key has its own repeatable wear pattern, while ribbon pressure varies by strike.
        The carriage baseline stays locked. Imperfection comes from ink transfer,
        occasional rebound/double strikes, line-start differences, double spaces and
        paper texture — not random vertical wobble.

        Because the random seed is fixed, a page can be regenerated exactly. Change the
        seed when you want a different physical-looking copy.
        """
    )
