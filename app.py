
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
        "opacity_min": 190,
        "opacity_max": 245,
        "double_strike_prob": 0.015,
        "double_strike_offset": 0.8,
        "dropout_prob": 0.008,
        "char_spacing_jitter": 0.04,
        "baseline_wander": 0.05,
        "paper_noise": 5.0,
        "blur": 0.05,
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



def machine_glyph_mask(ch, font, face, machine_seed):
    """
    Create a tight, readable typebar impression.

    The earlier version used oversized padded masks, which could overlap
    neighbouring characters. This version crops each glyph tightly and only
    applies subtle fixed wear.
    """
    bbox = font.getbbox(ch)
    left, top, right, bottom = bbox
    gw = max(4, right - left)
    gh = max(4, bottom - top)

    # Small safety padding only.
    pad = 3
    mask = Image.new("L", (gw + pad * 2, gh + pad * 2), 0)
    d = ImageDraw.Draw(mask)
    d.text((pad - left, pad - top), ch, font=font, fill=255)

    # Keep the face readable. No aggressive dilation.
    if face == "Davoren Pica":
        # Very slightly heavier.
        mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = ImageEnhance.Contrast(mask).enhance(1.05)
    else:
        # Slightly narrower Elite feel.
        new_w = max(3, int(mask.width * 0.93))
        mask = mask.resize((new_w, mask.height), Image.Resampling.LANCZOS)

    # Fixed, tiny wear marks per key.
    local_seed = (machine_seed * 1315423911 + ord(ch) * 2654435761) & 0xFFFFFFFF
    grng = np.random.default_rng(local_seed)

    arr = np.array(mask, dtype=np.uint8)
    ys, xs = np.where(arr > 100)
    if len(xs) > 0 and grng.random() < 0.45:
        k = int(grng.integers(0, len(xs)))
        cx, cy = int(xs[k]), int(ys[k])
        arr[max(0, cy-1):min(arr.shape[0], cy+2),
            max(0, cx-1):min(arr.shape[1], cx+2)] = (
            arr[max(0, cy-1):min(arr.shape[0], cy+2),
                max(0, cx-1):min(arr.shape[1], cx+2)] * 0.58
        ).astype(np.uint8)

    out = Image.fromarray(arr, "L")

    # Crop transparent padding so glyphs cannot collide due to huge masks.
    tight = out.getbbox()
    if tight:
        out = out.crop(tight)

    return out

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




def wrap_text_fixed_pitch(text, max_chars):
    """
    Wrap to a real typewriter carriage width using fixed character cells.
    Explicit line breaks are preserved.
    """
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = []
    for para in paragraphs:
        if para == "":
            lines.append("")
            continue
        words = para.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    lines.append(current)
                while len(word) > max_chars:
                    lines.append(word[:max_chars])
                    word = word[max_chars:]
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




def render_glyph(base, ch, font, x, baseline_y, rng, cfg, ink_rgb=(45, 43, 40)):
    """
    Render one readable typewriter strike on a locked baseline.
    """
    if ch == " ":
        return

    mask = machine_glyph_mask(ch, font, cfg["face"], cfg["machine_seed"])

    # Ribbon transfer varies per strike. Most letters sit near the ribbon-age
    # baseline, while occasional keys hit distinctly darker or lighter — like
    # the photographed pages.
    midpoint = (cfg["opacity_min"] + cfg["opacity_max"]) / 2
    alpha = int(rng.normal(midpoint, cfg["strike_variation"]))

    # Randomly stronger strike / fresher patch of ribbon.
    if rng.random() < cfg["dark_hit_prob"]:
        alpha += int(rng.integers(22, 55))

    # Randomly weaker ink transfer.
    if rng.random() < cfg["light_hit_prob"]:
        alpha -= int(rng.integers(22, 60))

    alpha = int(np.clip(alpha, 55, 255))

    strike_mask = mask.point(lambda p: int(p * alpha / 255))

    # Local ribbon weakness, but never enough to destroy legibility.
    if rng.random() < cfg["dropout_prob"]:
        arr = np.array(strike_mask, dtype=np.uint8)
        h, w = arr.shape
        if h > 4:
            band_y = int(rng.integers(1, h - 1))
            arr[band_y:band_y+1, :] = (arr[band_y:band_y+1, :] * 0.70).astype(np.uint8)
        strike_mask = Image.fromarray(arr, "L")

    patch = Image.new("RGBA", strike_mask.size, (*ink_rgb, 0))
    patch.putalpha(strike_mask)

    # Minimal blur only.
    if cfg["blur"] > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(min(cfg["blur"], 0.10)))

    # Horizontal-only registration variation.
    jx = float(rng.normal(0, cfg["jitter_x"]))
    px = int(round(x + jx))

    # Position glyph by its actual font ascent/descent relationship.
    bbox = font.getbbox(ch)
    glyph_top = bbox[1]
    py = int(round(baseline_y + glyph_top))

    base.alpha_composite(patch, (px, py))

    # Rare light rebound, horizontal only.
    if rng.random() < cfg["double_strike_prob"]:
        ghost = patch.copy()
        a = ghost.getchannel("A").point(lambda p: int(p * 0.22))
        ghost.putalpha(a)
        gx = px + int(rng.choice([-1, 1]))
        base.alpha_composite(ghost, (gx, py))

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

            # Real typewriter carriage pitch: fixed 10 CPI = 2.54 mm per keystroke.
            advance = settings["char_pitch_px"]
            advance += float(rng.normal(0, settings["char_spacing_jitter"]))

            # Occasional extra word space is an additional carriage step.
            if ch == " " and ci in item["space_boosts"]:
                advance += settings["char_pitch_px"] * item["space_boosts"][ci]

            x += max(1.0, advance)

    return base.convert("RGB")


def png_bytes(img):
    b = io.BytesIO()
    img.save(b, format="PNG", optimize=True)
    return b.getvalue()


st.set_page_config(page_title="Field Notes Typewriter", page_icon="⌨️", layout="wide")

st.title("Field Notes Typewriter")
st.caption(
    "True A5 output (148 × 210 mm at 300 DPI), calibrated to the uploaded A5 typewriter pages."
)

with st.sidebar:
    st.header("Typewriter")

    face = st.selectbox(
        "Font",
        ["Davoren Pica", "Davoren Elite"],
        index=0,
        help="Pica is broader and heavier; Elite is narrower and more compact."
    )

    ribbon_age = st.selectbox(
        "Ribbon age",
        ["Fresh", "Used", "Fading"],
        index=1,
        help="Fresh = dark/consistent; Used = mixed strikes; Fading = lighter with much more letter-to-letter variation."
    )

    imperfection = st.slider(
        "Imperfections",
        0, 100, 38, 1,
        help="Controls spacing quirks, line-start variation, rebound and mechanical irregularity. Ribbon age controls most ink variation."
    )

    font_upload = st.file_uploader(
        "Optional custom TTF/OTF font",
        type=["ttf", "otf"],
        help="Leave blank to use the built-in typewriter faces."
    )

    seed = st.number_input(
        "Page seed",
        min_value=0,
        max_value=9999999,
        value=650,
        step=1,
        help="Change this only if you want a different arrangement of imperfections."
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
---------
Can you repay the money?
"""
text = st.text_area("Paste your text", value=default_text, height=360, label_visibility="collapsed")

# Build effective settings from a very small user-facing control set.
dpi = 300
font_pt = 12.0
font_size_px = int(round(font_pt * dpi / 72))

# Fixed A5 layout tuned to the photographed field-note pages.
margin_left = 5.5
margin_right = 5.5
margin_top = 11.5
margin_bottom = 12
line_spacing = 1.0
tracking = 0.0

# One imperfections slider drives the mechanical behaviour.
imp = imperfection / 100.0

# Ribbon age controls ink separately from the mechanical imperfections.
if ribbon_age == "Fresh":
    opacity_min, opacity_max = 218, 252
    base_dropout = 0.001
    paper_noise = 2.2
    strike_variation = 10
    dark_hit_prob = 0.08
    light_hit_prob = 0.03
elif ribbon_age == "Used":
    opacity_min, opacity_max = 165, 235
    base_dropout = 0.006
    paper_noise = 3.8
    strike_variation = 24
    dark_hit_prob = 0.13
    light_hit_prob = 0.08
else:  # Fading
    opacity_min, opacity_max = 105, 205
    base_dropout = 0.014
    paper_noise = 4.8
    strike_variation = 34
    dark_hit_prob = 0.16
    light_hit_prob = 0.14

cfg = {
    "jitter_x": 0.03 + imp * 0.16,
    "jitter_y": 0.0,
    "rotation": 0.0,
    "opacity_min": opacity_min,
    "opacity_max": opacity_max,
    "strike_variation": strike_variation,
    "dark_hit_prob": dark_hit_prob,
    "light_hit_prob": light_hit_prob,
    "double_strike_prob": 0.002 + imp * 0.016,
    "double_strike_offset": 1.0,
    "dropout_prob": base_dropout + imp * base_dropout,
    "char_spacing_jitter": imp * 0.045,
    "baseline_wander": 0.0,
    "paper_noise": paper_noise,
    "blur": min(0.06, imp * 0.05),

    "dpi": dpi,
    "font_size_px": font_size_px,
    "margin_left_mm": margin_left,
    "margin_right_mm": margin_right,
    "margin_top_mm": margin_top,
    "margin_bottom_mm": margin_bottom,
    # Keep carriage line pitch fixed so Font size changes the LETTER SIZE,
    # not the distance between lines.
    "line_height_px": mm_to_px(9.0, dpi),
    "tracking_px": 0.0,
    "char_pitch_px": mm_to_px(2.54, dpi),
    "seed": int(seed),
    "paper_rgb": (246, 244, 236),
    "paper_warmth": 2,
    "ink_rgb": (48, 45, 42),
    "reference_texture_strength": 0,
    "line_start_jitter": imp * 0.22,
    "micro_baseline_jitter": 0.0,
    "indent_variation": int(round(imp * 4)),
    "extra_space_prob": 0.01 + imp * 0.055,
    "extra_space_amount": 1.3 + imp * 0.8,
    "face": face,
    "machine_seed": 650,
}

reference_image = None

font_for_wrap = load_font(font_upload, font_size_px, face)
page_w = mm_to_px(A5_MM[0], dpi)
page_h = mm_to_px(A5_MM[1], dpi)
max_text_width = page_w - mm_to_px(margin_left + margin_right, dpi)
max_text_height = page_h - mm_to_px(margin_top + margin_bottom, dpi)

max_chars = max(1, int(max_text_width // cfg["char_pitch_px"]))
lines = wrap_text_fixed_pitch(text, max_chars)
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
        "font": face,
        "ribbon_age": ribbon_age,
        "imperfections": imperfection,
        "seed": int(seed),
        "dpi": dpi,
        "font_pt": 12.0,
        "page_count": len(rendered),
    }

if "rendered" in st.session_state:
    rendered_bytes = st.session_state["rendered"]
    st.success(f"Generated {len(rendered_bytes)} calibrated A5 page(s) — 148 × 210 mm, 300 DPI, fixed 10-CPI typewriter pitch.")

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

with st.expander("How the imperfections work"):
    st.write(
        """
        **Font size** changes the actual letter size while the carriage line pitch stays fixed.

        **Ribbon age** changes both average ink darkness and how uneven individual
        strikes are. A Used or Fading ribbon will therefore contain random darker
        letters among lighter ones, rather than applying one uniform fade to the page.

        **Imperfections** controls the mechanical side: spacing quirks, occasional
        double spaces, line-start differences and rare horizontal rebound. The
        baseline remains straight.
        """
    )
