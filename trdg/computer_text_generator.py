import os
import subprocess
import io
import math
import re
import random as rnd
from typing import Tuple

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageChops

from trdg.utils import get_text_width, get_text_height
import matplotlib as mpl
import matplotlib.pyplot as plt

# Thai Unicode reference: https://jrgraphix.net/r/Unicode/0E00-0E7F
TH_TONE_MARKS = [
    "0xe47",
    "0xe48",
    "0xe49",
    "0xe4a",
    "0xe4b",
    "0xe4c",
    "0xe4d",
    "0xe4e",
]
TH_UNDER_VOWELS = ["0xe38", "0xe39", "\0xe3A"]
TH_UPPER_VOWELS = ["0xe31", "0xe34", "0xe35", "0xe36", "0xe37"]


def generate(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    orientation: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
    max_width: int = 0,
) -> Tuple:
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
            max_width,
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            stroke_width,
            stroke_fill,
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _compute_character_width(image_font: ImageFont, character: str) -> int:
    if len(character) == 1 and (
        "{0:#x}".format(ord(character))
        in TH_TONE_MARKS + TH_UNDER_VOWELS + TH_UNDER_VOWELS + TH_UPPER_VOWELS
    ):
        return 0
    # Casting as int to preserve the old behavior
    return round(image_font.getlength(character))


def render_asy(asy_code, max_width, background="black", color="white"):
    asy_code = asy_code[5:-6]  # Remove [asy] and [/asy]

    with open('temp.asy', 'w') as f:
        f.write(asy_code)

    try:
        subprocess.run(['asy', 'temp.asy'], check=True)
        img = Image.open('temp.eps')
        scale = max_width // img.size[0] + 1
        img.load(scale=scale)
        img = img.convert('RGBA')

        img = img.resize((max_width // 2, int(max_width * img.size[1] / img.size[0]) // 2), Image.LANCZOS)
        bg = Image.new('RGBA', (max_width, img.size[1]), (255, 255, 255))
        bg.paste(img, (max_width // 2 - img.size[0] // 2, 0))
        img = bg


        # set while background to transparent
        data = np.array(img)
        r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
        white_areas = (r == 255) & (g == 255) & (b == 255)
        data[:, :, 3][white_areas] = 0
        img = Image.fromarray(data)

    except subprocess.CalledProcessError as e:
        print(f"Error running Asymptote: {e}")
    finally:
        pass
        for file in ['temp.asy', 'temp.png']:
            if os.path.exists(file):
                os.remove(file)

    return img


def render_latex(formula, fontsize=32, background="black", color="white"):
    buf = io.BytesIO()
    plt.figure(facecolor=background)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}} \usepackage{{amssymb}}'
    plt.axis('off')
    plt.text(0.0, 0, f'Hq {formula}', size=fontsize, color=tuple([c / 255 for c in color]))
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()

    im = Image.open(buf)
    bbox = im.convert("RGB").getbbox()
    bbox = (bbox[0] + 60 * fontsize / 32, bbox[1] - 5 * fontsize / 32, bbox[2] + 5 * fontsize / 32, bbox[3] + 5 * fontsize / 32)
    im = im.crop(bbox).convert("RGBA")
    # set black background to transparent
    data = np.array(im)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    black_areas = (r == 0) & (g == 0) & (b == 0)
    data[:, :, 3][black_areas] = 0
    im = Image.fromarray(data)
    return im


def _generate_horizontal_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    word_split: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
    max_width: int = 0,
) -> Tuple:
    assert max_width >= 0

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    # preprocess
    text = text.replace("$$", "$")
    text = text.replace("\\[", "$")
    text = text.replace("\\]", "$")

    # math and asy code support
    render_scale = 1.0
    rendered = []
    math_expressions = list(set(re.findall(r"\$.*?\$", text, re.DOTALL)))
    asy_codes = list(set(re.findall(r"\[asy\].*?\[/asy\]", text, re.DOTALL)))
    for expr in math_expressions:
        text = text.replace(expr, f" <|{len(rendered)}|> ")
        rendered.append(render_latex(expr, fontsize=font_size, background=(0, 0, 0), color=fill))
    for asy_code in asy_codes:
        text = text.replace(asy_code, f" <|{len(rendered)}|> ")
        rendered.append(render_asy(asy_code, max_width, background=(0, 0, 0), color=fill))

    text = text.replace("\n", " ")

    image_font = ImageFont.truetype(font=font, size=font_size)
    space_width = int(get_text_width(image_font, " ") * space_width)

    if word_split:
        splitted_text = []
        for w in text.split(" "):
            if "<|" in w and "|>" in w:
                start = w.find("<|")
                end = w.find("|>")
                if start > 0:
                    splitted_text.append(w[:start])
                splitted_text.append(w[start:end + 2])
                if end + 2 < len(w):
                    splitted_text.append(w[end + 2:])
                splitted_text.append(" ")
            else:
                splitted_text.append(w)
                splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text

    def lut_width(c):
        if c == " ":
            return space_width
        if c.startswith("<|") and c.endswith("|>"):
            math_id = int(c[2:-2])
            return math.ceil(rendered[math_id].size[0] * render_scale)
        return _compute_character_width(image_font, c)

    def lut_height(c):
        if c.startswith("<|") and c.endswith("|>"):
            math_id = int(c[2:-2])
            return math.ceil(rendered[math_id].size[1] * render_scale)
        return get_text_height(image_font, p)

    piece_widths = [lut_width(c) for c in splitted_text]

    # get lines
    line_width = 0
    line_split_text = [[]]
    for p in splitted_text:
        w = lut_width(p)
        if line_width + w > max_width:
            line_split_text.append([p])
            line_width = w
        else:
            line_split_text[-1].append(p)
            line_width += w
    line_piece_widths = [[lut_width(p) for p in split_text] for split_text in line_split_text]

    text_width = min(sum(piece_widths), max_width)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    text_height_baseline = lut_height("Hq")
    text_height = [max([lut_height(p) for p in line]) for line in line_split_text]
    text_height = [0] + np.cumsum(text_height).tolist()

    txt_img = Image.new("RGBA", (text_width, text_height[-1]), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height[-1]), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    for i, (line, p_weights) in enumerate(zip(line_split_text, line_piece_widths)):
        for j, p in enumerate(line):
            if p.startswith("<|") and p.endswith("|>"):
                math_id = int(p[2:-2])
                if j == 0 and len(line) == 1 and rendered[math_id].size[0] > max_width:
                    new_height = int(max_width * rendered[math_id].size[1] / rendered[math_id].size[0])
                    resized_math = rendered[math_id].resize((max_width, new_height))
                    txt_img.paste(
                        resized_math,
                        (sum(p_weights[0:j]) + j * character_spacing, text_height[i]),
                    )
                else:
                    txt_img.paste(
                        rendered[math_id],
                        (sum(p_weights[0:j]) + j * character_spacing, text_height[i]),
                    )
            else:
                line_height = text_height[i + 1] - text_height[i]
                plot_height = text_height[i] + (line_height - text_height_baseline) // 2
                txt_img_draw.text(
                    (sum(p_weights[0:j]) + j * character_spacing * int(not word_split), plot_height),
                    p,
                    fill=fill,
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
                txt_mask_draw.text(
                    (sum(p_weights[0:j]) + j * character_spacing * int(not word_split), plot_height),
                    p,
                    fill=((j + 1) // (255 * 255), (j + 1) // 255, (j + 1) % 255),
                    font=image_font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask


def _generate_vertical_text(
    text: str,
    font: str,
    text_color: str,
    font_size: int,
    space_width: int,
    character_spacing: int,
    fit: bool,
    stroke_width: int = 0,
    stroke_fill: str = "#282828",
) -> Tuple:
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(get_text_height(image_font, " ") * space_width)

    char_heights = [
        get_text_height(image_font, c) if c != " " else space_height for c in text
    ]
    text_width = max([get_text_width(image_font, c) for c in text])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    for i, c in enumerate(text):
        txt_img_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (0, sum(char_heights[0:i]) + i * character_spacing),
            c,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox())
    else:
        return txt_img, txt_mask
