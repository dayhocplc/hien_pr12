# -*- coding: utf-8 -*-
"""
figstyle.py — Kiểu trình bày thống nhất cho toàn bộ hình của luận văn.

Nguyên tắc:
  • In được đen trắng: mọi cặp đường phân biệt được bằng nét, không chỉ bằng màu.
  • Không dùng màu đỏ/xanh lá cạnh nhau làm cặp đối lập duy nhất (mù màu đỏ–lục).
  • Mọi hình xuất cả PNG 300 dpi (chèn Word) và PDF vector (in ấn).
  • Không có tiêu đề trong hình — tiêu đề hình đặt ở caption của luận văn.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "out", "figures")

# Bảng màu an toàn cho người mù màu (Okabe–Ito)
C = {
    "xanh":    "#0072B2",   # nhánh luật / cấu hình chính
    "cam":     "#E69F00",   # nhánh k-NN
    "luc":     "#009E73",   # nhánh FCM
    "tim":     "#CC79A7",   # mô hình lai
    "vang":    "#F0E442",
    "xanhnhat": "#56B4E9",
    "donhat":  "#D55E00",   # lưới an toàn BYT / cảnh báo
    "xam":     "#666666",
    "xamnhat": "#CCCCCC",
}

DOMAIN_COLOR = {
    "GTNN": "#0072B2", "CNXH": "#D55E00", "BCHOC": "#009E73",
    "DHBT": "#CC79A7", "VDTINH": "#56B4E9", "VDTHO": "#E69F00",
}


def setup():
    rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })
    os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name: str, caption: str = "") -> str:
    """Lưu PNG 300 dpi + PDF vector. Trả về đường dẫn PNG."""
    os.makedirs(FIG_DIR, exist_ok=True)
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(png, dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    plt.close(fig)
    if caption:
        with open(os.path.join(FIG_DIR, "captions.txt"), "a", encoding="utf-8") as fh:
            fh.write(f"{name}\t{caption}\n")
    print(f"  ✓ {name}")
    return png


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8, lw=1.0, ls="-"):
    """Hộp chữ nhật bo góc kèm chữ căn giữa — dùng cho sơ đồ khối."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.45)


def arrow(ax, xy_from, xy_to, ec="#333333", lw=1.1, ls="-", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(xy_from, xy_to,
                                 arrowstyle="-|>", mutation_scale=10,
                                 color=ec, linewidth=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def blank_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)
