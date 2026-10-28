# -*- coding: utf-8 -*-
"""
run_sensitivity.py — PHÂN TÍCH ĐỘ NHẠY TOÀN CỤC trên không gian tham số của config.py.

    .venv/Scripts/python.exe src/run_sensitivity.py

Sinh ra:  out/reports/sensitivity.json   (mọi con số dùng trong bài P12)
          out/reports/sensitivity.txt    (bản in để đọc)

VÌ SAO CẦN TỆP NÀY
------------------
config.py phơi ra hơn 25 hằng số có thể chỉnh. Đặc tả không nói hằng số nào
đáng chỉnh và hằng số nào chỉ là chi tiết cài đặt. Bằng chứng duy nhất đang có
là một lát cắt MỘT chiều của phi_DHBT (R15): toàn bộ biên độ 0,0129 AUC, nhỏ
hơn sai số chuẩn của chính phép đo. Một lát cắt một chiều không trả lời được
câu hỏi toàn cục, vì nó giữ mọi tham số khác cố định ở đúng một điểm.

BA VIỆC TỆP NÀY LÀM  (T15 · T17 · T18 của §13)
----------------------------------------------
T15  Khung fANOVA/Sobol: lấy mẫu ngẫu nhiên trên lưới đầy đủ, khớp mô hình thay
     thế rừng ngẫu nhiên, phân rã phương sai thành chỉ số bậc nhất và bậc toàn
     phần (Sobol) — đúng thủ tục Hutter và cs. cho tầm quan trọng siêu tham số.
T17  Hiệu chỉnh Benjamini-Hochberg ở FDR = 0,05 trên toàn bộ các trục, kèm khai
     báo số cấu hình đã thử (§11.5).
T18  Khoá hạt giống và ghi nhật ký cấu hình: mọi con số tái lập được.

BA THỦ PHÁP LÀM CHO KẾT LUẬN CÓ THỂ KIỂM CHỨNG
----------------------------------------------
1. TRỤC GIẢ LẬP (sham axis). Hai trục GIA_LAP_A, GIA_LAP_B được đưa vào thiết
   kế nhưng KHÔNG nối vào bất kỳ phép tính nào. Mọi phương sai mà chúng nhận
   được là nhiễu thuần tuý. Đây là NGƯỠNG NHIỄU đo được từ chính thí nghiệm,
   không phải một ngưỡng tự đặt.
2. NGƯỠNG SAI SỐ CỦA PHÉP ĐO. Sai số chuẩn của AUC được ước lượng hai cách
   độc lập: bootstrap phân tầng trên cùng một đoàn hệ, và độ lệch chuẩn giữa
   các hạt giống sinh dữ liệu khác nhau. Một tham số chỉ đáng gọi là "quan
   trọng" khi biên độ của nó vượt được ngưỡng này.
3. SÀNG LỌC GIẢI TÍCH TRƯỚC. Một số hằng số KHÔNG THỂ tác động tới AUC vì
   chúng không nằm trên đường tính điểm (chỉ đổi khoảng bất định, nhãn tầng,
   hoặc không được đọc ở bất kỳ đâu trong mã nguồn). Chúng được loại bằng lập
   luận, không bằng thí nghiệm, và được liệt kê riêng.

GIỚI HẠN BẮT BUỘC KHAI BÁO
--------------------------
L3  Toàn bộ kết quả ở đây chạy trên D4 — dữ liệu MÔ PHỎNG. Kết luận đúng cho
    dữ liệu sinh theo giả định của bộ sinh, không tự động đúng cho trẻ thật.
    D4 là tập DUY NHẤT có đủ 6 lĩnh vực và có mốc tuổi kỳ vọng, nên là tập duy
    nhất kích hoạt được toàn bộ đường tính điểm (D2 chỉ phủ 4/6 lĩnh vực và
    không có LAG — xem L4).
L4  Vì lý do trên, kết quả ở đây KHÔNG chuyển giao trực tiếp sang D2.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config as cfg
import ts_fuzzy
from ts_aggregate import aggregate_rule_branch
from ts_dataset import load_questions
from ts_evaluate import roc_auc as _roc_auc_ref
from ts_fcm import FuzzyCMeans
from ts_features import build_feature_vector
from ts_fusion import apply_safety_net, byt_threshold_met, fuse
from ts_fuzzy import build_all_fis, infer_all_domains, risk_to_unit
from ts_mf import centroid as centroid_ref
from ts_simulate import generate_dataset

# =====================================================================
# T18 — HẠT GIỐNG VÀ QUY MÔ, KHOÁ CỨNG
# =====================================================================
SEED_COHORT = int(os.environ.get("SENS_SEED_COHORT", "20260829"))  # sinh đoàn hệ D4 chính
SEED_DESIGN = int(os.environ.get("SENS_SEED_DESIGN", "20260830"))  # lấy mẫu cấu hình trên lưới
SEED_SPLIT = 20260831           # chia huấn luyện / kiểm thử
SEED_BOOT = 20260832            # bootstrap sai số chuẩn của AUC
SEED_PERM = 20260833            # kiểm định hoán vị
SEED_SURROGATE = 20260834       # rừng ngẫu nhiên + Monte Carlo Sobol
SEED_REPLICA = [20260901, 20260902, 20260903, 20260904, 20260905]

N_RAW = int(os.environ.get("SENS_N_RAW", "9000"))       # số hồ sơ sinh ra
N_CONFIG = int(os.environ.get("SENS_N_CONFIG", "512"))  # số cấu hình lấy mẫu
N_BOOT = 2000
N_PERM = 5000
N_MC_SOBOL = 40_000            # cỡ mẫu Monte Carlo trên mô hình thay thế
FDR_Q = 0.05
# P12/T12: nhãn tệp đầu ra (vd "_n50k") và chế độ chỉ chạy lõi (bỏ lặp hạt
# giống, lát cắt, độ phủ, ràng buộc) cho lần lặp độc lập. Mặc định giữ nguyên
# hành vi cũ: sensitivity.json, chạy đủ.
OUT_TAG = os.environ.get("SENS_TAG", "")
CORE_ONLY = os.environ.get("SENS_CORE_ONLY", "0") == "1"

# Kịch bản giả định của BỘ SINH (phản biện MS.22, yêu cầu 4). Biến môi trường
# SENS_GEN="prev=0.01;dhbt=-3.5;noise=0.025" ghi đè tỷ lệ mắc, logit nền của
# dấu hiệu bất thường và xác suất nhiễu trả lời cơ sở. Áp ngay khi nạp mô-đun
# nên tiến trình con (spawn trên Windows) thừa hưởng qua biến môi trường.
# Để trống = đúng bộ sinh của luận văn; mọi kết quả đã công bố không đổi.
GEN_SPEC = os.environ.get("SENS_GEN", "").strip()
GEN_OVERRIDES = {}
for _kv in filter(None, GEN_SPEC.split(";")):
    _k, _v = _kv.split("=")
    GEN_OVERRIDES[_k.strip()] = float(_v)
if "prev" in GEN_OVERRIDES:
    cfg.SIM_CONFIG["prevalence"] = GEN_OVERRIDES["prev"]
if "dhbt" in GEN_OVERRIDES:
    cfg.DHBT_BASE_LOGIT = GEN_OVERRIDES["dhbt"]
if "noise" in GEN_OVERRIDES:
    cfg.P_NOISE_BASE = GEN_OVERRIDES["noise"]
if "noiseabs" in GEN_OVERRIDES:
    cfg.P_NOISE_ABSTRACT = GEN_OVERRIDES["noiseabs"]
if "theta" in GEN_OVERRIDES:
    # dịch năng lực tiềm ẩn của MỌI trẻ (hai lớp như nhau) → câu "dễ" hơn,
    # ít trẻ chạm ngưỡng hơn; khoảng cách giữa hai lớp giữ nguyên.
    import ts_simulate as _tss
    _orig_risk = _tss._risk_probability
    _shift = GEN_OVERRIDES["theta"]
    _tss._risk_probability = lambda q_, th_, a_: _orig_risk(q_, th_ + _shift, a_)
GEN_TAG = "" if not GEN_OVERRIDES else "_g" + "_".join(
    "{}{}".format(k, v) for k, v in sorted(GEN_OVERRIDES.items()))

# =====================================================================
# GIẢI MỜ — cfg.DEFUZZ được KHAI BÁO nhưng KHÔNG được đọc ở bất kỳ đâu
# trong mã nguồn: ts_fuzzy.infer gọi thẳng centroid. Để đưa DEFUZZ vào thí
# nghiệm như một trục thật, ta thay hàm giải mờ trong không gian tên của
# ts_fuzzy. Việc PHẢI làm như vậy chính là một phát hiện của bài toán này.
# =====================================================================


def defuzz_bisector(universe, mu):
    total = float(mu.sum())
    if total <= 0.0:
        return None
    c = np.cumsum(mu)
    return float(universe[int(np.searchsorted(c, total / 2.0))])


def defuzz_mom(universe, mu):
    peak = float(mu.max())
    if peak <= 0.0:
        return None
    return float(universe[mu >= peak - 1e-12].mean())


def roc_auc(y_true, scores):
    """AUC Mann–Whitney bằng hạng trung bình — TRÙNG KHÍT với
    ts_evaluate.roc_auc (điểm bằng nhau tính 1/2) nhưng O(n log n) thay vì
    O(n+ · n−). Với n = 50 000 (T12), bản gốc tốn ~0,7 s mỗi lần gọi và
    bootstrap 2 000 lần trở thành nút cổ chai. Hạng trung bình là bội của 1/2
    nên phép tính chính xác tuyệt đối trong dấu phẩy động ở cỡ mẫu này."""
    from scipy.stats import rankdata
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


DEFUZZ_FN = {"centroid": centroid_ref,
             "bisector": defuzz_bisector,
             "mom": defuzz_mom}

# =====================================================================
# TRỤC THAM SỐ — 18 trục thật + 2 trục giả lập
# =====================================================================
OWA_PROFILE = {
    "doc":        [0.40, 0.25, 0.15, 0.10, 0.06, 0.04],   # bộ đang dùng
    "tuyen_tinh": [0.2857, 0.2381, 0.1905, 0.1429, 0.0952, 0.0476],
    "deu":        [1 / 6] * 6,
}
FUSION_PROFILE = {
    "chuan":     {"rule": 0.50, "fknn": 0.30, "fcm": 0.20},   # bộ đang dùng
    "deu":       {"rule": 1 / 3, "fknn": 1 / 3, "fcm": 1 / 3},
    "luat_troi": {"rule": 0.80, "fknn": 0.12, "fcm": 0.08},
}

AXES = {
    "T_NORM":               ["product", "min"],
    "S_NORM":               ["bounded_sum", "prob_sum", "max"],
    "DEFUZZ":               ["centroid", "bisector", "mom"],
    "MF_SET":               ["v2", "v1"],
    "SHAPLEY":              ["v2", "v1"],
    "INTERACTION":          ["rong", "lam_sang"],
    "AGG_OP":               ["choquet", "owa_doc", "owa_tuyen_tinh", "owa_deu"],
    "DHBT_LAG_TABLE":       [False, True],
    # LAG_MAX bị CHẶN TRÊN bởi giá đỡ của MF_LAG (= 3,0). Đặt cao hơn thì
    # không luật nào kích hoạt ở LAG > 3 và điều kiện MF-4 vỡ — xem
    # `coupling_check()`. Hai hằng số này được ghi trong đặc tả như thể độc
    # lập, thực tế là ràng buộc. Vùng quét vì vậy chỉ tới 3,0.
    "LAG_MAX":              [2.0, 2.5, 3.0],
    "MISS_THRESHOLD_DOM":   [0.20, 0.30, 0.50],
    "DHBT_ZERO_AS_MISSING": [True, False],
    "AGG_POINT_ESTIMATE":   ["mean", "mid"],
    "SAFETY_FLOOR":         [55.0, 65.0, 75.0],
    "FUSION_WEIGHTS":       ["chuan", "deu", "luat_troi"],
    "FKNN_K":               [1, 3, 5, 7],
    "FKNN_M":               [1.5, 2.0, 2.5],
    "FCM_C":                [2, 3, 4, 5],
    "FCM_M":                [1.5, 2.0, 2.5],
    "GIA_LAP_A":            [0, 1, 2],      # trục giả lập — không nối vào gì
    "GIA_LAP_B":            [0, 1],         # trục giả lập — không nối vào gì
}
AXIS_NAMES = list(AXES)
SHAM_AXES = ["GIA_LAP_A", "GIA_LAP_B"]

# Nhãn tiếng Việt cho bảng và hình của bài P12.
AXIS_LABEL = {
    "T_NORM": "T_NORM — phép AND mờ",
    "S_NORM": "S_NORM — phép OR mờ",
    "DEFUZZ": "DEFUZZ — phép giải mờ",
    "MF_SET": "MF_D / MF_LAG — bộ hàm thuộc đầu vào",
    "SHAPLEY": "SHAPLEY — trọng số lĩnh vực",
    "INTERACTION": "INTERACTION — chỉ số tương tác",
    "AGG_OP": "Toán tử tổng hợp / OWA_WEIGHTS",
    "DHBT_LAG_TABLE": "DHBT_USE_LAG_TABLE — bảng luật cho DHBT",
    "LAG_MAX": "LAG_MAX — trần chỉ số trễ",
    "MISS_THRESHOLD_DOM": "MISS_THRESHOLD_DOM — ngưỡng bỏ trống",
    "DHBT_ZERO_AS_MISSING": "DHBT_ZERO_AS_MISSING",
    "AGG_POINT_ESTIMATE": "AGG_POINT_ESTIMATE — ước lượng điểm",
    "SAFETY_FLOOR": "SAFETY_FLOOR — sàn lưới an toàn",
    "FUSION_WEIGHTS": "FUSION_WEIGHTS — trọng số hợp nhất",
    "FKNN_K": "FKNN_K — số láng giềng",
    "FKNN_M": "FKNN_M — số mũ mờ k-NN",
    "FCM_C": "FCM_C — số cụm",
    "FCM_M": "FCM_M — số mũ mờ FCM",
    "GIA_LAP_A": "GIA_LAP_A — trục giả lập (đối chứng âm)",
    "GIA_LAP_B": "GIA_LAP_B — trục giả lập (đối chứng âm)",
}

BASELINE = {
    "T_NORM": "product", "S_NORM": "bounded_sum", "DEFUZZ": "centroid",
    "MF_SET": "v2", "SHAPLEY": "v2", "INTERACTION": "rong", "AGG_OP": "choquet",
    "DHBT_LAG_TABLE": False, "LAG_MAX": 3.0, "MISS_THRESHOLD_DOM": 0.30,
    "DHBT_ZERO_AS_MISSING": True, "AGG_POINT_ESTIMATE": "mean",
    "SAFETY_FLOOR": 65.0, "FUSION_WEIGHTS": "chuan",
    "FKNN_K": 3, "FKNN_M": 2.0, "FCM_C": 4, "FCM_M": 2.0,
    "GIA_LAP_A": 0, "GIA_LAP_B": 0,
}

# Kích thước lưới đầy đủ — con số phải khai báo theo §11.5.
N_GRID_FULL = 1
for _lv in AXES.values():
    N_GRID_FULL *= len(_lv)

# =====================================================================
# SÀNG LỌC GIẢI TÍCH — hằng số KHÔNG THỂ tác động tới AUC
# =====================================================================
INERT_BY_CONSTRUCTION = [
    ("UNC_BETA", "chỉ vào nửa bề rộng khoảng bất định (§7.3); điểm s không đổi "
                 "nên thứ tự xếp hạng không đổi, AUC không đổi"),
    ("UNC_GAMMA", "như trên"),
    ("UNC_W_MAX", "như trên"),
    ("REL_W_MISS / REL_W_INCONS / REL_W_SPEED",
     "chỉ vào chỉ số tin cậy, chỉ số này chỉ vào bề rộng khoảng bất định"),
    ("CONF_THRESHOLD", "chỉ chia tầng CHẮC CHẮN / RANH GIỚI sau khi đã có điểm"),
    ("REL_WARN_THRESHOLD", "chỉ bật cảnh báo hiển thị"),
    ("MF_OUTPUT / OUTPUT_LEVEL_ORDER",
     "chỉ gán nhãn ngôn ngữ cho điểm đã tính; ánh xạ đơn điệu theo điểm"),
    ("MEASURE_MC_SAMPLES",
     "chỉ dùng cho kiểm định đơn điệu, không nằm trên đường tính điểm"),
    ("WILSON_Z", "chỉ vào bề rộng khoảng tin cậy khi báo cáo"),
    ("DEFUZZ", "KHÔNG được đọc ở bất kỳ đâu trong mã nguồn: ts_fuzzy.infer gọi "
               "thẳng centroid. Hằng số chết — được đưa vào thí nghiệm bằng "
               "cách thay hàm giải mờ, xem phần đầu tệp"),
    ("FIS_BYPASS_WHEN_NO_LAG",
     "KHÔNG được đọc ở bất kỳ đâu trong mã nguồn; chỉ xuất hiện trong chuỗi "
     "tài liệu. Hằng số chết"),
]

# =====================================================================
# TRẠNG THÁI CỦA TIẾN TRÌNH CON
# =====================================================================
_STATE = {}


def _apply(levels):
    """Đặt config về đúng cấu hình được yêu cầu."""
    cfg.T_NORM = levels["T_NORM"]
    cfg.S_NORM = levels["S_NORM"]
    cfg.MF_D = cfg.MF_D_V1 if levels["MF_SET"] == "v1" else _STATE["MF_D0"]
    cfg.MF_LAG = cfg.MF_LAG_V1 if levels["MF_SET"] == "v1" else _STATE["MF_LAG0"]
    cfg.SHAPLEY = (cfg.SHAPLEY_V1 if levels["SHAPLEY"] == "v1"
                   else _STATE["SHAPLEY0"])
    cfg.INTERACTION = (cfg.INTERACTION_CLINICAL
                       if levels["INTERACTION"] == "lam_sang" else {})
    cfg.DHBT_USE_LAG_TABLE = bool(levels["DHBT_LAG_TABLE"])
    cfg.LAG_MAX = float(levels["LAG_MAX"])
    cfg.MISS_THRESHOLD_DOM = float(levels["MISS_THRESHOLD_DOM"])
    cfg.DHBT_ZERO_AS_MISSING = bool(levels["DHBT_ZERO_AS_MISSING"])
    cfg.AGG_POINT_ESTIMATE = levels["AGG_POINT_ESTIMATE"]
    cfg.SAFETY_FLOOR = float(levels["SAFETY_FLOOR"])
    cfg.FUSION_WEIGHTS = dict(FUSION_PROFILE[levels["FUSION_WEIGHTS"]])
    op = levels["AGG_OP"]
    if op.startswith("owa_"):
        cfg.OWA_WEIGHTS = list(OWA_PROFILE[op[4:]])
    else:
        cfg.OWA_WEIGHTS = list(OWA_PROFILE["doc"])
    ts_fuzzy.centroid = DEFUZZ_FN[levels["DEFUZZ"]]


def _fknn_batch(A, ytr, B, w, k, m):
    """Fuzzy k-NN của Keller (1985), vector hoá theo lô.

    Sao đúng đường nhanh của ts_fknn._neighbors + predict_memberships: khoảng
    cách Euclid có trọng số Shapley đã chuẩn hoá, k láng giềng gần nhất, trọng
    số D^(-2/(m-1)), trùng khít thì gán thẳng nhãn của điểm đó.
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    a2 = (A * A * w).sum(1)
    exponent = -2.0 / (m - 1.0)
    out = np.empty(len(B), dtype=float)
    for lo in range(0, len(B), 512):
        Bc = B[lo:lo + 512]
        d2 = ((Bc * Bc * w).sum(1)[:, None] + a2[None, :]
              - 2.0 * (Bc * w) @ A.T)
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2)
        kk = min(int(k), dist.shape[1])
        idx = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
        dk = np.take_along_axis(dist, idx, axis=1)
        yk = ytr[idx]
        exact = dk <= 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            wt = np.where(exact, 1.0, dk) ** exponent
        wt = np.where(exact, 0.0, wt)
        wt = np.where(np.isfinite(wt), wt, 0.0)
        u = (wt * yk).sum(1) / np.maximum(wt.sum(1), 1e-300)
        has_exact = exact.any(1)
        if has_exact.any():
            first = np.argmax(exact, axis=1)
            u = np.where(has_exact, yk[np.arange(len(yk)), first], u)
        out[lo:lo + 512] = u
    return 100.0 * out


def _build_cohort(seed):
    """Đoàn hệ CỐ ĐỊNH: hồ sơ chấm được đủ 6 lĩnh vực dưới cấu hình NGẶT NHẤT
    (ngưỡng bỏ trống 0,20). Mọi cấu hình lỏng hơn vẫn chấm được đúng các hồ sơ
    này, nên n không đổi giữa các cấu hình và AUC so sánh được với nhau."""
    q = _STATE["questions"]
    recs, _ = generate_dataset(N_RAW, q, seed=seed)
    _apply({**BASELINE, "MISS_THRESHOLD_DOM": 0.20})
    fis = build_all_fis()
    keep = []
    for r in recs:
        try:
            fv = build_feature_vector(r, q)
        except Exception:
            continue
        risk = infer_all_domains(fv, fis)
        if all(risk.get(d) is not None for d in cfg.DOMAINS):
            keep.append(r)
    return keep


def _install_cohort(records):
    _STATE["cohort"] = records
    rng = np.random.default_rng(SEED_SPLIT)
    order = rng.permutation(len(records))
    half = len(records) // 2
    _STATE["itr"] = np.sort(order[:half])
    _STATE["ite"] = np.sort(order[half:])
    _STATE["y"] = np.array([r.label for r in records], dtype=float)
    q = _STATE["questions"]
    _STATE["trig"] = np.array(
        [bool(byt_threshold_met(records[i], q)[0]) for i in _STATE["ite"]])


def no_power_throttling():
    """Windows 11 (EcoQoS) hạ xung tiến trình chạy nền: tiến trình con chỉ được
    ~20 % một lõi. Tự xin ra khỏi chế độ đó cho CHÍNH tiến trình này bằng
    SetProcessInformation(ProcessPowerThrottling) — không đổi thiết lập hệ
    thống; trên hệ điều hành khác là no-op. Không ảnh hưởng tới kết quả."""
    if os.name != "nt":
        return
    try:
        import ctypes

        class _PPTS(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                        ("StateMask", ctypes.c_ulong)]
        st = _PPTS(1, 0x1, 0x0)       # EXECUTION_SPEED: kiểm soát, trạng thái = tắt
        k32 = ctypes.windll.kernel32
        # Phải khai báo kiểu: thiếu argtypes/restype thì handle 64 bit bị cắt
        # và lời gọi không có tác dụng (đã đo: 27 % → 100 % một lõi).
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p, ctypes.c_ulong]
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4,   # ProcessPowerThrottling
                                  ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass


def init_worker():
    """Mỗi tiến trình con dựng đoàn hệ đúng một lần."""
    no_power_throttling()
    _STATE["MF_D0"] = cfg.MF_D
    _STATE["MF_LAG0"] = cfg.MF_LAG
    _STATE["SHAPLEY0"] = cfg.SHAPLEY
    _STATE["questions"] = load_questions()
    _install_cohort(_build_cohort(SEED_COHORT))


def score_cohort(levels):
    """Chạy toàn bộ đường tính điểm cho một cấu hình → điểm cuối của tập kiểm thử."""
    q = _STATE["questions"]
    recs = _STATE["cohort"]
    _apply(levels)
    fis = build_all_fis()

    xs = []
    for r in recs:
        fv = build_feature_vector(r, q)
        risk = infer_all_domains(fv, fis)
        xs.append(risk_to_unit(
            {d: iv for d, iv in risk.items() if iv is not None}, "point"))

    op = "owa" if levels["AGG_OP"].startswith("owa_") else "choquet"
    s_rule = np.array([aggregate_rule_branch(x, operator=op) for x in xs])

    doms = list(cfg.DOMAINS)
    M = np.array([[x.get(d, 0.0) for d in doms] for x in xs], dtype=float)
    itr, ite, y = _STATE["itr"], _STATE["ite"], _STATE["y"]

    w = np.array([cfg.SHAPLEY[d] for d in doms], dtype=float)
    s_knn = _fknn_batch(M[itr], y[itr], M[ite], w,
                        int(levels["FKNN_K"]), float(levels["FKNN_M"]))

    fcm = FuzzyCMeans(c=int(levels["FCM_C"]), m=float(levels["FCM_M"]),
                      domains=doms, seed=SEED_SPLIT)
    fcm.fit(M[itr])
    fcm.label_clusters([xs[i] for i in itr], [int(v) for v in y[itr]])
    s_fcm = np.array([fcm.score(xs[i]) for i in ite])

    raw = np.empty(len(ite), dtype=float)
    net = np.empty(len(ite), dtype=float)
    for j, i in enumerate(ite):
        s, _ = fuse({"rule": float(s_rule[i]),
                     "fknn": float(s_knn[j]),
                     "fcm": float(s_fcm[j])})
        raw[j] = s
        net[j] = apply_safety_net(s, recs[i], q)[0]
    return raw, net


def eval_config(levels):
    """Ba tiêu chí đánh giá, cùng một cấu hình.

    E1  auc_raw    — AUC của điểm hợp nhất TRƯỚC lưới an toàn. Đây là đại lượng
                     mà các tham số thật sự điều khiển.
    E2  auc_net    — AUC của điểm hệ thống THỰC SỰ trả ra, sau khi lưới an toàn
                     kẹp sàn. Đây là thứ người dùng nhận được.
    E3  auc_trong  — AUC TRONG NHÓM đã chạm ngưỡng BYT. Đây là bài toán phân
                     luồng của P1: mọi trẻ trong nhóm đều được theo dõi, câu hỏi
                     chỉ là khám ai trước. Là tiêu chí duy nhất trong ba tiêu chí
                     vừa có ý nghĩa vận hành vừa không bị lưới an toàn làm phẳng.
    """
    raw, net = score_cohort(levels)
    y = _STATE["y"][_STATE["ite"]].astype(int)
    trig = _STATE["trig"]
    a_raw = roc_auc(y.tolist(), raw.tolist())
    a_net = roc_auc(y.tolist(), net.tolist())
    yt = y[trig]
    a_in = roc_auc(yt.tolist(), raw[trig].tolist()) if 0 < yt.sum() < len(yt) else None
    return (float(a_raw),
            float(a_net) if a_net is not None else None,
            float(a_in) if a_in is not None else None)


def _job(payload):
    i, levels = payload
    try:
        return i, eval_config(levels), None
    except Exception as e:                       # ghi lại, không nuốt lỗi
        return i, None, "{}: {}".format(type(e).__name__, e)


# =====================================================================
# ĐO ĐỘ PHỦ — tiêu chí đúng cho hai tham số điều khiển tính HỢP LỆ của hồ sơ
# =====================================================================
def coverage_scan():
    """MISS_THRESHOLD_DOM và DHBT_ZERO_AS_MISSING không đổi thứ tự xếp hạng của
    các hồ sơ CHẤM ĐƯỢC; chúng đổi việc hồ sơ nào chấm được. Vì đoàn hệ đánh giá
    được chốt theo cấu hình ngặt nhất để AUC so sánh được, hai tham số này bị
    trung hoà trên ba tiêu chí AUC — phải đo trên tiêu chí đúng của chúng là
    ĐỘ PHỦ: tỷ lệ hồ sơ thô chấm được đủ sáu lĩnh vực."""
    q = _STATE["questions"]
    recs, _ = generate_dataset(N_RAW, q, seed=SEED_COHORT)
    out = []
    for thr in AXES["MISS_THRESHOLD_DOM"]:
        for zero in AXES["DHBT_ZERO_AS_MISSING"]:
            _apply({**BASELINE, "MISS_THRESHOLD_DOM": thr,
                    "DHBT_ZERO_AS_MISSING": zero})
            fis = build_all_fis()
            n_ok = n_full = 0
            for r in recs:
                try:
                    fv = build_feature_vector(r, q)
                except Exception:
                    continue
                risk = infer_all_domains(fv, fis)
                got = [d for d in cfg.DOMAINS if risk.get(d) is not None]
                if got:
                    n_ok += 1
                if len(got) == len(cfg.DOMAINS):
                    n_full += 1
            out.append({"MISS_THRESHOLD_DOM": thr, "DHBT_ZERO_AS_MISSING": zero,
                        "n_raw": len(recs), "n_scorable": n_ok, "n_full": n_full,
                        "ty_le_cham_duoc": n_ok / len(recs),
                        "ty_le_du_6_linh_vuc": n_full / len(recs)})
    return out


# =====================================================================
# KIỂM TRA RÀNG BUỘC GIỮA CÁC HẰNG SỐ
# =====================================================================
def coupling_check():
    """LAG_MAX và giá đỡ của MF_LAG được ghi trong đặc tả như hai hằng số độc
    lập. Chúng không độc lập: MF_LAG phủ đúng đoạn [0; 3], nên mọi LAG_MAX > 3
    tạo ra một vùng đầu vào mà KHÔNG luật nào kích hoạt — điều kiện MF-4 vỡ và
    FIS lớp 1 ném lỗi thay vì trả về một điểm số.

    Đây là dạng lỗi mà một lát cắt một chiều quanh giá trị mặc định không bao
    giờ gặp, vì lát cắt đó không đi ra khỏi vùng an toàn."""
    out = []
    for v in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        try:
            score_cohort({**BASELINE, "LAG_MAX": v})
            out.append({"LAG_MAX": v, "chay_duoc": True, "loi": None})
        except Exception as e:
            out.append({"LAG_MAX": v, "chay_duoc": False,
                        "loi": "{}: {}".format(type(e).__name__, e)})
    return out


# =====================================================================
# THỐNG KÊ
# =====================================================================
def bootstrap_auc_se(y, s, n_boot=N_BOOT, seed=SEED_BOOT):
    """Bootstrap phân tầng → sai số chuẩn của AUC trên chính đoàn hệ này."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        a = roc_auc(y[i].tolist(), s[i].tolist())
        if a is not None:
            vals.append(a)
    vals = np.asarray(vals)
    return {"se": float(vals.std(ddof=1)),
            "ci95": [float(np.percentile(vals, 2.5)),
                     float(np.percentile(vals, 97.5))]}


def eta_squared(values, codes):
    """Tỷ lệ phương sai giữa các mức trên tổng phương sai — thống kê một chiều."""
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_bet = 0.0
    for lv in np.unique(codes):
        g = values[codes == lv]
        ss_bet += len(g) * (g.mean() - grand) ** 2
    return float(ss_bet / ss_tot)


def permutation_p(values, codes, n_perm=N_PERM, seed=SEED_PERM):
    """H0: mức của trục này không liên quan tới AUC. Hoán vị nhãn mức."""
    rng = np.random.default_rng(seed)
    obs = eta_squared(values, codes)
    c = np.asarray(codes).copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(c)
        if eta_squared(values, c) >= obs - 1e-15:
            ge += 1
    return obs, float((1 + ge) / (1 + n_perm))


def benjamini_hochberg(pvals, q=FDR_Q):
    """T17 — thủ tục BH, trả về ngưỡng và cờ bác bỏ theo thứ tự gốc."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = np.flatnonzero(ranked <= thresh)
    k = int(below[-1]) + 1 if len(below) else 0
    crit = float(ranked[k - 1]) if k else 0.0
    rej = np.zeros(m, dtype=bool)
    if k:
        rej[order[:k]] = True
    # giá trị p đã hiệu chỉnh (BH step-up, ép đơn điệu)
    adj = np.empty(m, dtype=float)
    run = 1.0
    for i in range(m - 1, -1, -1):
        run = min(run, ranked[i] * m / (i + 1))
        adj[order[i]] = run
    return {"k_bac_bo": k, "nguong_toi_han": crit,
            "bac_bo": rej.tolist(), "p_hieu_chinh": adj.tolist()}


def sobol_on_surrogate(codes, values, seed=SEED_SURROGATE, n_mc=N_MC_SOBOL):
    """T15 — fANOVA/Sobol trên mô hình thay thế rừng ngẫu nhiên (Hutter và cs.).

    Ước lượng Saltelli–Jansen cho chỉ số bậc nhất S1 và bậc toàn phần ST. Mọi
    trục đều rời rạc nên mã hoá one-hot; R² ngoài túi của rừng được báo cáo kèm
    theo, vì nếu mô hình thay thế không dự báo được thì phép phân rã vô nghĩa.
    """
    from sklearn.ensemble import RandomForestRegressor

    d = codes.shape[1]
    n_lv = np.array([len(AXES[a]) for a in AXIS_NAMES])
    onehot_cols = []
    for j in range(d):
        for lv in range(n_lv[j]):
            onehot_cols.append((codes[:, j] == lv).astype(float))
    X = np.column_stack(onehot_cols)

    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                               oob_score=True, bootstrap=True,
                               random_state=seed, n_jobs=-1)
    rf.fit(X, values)
    oob_r2 = float(rf.oob_score_)

    rng = np.random.default_rng(seed)

    def encode(C):
        cols = []
        for j in range(d):
            for lv in range(n_lv[j]):
                cols.append((C[:, j] == lv).astype(float))
        return np.column_stack(cols)

    def f(C):
        return rf.predict(encode(C))

    A = np.column_stack([rng.integers(0, n_lv[j], n_mc) for j in range(d)])
    B = np.column_stack([rng.integers(0, n_lv[j], n_mc) for j in range(d)])
    fA, fB = f(A), f(B)
    var = float(np.var(np.concatenate([fA, fB])))
    S1, ST = [], []
    for j in range(d):
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = f(AB)
        S1.append(float(np.mean(fB * (fAB - fA)) / var) if var > 0 else 0.0)
        ST.append(float(np.mean((fA - fAB) ** 2) / (2.0 * var)) if var > 0 else 0.0)
    return {"oob_r2": oob_r2, "phuong_sai_thay_the": var,
            "S1": S1, "ST": ST}


# =====================================================================
# CHƯƠNG TRÌNH CHÍNH
# =====================================================================
def sample_design(n_config, seed=SEED_DESIGN):
    """Lấy mẫu trên lưới đầy đủ, phân tầng theo từng trục để mỗi mức xuất hiện
    xấp xỉ đều. Thiết kế cân bằng gần đúng là điều kiện để kiểm định một chiều
    không bị phồng phương sai giữa các mức."""
    rng = np.random.default_rng(seed)
    cols = []
    for a in AXIS_NAMES:
        k = len(AXES[a])
        reps = int(np.ceil(n_config / k))
        col = np.tile(np.arange(k), reps)[:n_config]
        rng.shuffle(col)
        cols.append(col)
    codes = np.column_stack(cols)
    levels = [{a: AXES[a][int(codes[i, j])] for j, a in enumerate(AXIS_NAMES)}
              for i in range(n_config)]
    return codes, levels


def vn(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return ("{:." + str(nd) + "f}").format(x).replace(".", ",")


def main():
    no_power_throttling()
    t_start = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say("=" * 96)
    say("P12 — PHÂN TÍCH ĐỘ NHẠY TOÀN CỤC TRÊN KHÔNG GIAN THAM SỐ CỦA config.py")
    say("=" * 96)

    init_worker()
    n_co = len(_STATE["cohort"])
    ite = _STATE["ite"]
    y_te = _STATE["y"][ite].astype(int)
    trig = _STATE["trig"]
    say("Đoàn hệ D4            : {} hồ sơ sinh ra → {} chấm được đủ 6 lĩnh vực"
        .format(N_RAW, n_co))
    say("Chia đôi              : huấn luyện {} · kiểm thử {}"
        .format(len(_STATE["itr"]), len(ite)))
    say("Tỷ lệ dương (kiểm thử): {:.4f}  ({} ca)"
        .format(y_te.mean(), int(y_te.sum())))
    say("Chạm ngưỡng BYT       : {:.4f}  ({} ca)"
        .format(trig.mean(), int(trig.sum())))
    say("Lưới đầy đủ           : {:,} cấu hình  ({} trục)"
        .format(N_GRID_FULL, len(AXIS_NAMES)))
    say("Số cấu hình đã thử    : {}   (khai báo bắt buộc theo §11.5)"
        .format(N_CONFIG))
    say("Hạt giống             : đoàn hệ {} · thiết kế {} · chia {} · "
        "bootstrap {} · hoán vị {} · thay thế {}"
        .format(SEED_COHORT, SEED_DESIGN, SEED_SPLIT, SEED_BOOT,
                SEED_PERM, SEED_SURROGATE))

    # --- ngưỡng nhiễu của phép đo ------------------------------------
    raw0, net0 = score_cohort(BASELINE)
    a_raw0 = float(roc_auc(y_te.tolist(), raw0.tolist()))
    a_net0 = float(roc_auc(y_te.tolist(), net0.tolist()))
    a_in0 = float(roc_auc(y_te[trig].tolist(), raw0[trig].tolist()))
    boot_raw = bootstrap_auc_se(y_te, raw0)
    boot_in = bootstrap_auc_se(y_te[trig], raw0[trig])

    say()
    say("-" * 96)
    say("BƯỚC 0 — NGƯỠNG NHIỄU CỦA PHÉP ĐO (cấu hình mặc định)")
    say("-" * 96)
    say("  E1 AUC trước lưới an toàn  : {}   SE bootstrap {}   KTC95 [{}; {}]"
        .format(vn(a_raw0), vn(boot_raw["se"]),
                vn(boot_raw["ci95"][0]), vn(boot_raw["ci95"][1])))
    say("  E2 AUC sau lưới an toàn    : {}".format(vn(a_net0)))
    say("  E3 AUC trong nhóm chạm BYT : {}   SE bootstrap {}   KTC95 [{}; {}]"
        .format(vn(a_in0), vn(boot_in["se"]),
                vn(boot_in["ci95"][0]), vn(boot_in["ci95"][1])))

    # --- độ lệch giữa các hạt giống sinh dữ liệu ---------------------
    say()
    say("  Lặp lại với hạt giống sinh dữ liệu khác (cùng cấu hình mặc định):")
    rep = []
    for sd in ([] if CORE_ONLY else SEED_REPLICA):
        _install_cohort(_build_cohort(sd))
        r, _n = score_cohort(BASELINE)
        yy = _STATE["y"][_STATE["ite"]].astype(int)
        tt = _STATE["trig"]
        rep.append({"seed": sd,
                    "n": len(_STATE["cohort"]),
                    "E1": float(roc_auc(yy.tolist(), r.tolist())),
                    "E3": float(roc_auc(yy[tt].tolist(), r[tt].tolist()))})
        say("    hạt giống {}: n = {}   E1 = {}   E3 = {}"
            .format(sd, rep[-1]["n"], vn(rep[-1]["E1"]), vn(rep[-1]["E3"])))
    sd_e1 = float(np.std([x["E1"] for x in rep], ddof=1)) if len(rep) > 1 else None
    sd_e3 = float(np.std([x["E3"] for x in rep], ddof=1)) if len(rep) > 1 else None
    say("    → độ lệch chuẩn giữa hạt giống: E1 {} · E3 {}"
        .format(vn(sd_e1), vn(sd_e3)))
    if rep:
        _install_cohort(_build_cohort(SEED_COHORT))     # trả lại đoàn hệ chính

    # --- quét lưới ---------------------------------------------------
    codes, levels = sample_design(N_CONFIG)
    say()
    say("-" * 96)
    say("BƯỚC 1 — QUÉT {} CẤU HÌNH".format(N_CONFIG))
    say("-" * 96)
    t0 = time.time()
    res = [None] * N_CONFIG
    errs = []
    n_workers = int(os.environ.get("SENS_WORKERS", "0")) or max(1, min(10, (os.cpu_count() or 4) - 2))
    # Bộ đệm theo cấu hình: tiến trình con chết giữa chừng (BrokenProcessPool)
    # thì chạy lại chỉ tính phần còn thiếu. Khoá bộ đệm gồm N_RAW, hạt giống
    # đoàn hệ/thiết kế và N_CONFIG, nên không lẫn giữa các lần quét.
    cache_p = os.path.join(cfg.REPORTS_DIR, "_sens_cache",
                           "n{}_c{}_d{}_k{}{}.jsonl".format(
                               N_RAW, SEED_COHORT, SEED_DESIGN, N_CONFIG, GEN_TAG))
    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
    if os.path.exists(cache_p):
        with open(cache_p, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["levels"] == {a: str(v) for a, v in levels[row["i"]].items()}:
                    res[row["i"]] = tuple(row["val"])
    todo = [i for i in range(N_CONFIG) if res[i] is None]
    say("  Bộ đệm: {} cấu hình đã có, còn {} cần chạy"
        .format(N_CONFIG - len(todo), len(todo)))
    # Tiến trình con trên Windows đôi khi chết đột ngột (WinError 6 ở hàng đợi,
    # BrokenProcessPool). Kết quả là hàm tất định của cấu hình và hạt giống, nên
    # dựng lại bể và chạy tiếp phần còn thiếu không làm đổi con số nào.
    from concurrent.futures import as_completed
    from concurrent.futures.process import BrokenProcessPool
    attempt = 0
    while todo and attempt < 30:
        attempt += 1
        try:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=init_worker) as ex, \
                    open(cache_p, "a", encoding="utf-8") as fc:
                futs = [ex.submit(_job, (i, levels[i])) for i in todo]
                for fu in as_completed(futs):
                    i, val, err = fu.result()
                    res[i] = val
                    if err:
                        errs.append((i, err))
                    else:
                        fc.write(json.dumps({"i": i, "val": list(val), "levels": {
                            a: str(v) for a, v in levels[i].items()}}) + "\n")
                        fc.flush()
                    n_done = sum(r is not None for r in res)
                    if n_done % 32 == 0:
                        say("    {}/{} cấu hình  ({:.0f} s)"
                            .format(n_done, N_CONFIG, time.time() - t0))
        except BrokenProcessPool as e:
            say("  ⚠ bể tiến trình vỡ ({}); lần thử {} — chạy tiếp từ bộ đệm"
                .format(type(e).__name__, attempt))
        todo = [i for i in range(N_CONFIG)
                if res[i] is None and i not in {j for j, _ in errs}]
    say("  Xong trong {:.0f} s trên {} tiến trình; {} cấu hình lỗi"
        .format(time.time() - t0, n_workers, len(errs)))
    for i, e in errs[:5]:
        say("    lỗi ở cấu hình {}: {}".format(i, e))

    ok = np.array([r is not None for r in res])
    E1 = np.array([r[0] if r else np.nan for r in res])
    E2 = np.array([r[1] if r else np.nan for r in res])
    E3 = np.array([r[2] if r else np.nan for r in res])
    for tag, arr in (("E1", E1), ("E2", E2), ("E3", E3)):
        say("  {} trên {} cấu hình: nhỏ nhất {} · lớn nhất {} · biên độ {} · "
            "độ lệch chuẩn {}"
            .format(tag, int(ok.sum()), vn(np.nanmin(arr)), vn(np.nanmax(arr)),
                    vn(np.nanmax(arr) - np.nanmin(arr)),
                    vn(np.nanstd(arr, ddof=1))))

    # --- phân tích từng tiêu chí -------------------------------------
    analyses = {}
    for tag, vals, se_ref in (("E1", E1, boot_raw["se"]),
                              ("E2", E2, boot_raw["se"]),
                              ("E3", E3, boot_in["se"])):
        m = ok & np.isfinite(vals)
        v = vals[m]
        cd = codes[m]
        rows = []
        for j, a in enumerate(AXIS_NAMES):
            eta, p = permutation_p(v, cd[:, j])
            means = [float(v[cd[:, j] == lv].mean()) if (cd[:, j] == lv).any()
                     else float("nan") for lv in range(len(AXES[a]))]
            amp = float(np.nanmax(means) - np.nanmin(means))
            rows.append({"truc": a, "nhan": AXIS_LABEL[a],
                         "muc": [str(x) for x in AXES[a]],
                         "trung_binh_theo_muc": means,
                         "bien_do": amp, "eta2": eta, "p": p,
                         "so_lan_SE": (amp / se_ref if se_ref > 0
                                       else float("inf")),
                         "gia_lap": a in SHAM_AXES})
        bh = benjamini_hochberg([r["p"] for r in rows])
        for r, adj, rej in zip(rows, bh["p_hieu_chinh"], bh["bac_bo"]):
            r["p_BH"] = adj
            r["y_nghia_BH"] = bool(rej)
        sob = sobol_on_surrogate(cd, v)
        for j, r in enumerate(rows):
            r["S1"] = sob["S1"][j]
            r["ST"] = sob["ST"][j]
        analyses[tag] = {
            "n_cau_hinh": int(m.sum()),
            "se_tham_chieu": se_ref,
            "auc_mac_dinh": {"E1": a_raw0, "E2": a_net0, "E3": a_in0}[tag],
            "bien_do_toan_cuc": float(v.max() - v.min()),
            "do_lech_chuan": float(v.std(ddof=1)),
            "BH": {"k_bac_bo": bh["k_bac_bo"],
                   "nguong_toi_han": bh["nguong_toi_han"]},
            "surrogate": {"oob_r2": sob["oob_r2"],
                          "phuong_sai": sob["phuong_sai_thay_the"]},
            "truc": rows}

    for tag, title in (("E1", "E1 — AUC TRƯỚC LƯỚI AN TOÀN"),
                       ("E2", "E2 — AUC SAU LƯỚI AN TOÀN (điểm hệ thống trả ra)"),
                       ("E3", "E3 — AUC TRONG NHÓM ĐÃ CHẠM NGƯỠNG BYT")):
        A = analyses[tag]
        say()
        say("-" * 96)
        say("BƯỚC 2{} — {}".format(tag, title))
        say("-" * 96)
        say("  R² ngoài túi của mô hình thay thế: {}   "
            "(thấp → phép phân rã phương sai không đáng tin)"
            .format(vn(A["surrogate"]["oob_r2"])))
        say("  SE tham chiếu của phép đo: {}".format(vn(A["se_tham_chieu"])))
        say("  Số trục vượt được BH: {}/{}"
            .format(A["BH"]["k_bac_bo"], len(AXIS_NAMES)))
        say()
        say("  {:<44}{:>8}{:>8}{:>10}{:>7}{:>9}"
            .format("Trục", "ST", "S1", "biên độ", "/SE", "p BH"))
        say("  " + "-" * 88)
        for r in sorted(A["truc"], key=lambda r: -r["ST"]):
            mark = "  «giả lập" if r["gia_lap"] else ("  *" if r["y_nghia_BH"]
                                                     else "")
            say("  {:<44}{:>8}{:>8}{:>10}{:>7.1f}{:>9}{}"
                .format(r["nhan"][:43], vn(r["ST"], 3), vn(r["S1"], 3),
                        vn(r["bien_do"]), r["so_lan_SE"], vn(r["p_BH"], 4),
                        mark))
        say("  (* = có ý nghĩa sau hiệu chỉnh Benjamini–Hochberg, FDR = 0,05)")

    # --- lát cắt 1 chiều ---------------------------------------------
    say()
    say("-" * 96)
    say("BƯỚC 3 — LÁT CẮT MỘT CHIỀU QUANH CẤU HÌNH MẶC ĐỊNH")
    say("-" * 96)
    top = [r["truc"] for r in sorted(analyses["E1"]["truc"],
                                     key=lambda r: -r["ST"])
           if not r["gia_lap"]][:4]
    slices = {}
    for a in ([] if CORE_ONLY else top + SHAM_AXES):
        row = []
        for lv in AXES[a]:
            c = dict(BASELINE)
            c[a] = lv
            raw, net = score_cohort(c)
            row.append({
                "muc": str(lv),
                "E1": float(roc_auc(y_te.tolist(), raw.tolist())),
                "E2": float(roc_auc(y_te.tolist(), net.tolist())),
                "E3": float(roc_auc(y_te[trig].tolist(),
                                    raw[trig].tolist()))})
        slices[a] = row
        amp1 = max(r["E1"] for r in row) - min(r["E1"] for r in row)
        say("  {:<46}{}".format(
            AXIS_LABEL[a][:44],
            " · ".join("{}={}".format(r["muc"], vn(r["E1"])) for r in row)))
        say("  {:<46}biên độ E1 = {}  = {:.1f}× SE"
            .format("", vn(amp1), amp1 / boot_raw["se"]))

    # --- độ phủ -------------------------------------------------------
    say()
    say("-" * 96)
    say("BƯỚC 4 — ĐỘ PHỦ (tiêu chí đúng của hai tham số điều khiển tính hợp lệ)")
    say("-" * 96)
    cov = [] if CORE_ONLY else coverage_scan()
    for c in cov:
        say("  ngưỡng bỏ trống {:.2f} · DHBT_ZERO_AS_MISSING={:<5} → "
            "chấm được {:.4f} · đủ 6 lĩnh vực {:.4f}"
            .format(c["MISS_THRESHOLD_DOM"], str(c["DHBT_ZERO_AS_MISSING"]),
                    c["ty_le_cham_duoc"], c["ty_le_du_6_linh_vuc"]))

    # --- ràng buộc giữa các hằng số -----------------------------------
    say()
    say("-" * 96)
    say("BƯỚC 5 — RÀNG BUỘC GIỮA CÁC HẰNG SỐ (LAG_MAX × giá đỡ của MF_LAG)")
    say("-" * 96)
    coup = [] if CORE_ONLY else coupling_check()
    for c in coup:
        if c["chay_duoc"]:
            say("  LAG_MAX = {:<4} → chạy được".format(c["LAG_MAX"]))
        else:
            say("  LAG_MAX = {:<4} → VỠ: {}".format(c["LAG_MAX"], c["loi"][:78]))
    n_bad = sum(1 for c in coup if not c["chay_duoc"])
    if coup:
        say("  → {}/{} giá trị thử làm vỡ điều kiện MF-4. Trần thật của LAG_MAX "
            "là giá đỡ của MF_LAG (3,0),".format(n_bad, len(coup)))
        say("    không phải một hằng số tự do như đặc tả mô tả.")

    # --- hằng số trơ theo lập luận ------------------------------------
    say()
    say("-" * 96)
    say("BƯỚC 6 — HẰNG SỐ KHÔNG THỂ TÁC ĐỘNG TỚI AUC (loại bằng lập luận)")
    say("-" * 96)
    for name, why in INERT_BY_CONSTRUCTION:
        say("  {:<38} {}".format(name, why))

    # --- ghi tệp -------------------------------------------------------
    out = {
        "meta": {
            "n_raw": N_RAW, "n_cohort": n_co, "core_only": CORE_ONLY,
            "generator_overrides": GEN_OVERRIDES,
            "n_pos_test": int(y_te.sum()),
            "n_pos_trig": int(y_te[trig].sum()), "n_trig": int(trig.sum()),
            "n_train": len(_STATE["itr"]), "n_test": len(ite),
            "ty_le_duong": float(y_te.mean()),
            "ty_le_cham_byt": float(trig.mean()),
            "n_config": N_CONFIG, "n_grid_full": N_GRID_FULL,
            "n_axes": len(AXIS_NAMES), "n_sham": len(SHAM_AXES),
            "n_boot": N_BOOT, "n_perm": N_PERM, "n_mc_sobol": N_MC_SOBOL,
            "fdr_q": FDR_Q,
            "seeds": {"cohort": SEED_COHORT, "design": SEED_DESIGN,
                      "split": SEED_SPLIT, "boot": SEED_BOOT,
                      "perm": SEED_PERM, "surrogate": SEED_SURROGATE,
                      "replica": SEED_REPLICA},
            "giay_chay": round(time.time() - t_start, 1),
        },
        "axes": {a: [str(x) for x in AXES[a]] for a in AXIS_NAMES},
        "axis_label": AXIS_LABEL,
        "baseline": {k: str(v) for k, v in BASELINE.items()},
        "auc_mac_dinh": {"E1": a_raw0, "E2": a_net0, "E3": a_in0},
        "nguong_nhieu": {"bootstrap_E1": boot_raw, "bootstrap_E3": boot_in,
                         "sd_giua_hat_giong_E1": sd_e1,
                         "sd_giua_hat_giong_E3": sd_e3,
                         "lap_lai": rep},
        "phan_tich": analyses,
        "lat_cat": slices,
        "do_phu": cov,
        "rang_buoc_hang_so": coup,
        "hang_so_tro": [{"ten": n, "ly_do": w}
                        for n, w in INERT_BY_CONSTRUCTION],
        "nhat_ky_cau_hinh": [{a: str(lv[a]) for a in AXIS_NAMES}
                             for lv in levels],
        "auc_tung_cau_hinh": {"E1": E1.tolist(), "E2": E2.tolist(),
                              "E3": E3.tolist()},
    }
    os.makedirs(cfg.REPORTS_DIR, exist_ok=True)
    pj = os.path.join(cfg.REPORTS_DIR, "sensitivity{}.json".format(OUT_TAG))
    pt = os.path.join(cfg.REPORTS_DIR, "sensitivity{}.txt".format(OUT_TAG))
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    say()
    say("Đã ghi {}".format(pj))
    with open(pt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\nĐã ghi {}\n".format(pt))
    print("Đã ghi {}".format(pt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
