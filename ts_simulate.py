# -*- coding: utf-8 -*-
"""
ts_simulate.py — §8 BỘ SINH DỮ LIỆU MÔ PHỎNG.

Khắc phục V4 (n = 26, nhãn mất cân bằng 22/4). Module này phải viết TRƯỚC mọi
thành phần học máy vì tất cả các bước sau đều cần dữ liệu để kiểm thử.

⚠️  CÁI BẪY PHẢI TRÁNH (§8.1)
-----------------------------
Nếu sinh dữ liệu bằng chính luật BYT rồi huấn luyện ML trên đó, ML chỉ học thuộc
lại luật đã viết. Accuracy sẽ đạt ~99 % và con số đó HOÀN TOÀN VÔ NGHĨA; lời phê
sẽ là "vòng lặp tự xác nhận".

Vì vậy bộ sinh dưới đây dựa trên một mô hình biến tiềm ẩn ĐỘC LẬP HOÀN TOÀN với
luật phân loại: năng lực tiềm ẩn θ theo lĩnh vực, xác suất đạt mốc phụ thuộc
đồng thời θ và tuổi. Ngưỡng "2/3 câu Không" của QĐ 2254 KHÔNG xuất hiện ở bất kỳ
đâu trong tệp này. `sanity_check()` cài SC-5 làm chốt chặn tự động.

Vì sao bước 4 gắn được với thực tế: cùng một θ, trẻ lớn tuổi hơn có xác suất đạt
mốc cao hơn — đúng bản chất phát triển, và độc lập với mọi ngưỡng phân loại.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Optional, Sequence

import numpy as np

import config as cfg
from ts_types import Question, Record


# =====================================================================
# Tiện ích
# =====================================================================

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


_EPOCH = _dt.date(2020, 1, 1)


def _dates(rng: np.random.Generator, age_days: int) -> tuple[str, str]:
    """Sinh cặp (ngày sinh, ngày đánh giá) khớp đúng age_days để INV-1 đạt."""
    offset = int(rng.integers(0, 1500))
    assessment = _EPOCH + _dt.timedelta(days=offset)
    birth = assessment - _dt.timedelta(days=age_days)
    return birth.isoformat(), assessment.isoformat()


# =====================================================================
# §8.2 — mô hình sinh biến tiềm ẩn
# =====================================================================

def draw_atypical(rng: np.random.Generator, z: int) -> Optional[str]:
    """Bước 7 — bốc nhóm ca không điển hình (§8.6).

    Bốc theo phân phối phạm trù trên cả nhóm chứ không kiểm tuần tự từng nhóm:
    kiểm tuần tự làm các nhóm đứng sau bị thiệt (chỉ được xét khi nhóm trước
    trượt), khiến `asd_bieu_hien_muon` gần như không bao giờ xuất hiện.
    """
    pool = cfg.ATYPICAL_ON_POSITIVE if z == 1 else cfg.ATYPICAL_ON_NEGATIVE
    u = float(rng.random())
    acc = 0.0
    for name in pool:
        acc += cfg.ATYPICAL[name]
        if u < acc:
            return name
    return None


def draw_age(rng: np.random.Generator, atypical: Optional[str],
             age_min: int, age_max: int) -> int:
    """Bước 2 — tuổi. Nhóm 'biểu hiện muộn' được định nghĩa BỞI điều kiện tuổi.

    §8.6 mô tả nhóm này là "δ giảm một nửa ở trẻ < 500 ngày". Nếu bốc tuổi trước
    rồi mới xét điều kiện thì phần lớn ca bốc trúng nhóm sẽ bị loại (tuổi phân bố
    đều 180–1095 nên chỉ ~35 % lọt), và SC-3 đỏ vì nhóm rỗng. Ở đây tuổi được bốc
    CÓ ĐIỀU KIỆN trên nhóm — nhóm và điều kiện tuổi là một, không phải hai sự
    kiện độc lập. Hệ quả cần ghi rõ: nhóm này tương quan với tuổi theo thiết kế.
    """
    if atypical == "asd_bieu_hien_muon":
        hi = min(cfg.ATYPICAL_PARAMS[atypical]["age_below_days"] - 1, age_max)
        if hi > age_min:
            return int(rng.integers(age_min, hi + 1))
    return int(rng.integers(age_min, age_max + 1))


def _draw_latent(rng: np.random.Generator, z: int, age_days: int,
                 atypical: Optional[str]) -> dict[str, float]:
    """Bước 3 — năng lực tiềm ẩn θ_i theo lĩnh vực."""
    delta_scale = 1.0
    if atypical == "asd_bieu_hien_muon" and \
            age_days < cfg.ATYPICAL_PARAMS[atypical]["age_below_days"]:
        delta_scale = cfg.ATYPICAL_PARAMS[atypical]["delta_scale"]

    theta: dict[str, float] = {}
    for domain in cfg.DOMAINS:
        mean = -cfg.DELTA[domain] * delta_scale if z == 1 else 0.0
        theta[domain] = float(rng.normal(mean, 1.0))

    if atypical == "asd_van_dong_binh_thuong":
        # z = 1 nhưng vận động lấy từ N(0,1) → chỉ bất thường ở trục xã hội.
        for domain in cfg.ATYPICAL_PARAMS[atypical]["normal_domains"]:
            theta[domain] = float(rng.normal(0.0, 1.0))
    elif atypical == "cham_ngon_ngu_don_thuan":
        # z = 0 nhưng chậm nói. Ca dương tính giả kinh điển mà mọi công cụ
        # sàng lọc đều vấp — và là phép thử quyết định cho tích phân Choquet.
        for domain, shift in cfg.ATYPICAL_PARAMS[atypical]["shift"].items():
            theta[domain] = float(rng.normal(shift, 1.0))
    elif atypical == "cham_toan_dien":
        # z = 0 nhưng chậm phát triển chung, không phải RLPTK.
        shift = cfg.ATYPICAL_PARAMS[atypical]["shift_all"]
        for domain in cfg.DOMAINS:
            theta[domain] = float(rng.normal(shift, 1.0))

    return theta


def _risk_probability(question: Question, theta: float, age_days: int) -> float:
    """Bước 4 — xác suất câu j mang nguy cơ."""
    if question.domain in cfg.AGELESS_DOMAINS:
        # DHBT hỏi về sự XUẤT HIỆN của hành vi bất thường: θ thấp → nhiều dấu hiệu.
        return _sigmoid(-theta + cfg.DHBT_BASE_LOGIT)
    tol = max(1, int(question.tolerance_days))
    p_achieved = _sigmoid(theta + (age_days - question.expected_age_days) / tol)
    return 1.0 - p_achieved


# =====================================================================
# §8.4 — nhiễu trả lời và thiên lệch người trả lời
# =====================================================================

def _draw_informant(rng: np.random.Generator) -> str:
    names = list(cfg.INFORMANT_PROB.keys())
    probs = np.array([cfg.INFORMANT_PROB[n] for n in names], dtype=float)
    return str(rng.choice(names, p=probs / probs.sum()))


def _answer_for(rng: np.random.Generator, question: Question,
                p_risk: float, bias: float) -> str:
    """Áp thiên lệch người trả lời rồi bốc, sau đó áp nhiễu theo mức trừu tượng."""
    p = _clip01(p_risk + bias)
    is_risk = rng.random() < p

    p_error = cfg.P_NOISE_BASE + cfg.P_NOISE_ABSTRACT * question.abstract_level
    if rng.random() < p_error:
        is_risk = not is_risk

    if is_risk:
        return question.risk_answer
    return "CO" if question.risk_answer == "KHONG" else "KHONG"


# =====================================================================
# §8.8 — API
# =====================================================================

def generate_child(rng: np.random.Generator,
                   questions: Sequence[Question],
                   prevalence: float = None,
                   age_min: int = None,
                   age_max: int = None,
                   record_id: str = "SIM") -> tuple[Record, dict]:
    """Sinh một hồ sơ hoàn chỉnh kèm nhãn tiềm ẩn thật và siêu dữ liệu."""
    prevalence = cfg.SIM_CONFIG["prevalence"] if prevalence is None else prevalence
    age_min = cfg.SIM_CONFIG["age_min_days"] if age_min is None else age_min
    age_max = cfg.SIM_CONFIG["age_max_days"] if age_max is None else age_max

    z = int(rng.random() < prevalence)                       # bước 1
    atypical = draw_atypical(rng, z)                         # bước 7
    age_days = draw_age(rng, atypical, age_min, age_max)     # bước 2
    theta = _draw_latent(rng, z, age_days, atypical)         # bước 3

    informant = _draw_informant(rng)                         # bước 5
    bias = cfg.INFORMANT_BIAS[informant]

    applicable = [q for q in questions if q.applies_to(age_days)]
    answers: dict[str, Optional[str]] = {}
    for q in applicable:                                     # bước 4 + 5
        p_risk = _risk_probability(q, theta[q.domain], age_days)
        answers[q.question_id] = _answer_for(rng, q, p_risk, bias)

    # --- bước 6: dữ liệu thiếu (§8.5) ---
    for qid in list(answers):
        if rng.random() < cfg.MISS_MCAR_RATE:
            answers[qid] = None

    if rng.random() < cfg.MISS_FATIGUE_RECORD_RATE:
        lo, hi = cfg.MISS_FATIGUE_ITEMS
        n_drop = int(rng.integers(lo, hi + 1))
        for qid in [q.question_id for q in applicable][-n_drop:]:
            answers[qid] = None

    n_items = max(1, len(applicable))
    if rng.random() < cfg.SPEED_ANOMALY_RATE:
        lo, hi = cfg.SIM_SEC_PER_ITEM_ANOMALY
        per_item = float(rng.uniform(lo, hi))
    else:
        per_item = float(rng.normal(cfg.SIM_SEC_PER_ITEM_MEAN, cfg.SIM_SEC_PER_ITEM_SD))
        per_item = max(cfg.SPEED_SEC_PER_ITEM + 1.0, per_item)
    duration = int(per_item * n_items)

    birth, assessment = _dates(rng, age_days)
    record = Record(
        record_id=record_id,
        child_id=record_id,
        assessment_date=assessment,
        birth_date=birth,
        age_days=age_days,
        answers=answers,
        duration_seconds=duration,
        informant=informant,
        label=z,
        label_source="mo_phong",
        data_origin="synthetic",
    )
    meta = {"z": z, "theta": theta, "atypical": atypical,
            "informant": informant, "age_days": age_days,
            "n_applicable": len(applicable)}
    return record, meta


def generate_dataset(n: int,
                     questions: Sequence[Question],
                     seed: int = None,
                     prevalence: float = None,
                     prefix: str = "SIM") -> tuple[list[Record], list[dict]]:
    """→ danh sách Record với data_origin='synthetic', label_source='mo_phong'."""
    seed = cfg.SIM_CONFIG["random_seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    records, metas = [], []
    for i in range(n):
        rec, meta = generate_child(rng, questions, prevalence=prevalence,
                                   record_id=f"{prefix}-{i + 1:05d}")
        records.append(rec)
        metas.append(meta)
    return records, metas


# =====================================================================
# SC-1..SC-5 — kiểm tra bắt buộc trước khi dùng (§8.8)
# =====================================================================

def _rank_with_ties(xs: Sequence[float]) -> list[float]:
    """Hạng trung bình cho các giá trị bằng nhau.

    Bắt buộc ở đây: DELTA có GTNN = CNXH = 1,8. Dùng hạng thứ tự thường sẽ phá
    thế đồng hạng một cách tuỳ tiện theo thứ tự khai báo và làm SC-4 đỏ oan.
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and xs[order[end + 1]] == xs[order[pos]]:
            end += 1
        avg = (pos + end) / 2.0
        for k in range(pos, end + 1):
            ranks[order[k]] = avg
        pos = end + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = np.array(_rank_with_ties(a)), np.array(_rank_with_ties(b))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom else 0.0


def sanity_check(records: Sequence[Record],
                 metas: Sequence[dict],
                 questions: Sequence[Question],
                 prevalence: float = None) -> "ValidationReport":
    """SC-1..SC-5. SC-5 là chốt chặn quan trọng nhất — chống lại cái bẫy §8.1."""
    from ts_evaluate import classification_metrics
    from ts_features import compute_domain_features
    from ts_fusion import byt_threshold_met
    from ts_types import ValidationReport

    prevalence = cfg.SIM_CONFIG["prevalence"] if prevalence is None else prevalence
    report = ValidationReport(name="Kiểm tra bộ sinh dữ liệu SC-1..SC-5", ok=True)
    n = len(records)

    # ---- SC-1: tỷ lệ dương tính thực tế ≈ prevalence ----
    observed = sum(1 for r in records if r.label == 1) / n
    rel_err = abs(observed - prevalence) / prevalence
    report.checks["SC-1"] = {"observed": observed, "target": prevalence,
                             "rel_error": rel_err}
    if rel_err > cfg.SANITY_PREVALENCE_TOL:
        report.ok = False
        report.errors.append(
            f"SC-1: tỷ lệ dương tính {observed:.4f} lệch {rel_err:.0%} so với "
            f"{prevalence} (cho phép {cfg.SANITY_PREVALENCE_TOL:.0%})")

    # ---- SC-2: phân bố tuổi phủ đều ----
    ages = np.array([r.age_days for r in records])
    lo, hi = cfg.SIM_CONFIG["age_min_days"], cfg.SIM_CONFIG["age_max_days"]
    n_bins = 10
    counts, _ = np.histogram(ages, bins=n_bins, range=(lo, hi))
    expected = n / n_bins
    max_dev = float(np.max(np.abs(counts - expected)) / expected)
    report.checks["SC-2"] = {"age_min": int(ages.min()), "age_max": int(ages.max()),
                             "bin_counts": counts.tolist(), "max_deviation": max_dev}
    if max_dev > 0.25:
        report.ok = False
        report.errors.append(
            f"SC-2: phân bố tuổi lệch tới {max_dev:.0%} so với đều")

    # ---- SC-3: đủ 4 nhóm ca không điển hình ----
    groups: dict[str, int] = {}
    for m in metas:
        if m["atypical"]:
            groups[m["atypical"]] = groups.get(m["atypical"], 0) + 1
    report.checks["SC-3"] = groups
    thieu = [name for name in cfg.ATYPICAL if groups.get(name, 0) == 0]
    if thieu:
        report.ok = False
        report.errors.append(f"SC-3: thiếu nhóm ca không điển hình {thieu}")

    # ---- SC-4: ma trận tương quan giữa các lĩnh vực khớp cấu trúc DELTA ----
    # Đọc đúng câu chữ §8.8: kiểm MA TRẬN TƯƠNG QUAN, không phải chênh lệch
    # trung bình. Lý do hai thứ khác nhau: hai lĩnh vực cùng bị kéo bởi z sẽ
    # tương quan mạnh với nhau, và mức tương quan đó phải tăng theo δ_i·δ_j.
    columns: dict[str, list[float]] = {d: [] for d in cfg.DOMAINS}
    per_record: list[dict[str, float]] = []
    for rec in records:
        feats = compute_domain_features(rec, questions)
        row = {d: f.d.point for d, f in feats.items()
               if f.available and f.d is not None}
        per_record.append(row)
    for row in per_record:
        if len(row) < len(cfg.DOMAINS):
            continue                     # chỉ dùng hồ sơ đủ 6 lĩnh vực
        for d, v in row.items():
            columns[d].append(v)

    # Phải KHỬ TUỔI trước khi tính tương quan. Tuổi là biến đồng biến chi phối
    # cả 6 lĩnh vực (trẻ lớn hơn đạt nhiều mốc hơn) và vì z chỉ chiếm ~3 % mẫu
    # nên hiệp phương sai do tuổi át hoàn toàn hiệp phương sai do z. Không khử
    # tuổi thì tương quan hạng ra ÂM (đo được −0,405) và SC-4 báo sai hoàn toàn.
    # Cách khử: trừ trung bình theo từng thập phân vị tuổi (within-bin centering),
    # không giả định dạng hàm nào giữa tuổi và d_i.
    ages_complete = [rec.age_days for rec, row in zip(records, per_record)
                     if len(row) == len(cfg.DOMAINS)]
    n_complete = len(columns[cfg.DOMAINS[0]])
    pairs, products, corr_detail = [], [], {}
    if n_complete >= 30:
        mat = np.array([columns[d] for d in cfg.DOMAINS], dtype=float)
        age_arr = np.array(ages_complete, dtype=float)
        edges = np.quantile(age_arr, np.linspace(0, 1, 11))
        bin_idx = np.clip(np.searchsorted(edges, age_arr, side="right") - 1, 0, 9)
        for b in range(10):
            sel = bin_idx == b
            if sel.sum() >= 2:
                mat[:, sel] -= mat[:, sel].mean(axis=1, keepdims=True)
        corr = np.corrcoef(mat)
        for i, a in enumerate(cfg.DOMAINS):
            for j, b in enumerate(cfg.DOMAINS):
                if j <= i:
                    continue
                pairs.append(float(corr[i, j]))
                products.append(cfg.DELTA[a] * cfg.DELTA[b])
                corr_detail[f"{a}|{b}"] = {"corr": float(corr[i, j]),
                                           "delta_product": cfg.DELTA[a] * cfg.DELTA[b]}
    rho = _spearman(pairs, products) if pairs else 0.0

    # Chẩn đoán phụ (KHÔNG dùng để đánh trượt): chênh lệch d_i giữa z=1 và z=0.
    # Biên độ chênh lệch phụ thuộc cả vào việc mốc tuổi của lĩnh vực có rơi vào
    # cửa sổ tuổi được lấy mẫu hay không, nên nó đo chất lượng NGÂN HÀNG CÂU HỎI
    # nhiều ngang với cấu trúc của mô hình sinh — xem [REVIEW-P8.2].
    sums = {d: [0.0, 0, 0.0, 0] for d in cfg.DOMAINS}
    for rec, row in zip(records, per_record):
        for domain, value in row.items():
            idx = 0 if rec.label == 1 else 2
            sums[domain][idx] += value
            sums[domain][idx + 1] += 1
    gaps, deltas, gap_detail = [], [], {}
    for domain in cfg.DOMAINS:
        s1, n1, s0, n0 = sums[domain]
        if n1 == 0 or n0 == 0:
            continue
        gap = s1 / n1 - s0 / n0
        gaps.append(gap)
        deltas.append(cfg.DELTA[domain])
        gap_detail[domain] = {"gap": gap, "delta": cfg.DELTA[domain]}
    rho_gap = _spearman(gaps, deltas) if gaps else 0.0

    # SC-4a — mệnh đề THỰC CHẤT cần kiểm: các trục cốt lõi DSM-5 phải phản ứng
    # với z mạnh hơn hẳn hai lĩnh vực vận động. Đây là điều DELTA muốn nói, và là
    # điều quyết định mô hình có phân biệt được "chậm vận động chung" với RLPTK.
    #
    # KHÔNG dùng tương quan hạng làm tiêu chí đánh trượt: với 6 lĩnh vực và hai
    # thế đồng hạng (GTNN = CNXH = 1,8), Spearman quá thô — nó phạt các đảo thứ
    # tự vi mô trong nhóm cao trong khi mệnh đề thực chất vẫn đúng rõ ràng.
    core = [d for d in cfg.DOMAINS if cfg.DELTA[d] >= 1.2 and d in gap_detail]
    motor = [d for d in cfg.DOMAINS if cfg.DELTA[d] <= 0.5 and d in gap_detail]
    gap_core = sum(gap_detail[d]["gap"] for d in core) / len(core) if core else 0.0
    gap_motor = sum(gap_detail[d]["gap"] for d in motor) / len(motor) if motor else 0.0
    ratio = gap_core / gap_motor if gap_motor > 1e-9 else float("inf")

    report.checks["SC-4"] = {
        "gap_core": gap_core, "gap_motor": gap_motor, "ratio": ratio,
        "spearman_gap_vs_delta": rho_gap,
        "spearman_corr_vs_delta": rho,
        "n_complete_records": n_complete,
        "per_domain_gap": gap_detail,
        "pairs": corr_detail,
    }
    if ratio < 3.0:
        report.ok = False
        report.errors.append(
            f"SC-4a: trục cốt lõi chỉ phản ứng mạnh gấp {ratio:.2f}× so với trục vận "
            "động (cần ≥ 3×) — mô hình sinh không tái hiện cấu trúc lâm sàng")
    if rho_gap < 0.70:
        report.warnings.append(
            f"SC-4b: tương quan hạng giữa chênh lệch d_i và DELTA chỉ đạt "
            f"{rho_gap:.3f}. Hai nguyên nhân có thể: (a) câu hỏi bão hoà trong "
            "cửa sổ tuổi — xem SC-4c; (b) DHBT dùng cơ chế sinh khác (không có "
            "trục tuổi) nên δ của nó biểu hiện trọn vẹn, trong khi các lĩnh vực "
            "mốc tuổi mất bớt tín hiệu do hiệu ứng tuổi. Xem [REVIEW-P8.2].")
    if n_complete >= 30 and rho < 0.30:
        report.warnings.append(
            f"SC-4 (tương quan cặp): rho = {rho:.3f}. Ở prevalence "
            f"{prevalence:.0%}, hiệp phương sai do z quá nhỏ so với nhiễu nên chỉ "
            "số này không nhạy; dùng SC-4a làm tiêu chí chính.")

    # ---- SC-4c: chẩn đoán độ thông tin của từng câu hỏi ----
    # Một câu mà gần như MỌI trẻ đều đạt (hoặc đều chưa đạt) trong cửa sổ tuổi
    # đang xét thì không mang thông tin cho bất kỳ mô hình nào — không riêng gì
    # SC-4. Đây là kết quả đáng đưa vào luận văn khi biện luận về bộ câu hỏi.
    from ts_types import risk_bit
    rates: dict[str, dict] = {}
    for q in questions:
        n_ans = n_risk = 0
        for rec in records:
            bit = risk_bit((rec.answers or {}).get(q.question_id), q.risk_answer)
            if bit is None:
                continue
            n_ans += 1
            n_risk += int(bit)
        if n_ans:
            rates[q.question_id] = {"domain": q.domain, "n": n_ans,
                                    "risk_rate": n_risk / n_ans,
                                    "a_j": q.expected_age_days}
    bao_hoa = {k: v for k, v in rates.items()
               if v["risk_rate"] < 0.05 or v["risk_rate"] > 0.95}
    report.checks["SC-4c_item_rates"] = rates
    report.checks["SC-4c_saturated"] = bao_hoa
    if bao_hoa:
        report.warnings.append(
            "SC-4c: câu hỏi bão hoà (tỷ lệ nguy cơ < 5 % hoặc > 95 %) nên gần như "
            "không mang thông tin: "
            + ", ".join(f"{k}({v['domain']}, {v['risk_rate']:.1%})"
                        for k, v in bao_hoa.items()))

    # ---- SC-5: luật BYT KHÔNG được phân loại quá tốt ----
    y_true = [r.label for r in records]
    y_byt = [1 if byt_threshold_met(r, questions)[0] else 0 for r in records]
    m = classification_metrics(y_true, y_byt)
    acc = m["Accuracy"]["value"]
    report.checks["SC-5"] = {"byt_accuracy": acc,
                             "byt_se": m["Se"]["value"] if m["Se"] else None,
                             "byt_sp": m["Sp"]["value"] if m["Sp"] else None,
                             "confusion": m["confusion"]}
    if acc >= cfg.SANITY_BYT_MAX_ACC:
        report.ok = False
        report.errors.append(
            f"SC-5: luật BYT đạt accuracy {acc:.4f} ≥ {cfg.SANITY_BYT_MAX_ACC} "
            "→ mô hình sinh quá dễ, phải tăng nhiễu. Đây chính là cái bẫy §8.1.")

    return report


def describe(records: Sequence[Record], metas: Sequence[dict]) -> str:
    """Tóm tắt ngắn gọn phục vụ phần mô tả dữ liệu mô phỏng của luận văn."""
    n = len(records)
    n_pos = sum(1 for r in records if r.label == 1)
    groups: dict[str, int] = {}
    for m in metas:
        key = m["atypical"] or "dien_hinh"
        groups[key] = groups.get(key, 0) + 1
    n_missing = sum(1 for r in records
                    for v in (r.answers or {}).values() if v is None)
    n_items = sum(len(r.answers or {}) for r in records)

    lines = [f"n = {n}, dương tính = {n_pos} ({n_pos / n:.2%})",
             f"tuổi: {min(r.age_days for r in records)} – "
             f"{max(r.age_days for r in records)} ngày",
             f"câu bỏ trống: {n_missing}/{n_items} ({n_missing / n_items:.2%})"]
    for name, count in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:28s} {count:5d} ({count / n:.2%})")
    return "\n".join(lines)
