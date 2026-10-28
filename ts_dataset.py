# -*- coding: utf-8 -*-
"""
ts_dataset.py — Nạp ngân hàng câu hỏi và hồ sơ sàng lọc, kiểm bất biến INV-1..INV-5.

Hỗ trợ hai dạng hồ sơ:
  1. TỪNG CÂU  — `answers` {"Q01": "CO", ...}; đúng như đặc tả §1.3.
  2. MỨC LĨNH VỰC — `domain_counts` {"GTNN": {"n": 3, "risk": 2}, ...}; đây là
     dạng của toàn bộ dữ liệu thật hiện có (xem [REVIEW-P3]).

Bộ chuyển đổi `load_real_csv()` đọc trực tiếp `Mau_thu_thap_du_lieu_that_26.csv`
với header hai dòng của nó.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
from typing import Iterable, Optional, Sequence

import config as cfg
from ts_types import Question, Record


class DataError(ValueError):
    """Vi phạm bất biến dữ liệu INV-1..INV-5."""


# =====================================================================
# Ngân hàng câu hỏi (§1.1)
# =====================================================================

_REQUIRED_COLUMNS = [
    "question_id", "domain", "question_content", "risk_answer",
    "expected_age_days", "tolerance_days", "weight", "is_critical",
    "age_min_days", "age_max_days", "abstract_level",
]


def load_questions(path: str = None) -> list[Question]:
    path = path or cfg.QUESTIONS_CSV
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise DataError(f"{path}: không có dòng dữ liệu nào")
    missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise DataError(f"{path}: thiếu cột {missing}")

    questions = []
    seen: set[str] = set()
    for row in rows:
        qid = row["question_id"].strip()
        if qid in seen:
            raise DataError(f"question_id trùng lặp: {qid}")
        seen.add(qid)

        domain = row["domain"].strip()
        if domain not in cfg.DOMAINS:
            raise DataError(f"[{qid}] lĩnh vực không hợp lệ: {domain!r}")
        risk_answer = row["risk_answer"].strip().upper()
        if risk_answer not in ("CO", "KHONG"):
            raise DataError(f"[{qid}] risk_answer phải là CO hoặc KHONG, nhận {risk_answer!r}")

        questions.append(Question(
            question_id=qid,
            domain=domain,
            question_content=row["question_content"].strip(),
            risk_answer=risk_answer,
            expected_age_days=int(row["expected_age_days"]),
            tolerance_days=int(row["tolerance_days"]),
            weight=float(row["weight"]),
            is_critical=int(row["is_critical"]),
            age_min_days=int(row["age_min_days"]),
            age_max_days=int(row["age_max_days"]),
            abstract_level=float(row["abstract_level"]),
            source=row.get("source", "").strip(),
        ))

    _check_question_bank(questions)
    return questions


def _check_question_bank(questions: Sequence[Question]) -> None:
    counts: dict[str, int] = {}
    for q in questions:
        counts[q.domain] = counts.get(q.domain, 0) + 1
        if q.age_min_days > q.age_max_days:
            raise DataError(f"[{q.question_id}] age_min > age_max")
        if q.domain not in cfg.AGELESS_DOMAINS and q.tolerance_days <= 0:
            raise DataError(f"[{q.question_id}] tolerance_days phải dương")
        if not (0.0 <= q.abstract_level <= 1.0):
            raise DataError(f"[{q.question_id}] abstract_level ngoài [0,1]")

    for domain in cfg.DOMAINS:
        if counts.get(domain, 0) == 0:
            raise DataError(f"Lĩnh vực {domain} không có câu hỏi nào")


def question_counts(questions: Sequence[Question]) -> dict[str, int]:
    out = {d: 0 for d in cfg.DOMAINS}
    for q in questions:
        out[q.domain] += 1
    return out


# =====================================================================
# Bất biến hồ sơ (§1.4)
# =====================================================================

def _parse_date(value: str) -> _dt.date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return _dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise DataError(f"Không đọc được ngày: {value!r}")


def validate_record(record: Record, questions: Sequence[Question]) -> list[str]:
    """Kiểm INV-1..INV-5, trả về danh sách cảnh báo (lỗi thì raise)."""
    warnings: list[str] = []

    # INV-1
    if record.birth_date and record.assessment_date:
        expected = (_parse_date(record.assessment_date)
                    - _parse_date(record.birth_date)).days
        if expected != record.age_days:
            raise DataError(
                f"INV-1 [{record.record_id}]: age_days = {record.age_days} nhưng "
                f"(ngày đánh giá − ngày sinh) = {expected}")

    known = {q.question_id for q in questions}

    # INV-2
    if record.answers is not None:
        unknown = [k for k in record.answers if k not in known]
        if unknown:
            raise DataError(f"INV-2 [{record.record_id}]: câu hỏi lạ {unknown}")

    if record.domain_counts is not None:
        bad = [d for d in record.domain_counts if d not in cfg.DOMAINS]
        if bad:
            raise DataError(f"INV-2 [{record.record_id}]: lĩnh vực lạ {bad}")

    # INV-3 — [REVIEW-P4] không loại hồ sơ, chỉ cảnh báo và gắn nhãn nhóm tuổi.
    lo, hi = cfg.AGE_SCOPE_DAYS
    if not (lo <= record.age_days <= hi):
        warnings.append(
            f"[{record.record_id}] tuổi {record.age_days} ngày nằm ngoài phạm vi "
            f"chính danh {lo}–{hi} ngày → nhóm '{record.age_group}'")
    if record.age_days < cfg.AGE_BANK_MIN_DAYS:
        warnings.append(
            f"[{record.record_id}] tuổi {record.age_days} ngày nhỏ hơn "
            f"{cfg.AGE_BANK_MIN_DAYS} ngày: một số lĩnh vực không có câu hỏi áp "
            "dụng được, hồ sơ 6 lĩnh vực sẽ không đầy đủ (CH-5)")
    if record.age_days > cfg.AGE_HARD_MAX_DAYS:
        raise DataError(
            f"INV-3 [{record.record_id}]: tuổi {record.age_days} vượt giới hạn kỹ "
            f"thuật {cfg.AGE_HARD_MAX_DAYS} ngày của ngân hàng câu hỏi")

    # INV-5
    if record.data_origin == "synthetic" and record.label_source != "mo_phong":
        raise DataError(
            f"INV-5 [{record.record_id}]: data_origin='synthetic' thì "
            f"label_source phải là 'mo_phong', nhận {record.label_source!r}")

    # [REVIEW-P6a] nguồn nhãn là mục hội đồng sẽ hỏi.
    if record.label is not None and not record.label_source:
        warnings.append(
            f"[{record.record_id}] có nhãn nhưng thiếu label_source — "
            "Se/Sp báo cáo trên hồ sơ này không diễn giải được")

    return warnings


# =====================================================================
# Nạp / ghi records.jsonl (§1.3)
# =====================================================================

def load_records(path: str = None, validate: bool = True,
                 questions: Sequence[Question] = None) -> tuple[list[Record], list[str]]:
    path = path or cfg.RECORDS_JSONL
    questions = questions if questions is not None else load_questions()

    records, warnings = [], []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_no} JSON không hợp lệ: {exc}") from None
            rec = Record(**raw)
            if validate:
                warnings.extend(validate_record(rec, questions))
            records.append(rec)
    return records, warnings


def save_records(records: Iterable[Record], path: str) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            payload = {k: v for k, v in vars(rec).items() if v is not None or k in
                       ("answers", "domain_counts", "label")}
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


# =====================================================================
# Bộ chuyển đổi CSV thu thập thực tế → Record  [REVIEW-P3]
# =====================================================================

# Thứ tự cột trong Mau_thu_thap_du_lieu_that_*.csv (header hai dòng).
_REAL_CSV_LAYOUT = [
    ("GTNN",   4,  5),      # (lĩnh vực, cột "số câu Có", cột "số câu Không")
    ("VDTHO",  6,  7),
    ("VDTINH", 8,  9),
    ("BCHOC",  10, 11),
    ("CNXH",   12, 13),
]
_REAL_CSV_DHBT_COL = 14     # "số câu Có / 5"
_REAL_CSV_LABEL_COL = 15
_REAL_CSV_SOURCE_COL = 16
_REAL_CSV_NOTE_COL = 17


def load_real_csv(path: str,
                  questions: Sequence[Question] = None,
                  label_source: str = None,
                  dhbt_zero_as_missing: bool = None) -> tuple[list[Record], list[str]]:
    """Đọc bảng thu thập mức lĩnh vực thành danh sách Record.

    Với 5 lĩnh vực chính, đáp án mang nguy cơ là "Không" nên
    `risk = số câu Không`. Với DHBT, đáp án mang nguy cơ là "Có" nên
    `risk = số câu Có` (§1.1).
    """
    questions = questions if questions is not None else load_questions()
    counts = question_counts(questions)
    if dhbt_zero_as_missing is None:
        dhbt_zero_as_missing = cfg.DHBT_ZERO_AS_MISSING

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))

    records: list[Record] = []
    warnings: list[str] = []

    for row in rows[2:]:                       # bỏ hai dòng header
        if not row or not row[0].strip():
            continue
        code = row[0].strip()

        birth = _parse_date(row[1])
        assess = _parse_date(row[2])
        age_days = (assess - birth).days
        if row[3].strip() and int(row[3]) != age_days:
            warnings.append(
                f"[{code}] cột 'Ngày tuổi' ghi {row[3]} nhưng tính lại được "
                f"{age_days} — dùng giá trị tính lại (INV-1)")

        domain_counts: dict[str, dict[str, int]] = {}
        for domain, col_yes, col_no in _REAL_CSV_LAYOUT:
            n_yes = int(row[col_yes] or 0)
            n_no = int(row[col_no] or 0)
            n_total = n_yes + n_no
            if n_total > counts[domain]:
                raise DataError(
                    f"[{code}] {domain}: {n_total} câu đã trả lời nhưng ngân hàng "
                    f"chỉ có {counts[domain]} câu")
            domain_counts[domain] = {"n": n_total, "risk": n_no}

        n_dhbt = int(row[_REAL_CSV_DHBT_COL] or 0)
        if n_dhbt == 0 and dhbt_zero_as_missing:
            # [REVIEW-P5] không đưa DHBT vào hồ sơ → lĩnh vực bị loại,
            # φ được chuẩn hoá lại theo CH-5 thay vì ngầm khẳng định "không có
            # dấu hiệu bất thường nào".
            warnings.append(
                f"[{code}] DHBT = 0 được coi là THIẾU DỮ LIỆU "
                "(cfg.DHBT_ZERO_AS_MISSING = True)")
        else:
            domain_counts["DHBT"] = {"n": counts["DHBT"], "risk": n_dhbt}

        label_text = row[_REAL_CSV_LABEL_COL].strip() if len(row) > _REAL_CSV_LABEL_COL else ""
        if label_text.startswith("Có"):
            label = 1
        elif label_text.startswith("Không"):
            label = 0
        else:
            label = None

        src = ""
        if len(row) > _REAL_CSV_SOURCE_COL:
            src = row[_REAL_CSV_SOURCE_COL].strip()
        src = src or label_source or ""

        rec = Record(
            record_id=code,
            child_id=code,
            assessment_date=assess.isoformat(),
            birth_date=birth.isoformat(),
            age_days=age_days,
            domain_counts=domain_counts,
            label=label,
            label_source=src or None,
            data_origin="real",
            notes=row[_REAL_CSV_NOTE_COL].strip() if len(row) > _REAL_CSV_NOTE_COL else "",
        )
        warnings.extend(validate_record(rec, questions))
        records.append(rec)

    return records, warnings


def summarize(records: Sequence[Record]) -> dict:
    """Thống kê nhanh phục vụ B0/B1 và phần mô tả dữ liệu của luận văn."""
    n = len(records)
    pos = sum(1 for r in records if r.label == 1)
    neg = sum(1 for r in records if r.label == 0)
    strata: dict[str, dict[str, int]] = {}
    for r in records:
        bucket = strata.setdefault(r.age_group, {"n": 0, "pos": 0, "neg": 0})
        bucket["n"] += 1
        if r.label == 1:
            bucket["pos"] += 1
        elif r.label == 0:
            bucket["neg"] += 1

    ages = [r.age_days for r in records]
    return {
        "n": n,
        "n_positive": pos,
        "n_negative": neg,
        "n_unlabeled": n - pos - neg,
        "prevalence": pos / n if n else None,
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "by_age_group": strata,
        "n_itemwise": sum(1 for r in records if r.is_itemwise),
        "n_aggregated": sum(1 for r in records if not r.is_itemwise),
        "label_sources": sorted({r.label_source or "(trống)" for r in records}),
    }
