# -*- coding: utf-8 -*-
"""
ts_fcm.py — §6 NHÁNH 3: FUZZY C-MEANS.

Nhánh KHÔNG GIÁM SÁT. Điểm mấu chốt: nó dùng được toàn bộ hồ sơ đã khảo sát, kể
cả ca chưa có nhãn chẩn đoán, nên né được trực diện vấn đề khan hiếm nhãn (V4).

Mục tiêu là PHÁT HIỆN các kiểu hồ sơ hành vi tồn tại trong dữ liệu thay vì áp
đặt sẵn. Cụm B (chậm vận động đơn thuần) và cụm D (chậm phát triển toàn diện) đặc
biệt quan trọng: đó là những ca mà phương pháp cộng tổng điểm hay báo động nhầm.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

import config as cfg
from ts_aggregate import choquet_2additive, renormalize_for_missing


class FuzzyCMeans:
    def __init__(self, c: int = None, m: float = None,
                 max_iter: int = None, tol: float = None,
                 domains: Sequence[str] = None, seed: int = 20260826):
        self.c = cfg.FCM_C if c is None else c
        self.m = cfg.FCM_M if m is None else m
        self.max_iter = cfg.FCM_MAX_ITER if max_iter is None else max_iter
        self.tol = cfg.FCM_TOL if tol is None else tol
        self.domains = list(cfg.DOMAINS if domains is None else domains)
        self.seed = seed
        self.centers: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None
        self.cluster_risk: dict[int, float] = {}

    # -- tiện ích -----------------------------------------------------

    def to_matrix(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        """Chỉ giữ hồ sơ đủ mọi lĩnh vực mà mô hình đang dùng."""
        out = [[r[d] for d in self.domains] for r in rows
               if all(d in r for d in self.domains)]
        return np.array(out, dtype=float)

    def _distances(self, x: Mapping[str, float]) -> np.ndarray:
        """Khoảng cách từ x tới từng tâm cụm, chịu được hồ sơ thiếu lĩnh vực.

        Khi hồ sơ thiếu một lĩnh vực (bị loại vì bỏ trống > 30 %), khoảng cách
        được tính trên các chiều còn lại rồi nhân bù sqrt(n_all / n_shared) để
        giữ tính so sánh được về độ lớn — chiếu lên không gian con thay vì điền
        khuyết, đúng tinh thần "cấm tuyệt đối điền khuyết bằng 0" của §2.5.
        """
        shared = [i for i, d in enumerate(self.domains) if d in x]
        if not shared:
            raise ValueError(f"Hồ sơ không có lĩnh vực nào thuộc {self.domains}")
        v = np.array([x[self.domains[i]] for i in shared], dtype=float)
        diff = self.centers[:, shared] - v[None, :]
        scale = np.sqrt(len(self.domains) / len(shared))
        return np.maximum(np.linalg.norm(diff, axis=1) * scale, 1e-12)

    def _vector(self, x: Mapping[str, float]) -> np.ndarray:
        missing = [d for d in self.domains if d not in x]
        if missing:
            raise ValueError(f"Hồ sơ thiếu lĩnh vực {missing}; FCM cần đủ "
                             f"{self.domains}")
        return np.array([x[d] for d in self.domains], dtype=float)

    # -- §6.2 thuật toán ----------------------------------------------

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < self.c:
            raise ValueError(f"Cần ít nhất {self.c} hồ sơ đủ lĩnh vực, nhận {X.shape}")
        self.X = X
        n = X.shape[0]

        rng = np.random.default_rng(self.seed)
        U = rng.random((n, self.c))
        U /= U.sum(axis=1, keepdims=True)

        for _ in range(self.max_iter):
            Um = U ** self.m
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]
            dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            dist = np.maximum(dist, 1e-12)
            inv = dist ** (-2.0 / (self.m - 1.0))
            U_new = inv / inv.sum(axis=1, keepdims=True)
            if np.max(np.abs(U_new - U)) < self.tol:
                U = U_new
                break
            U = U_new

        self.U = U
        self.centers = (U ** self.m).T @ X / (U ** self.m).sum(axis=0)[:, None]
        return self

    def predict_memberships(self, x) -> np.ndarray:
        if self.centers is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        if isinstance(x, Mapping):
            dist = self._distances(x)
        else:
            v = np.asarray(x, dtype=float)
            dist = np.maximum(np.linalg.norm(self.centers - v[None, :], axis=1), 1e-12)
        inv = dist ** (-2.0 / (self.m - 1.0))
        return inv / inv.sum()

    # -- §6.3 chỉ số chọn số cụm --------------------------------------

    def xie_beni(self) -> float:
        """XB nhỏ hơn thì tốt hơn."""
        if self.U is None or self.centers is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        dist2 = np.linalg.norm(self.X[:, None, :] - self.centers[None, :, :], axis=2) ** 2
        numer = float(((self.U ** self.m) * dist2).sum())
        sep = np.linalg.norm(self.centers[:, None, :] - self.centers[None, :, :], axis=2) ** 2
        np.fill_diagonal(sep, np.inf)
        min_sep = float(sep.min())
        if not np.isfinite(min_sep) or min_sep <= 1e-12:
            # Hai tâm cụm trùng nhau → phân hoạch suy biến. Trả vô cực để phép
            # chọn c ở select_c() tự loại giá trị này thay vì sập.
            return float("inf")
        return numer / (self.X.shape[0] * min_sep)

    def partition_coefficient(self) -> float:
        """PC lớn hơn thì tốt hơn."""
        return float((self.U ** 2).sum() / self.X.shape[0])

    def normalized_partition_coefficient(self) -> float:
        """NPC = (PC − 1/c)/(1 − 1/c) ∈ [0, 1]; 0 ⇔ mọi độ thuộc bằng 1/c."""
        c = self.c
        return float((self.partition_coefficient() - 1.0 / c) / (1.0 - 1.0 / c))

    @property
    def is_usable(self) -> tuple[bool, str]:
        """[P7 / N6] — cùng vai trò với FuzzyKNN.is_usable: nhánh suy biến thì
        không được tham gia hợp nhất; fuse() chuẩn hoá lại trọng số."""
        if self.U is None:
            return False, "chưa huấn luyện"
        npc = self.normalized_partition_coefficient()
        if npc < cfg.FCM_MIN_NPC:
            return False, (f"phân hoạch suy biến: PC chuẩn hoá = {npc:.4f} "
                           f"< {cfg.FCM_MIN_NPC}")
        return True, f"PC chuẩn hoá = {npc:.4f}"

    # -- §6.4 gán mức nguy cơ cho cụm ---------------------------------

    def label_clusters(self, X_labeled: Sequence[Mapping[str, float]],
                       y: Sequence[int]) -> dict[int, float]:
        """r_k theo §6.4, có ghi lại nguồn gốc của từng giá trị."""
        self.cluster_risk = {}
        self.cluster_source: dict[int, str] = {}

        rows = [r for r in X_labeled if all(d in r for d in self.domains)]
        labels = [v for r, v in zip(X_labeled, y) if all(d in r for d in self.domains)]

        memberships = [self.predict_memberships(r) for r in rows] if rows else []

        for k in range(self.c):
            weight_sum = sum(u[k] for u in memberships)
            n_members = sum(1 for u in memberships if int(np.argmax(u)) == k)
            if n_members >= cfg.FCM_MIN_LABELED_PER_CLUSTER and weight_sum > 1e-9:
                numer = sum(u[k] * lab for u, lab in zip(memberships, labels))
                self.cluster_risk[k] = 100.0 * numer / weight_sum
                self.cluster_source[k] = f"từ {n_members} ca có nhãn"
            else:
                # Không đủ ca có nhãn → dùng chính Choquet của tâm cụm.
                center = {d: float(v) for d, v in zip(self.domains, self.centers[k])}
                phi, itr = renormalize_for_missing(cfg.SHAPLEY, cfg.INTERACTION,
                                                   self.domains)
                self.cluster_risk[k] = 100.0 * choquet_2additive(center, phi, itr)
                self.cluster_source[k] = "từ Choquet của tâm cụm (thiếu ca có nhãn)"
        return self.cluster_risk

    def score(self, x: Mapping[str, float]) -> float:
        """s₃ = Σ_k u_k(x) · r_k  (§6.4)"""
        if not self.cluster_risk:
            raise ValueError("Chưa gán mức nguy cơ cho cụm; gọi label_clusters() trước")
        u = self.predict_memberships(x)
        return float(sum(u[k] * self.cluster_risk[k] for k in range(self.c)))

    # -- §6.5 mô tả bằng ngôn ngữ tự nhiên ----------------------------

    def describe_cluster(self, k: int) -> str:
        """Sinh mô tả từ tâm cụm — dùng trực tiếp trong báo cáo PDF và phần
        bàn luận của luận văn."""
        if self.centers is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        center = {d: float(v) for d, v in zip(self.domains, self.centers[k])}

        def level(v: float) -> str:
            return "cao" if v >= 0.60 else ("trung bình" if v >= 0.35 else "thấp")

        cao = [d for d, v in center.items() if v >= 0.60]
        thap = [d for d, v in center.items() if v < 0.35]
        motor = {"VDTHO", "VDTINH"}
        social = {"GTNN", "CNXH", "BCHOC", "DHBT"}

        if not cao and len(thap) >= len(self.domains) - 1:
            nhan = "Phát triển điển hình"
        elif set(cao) and set(cao) <= motor:
            nhan = "Chậm vận động đơn thuần"
        elif set(cao) and set(cao) <= social and motor <= set(thap):
            nhan = "Khiếm khuyết giao tiếp – xã hội nổi trội, vận động bình thường"
        elif len(cao) >= len(self.domains) - 1:
            nhan = "Chậm phát triển toàn diện"
        elif cao == ["GTNN"]:
            nhan = "Chậm ngôn ngữ đơn thuần"
        else:
            nhan = "Hồ sơ hỗn hợp"

        chi_tiet = ", ".join(f"{cfg.DOMAIN_NAMES[d]} {level(v)} ({v:.2f})"
                             for d, v in center.items())
        risk = self.cluster_risk.get(k)
        hau_to = f" — mức nguy cơ cụm {risk:.1f}" if risk is not None else ""
        return f"{nhan}: {chi_tiet}{hau_to}"

    def describe_nearest(self, x: Mapping[str, float]) -> str:
        u = self.predict_memberships(x)
        k = int(np.argmax(u))
        return f"{self.describe_cluster(k)} (độ thuộc {u[k]:.2f})"


# =====================================================================
# §6.3 — quét số cụm
# =====================================================================

def select_c(X: np.ndarray, c_grid: Sequence[int] = None,
             domains: Sequence[str] = None, seed: int = 20260826) -> dict:
    """Quét c ∈ {2, 3, 4, 5}, báo cáo CẢ HAI chỉ số Xie–Beni và PC.

    Với n nhỏ, cố định c = 4 theo giả thuyết lâm sàng §6.1 cũng là lựa chọn hợp
    lệ — nhưng phải nêu rõ đó là chọn theo tri thức chuyên môn chứ không phải
    tối ưu từ dữ liệu.
    """
    c_grid = list(cfg.FCM_C_GRID if c_grid is None else c_grid)
    out = {}
    for c in c_grid:
        if X.shape[0] < c:
            continue
        model = FuzzyCMeans(c=c, domains=domains, seed=seed).fit(X)
        out[c] = {"xie_beni": model.xie_beni(),
                  "partition_coefficient": model.partition_coefficient(),
                  "model": model}
    if not out:
        raise ValueError("Không có giá trị c nào hợp lệ")
    best_xb = min(out, key=lambda c: out[c]["xie_beni"])
    best_pc = max(out, key=lambda c: out[c]["partition_coefficient"])
    return {"per_c": out, "best_by_xie_beni": best_xb, "best_by_pc": best_pc}
