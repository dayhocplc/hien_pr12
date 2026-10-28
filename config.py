# -*- coding: utf-8 -*-
"""
config.py — Toàn bộ hằng số của mô hình lai luật – học máy với đầu ra mờ.

Nguyên tắc §0.3.4: KHÔNG hard-code số trong logic. Mọi tham số nằm ở đây để
phục vụ phân tích độ nhạy §9.6.

Tham chiếu: DAC_TA_MO_HINH_LAI_FUZZY.md, Phụ lục A.
Các sai lệch so với đặc tả v1.0 được đánh dấu [REVIEW-Pxx] và giải thích tại
REVIEW_DAC_TA.md.
"""

from __future__ import annotations

# =====================================================================
# LĨNH VỰC
# =====================================================================

DOMAINS = ["GTNN", "VDTHO", "VDTINH", "BCHOC", "CNXH", "DHBT"]

DOMAIN_NAMES = {
    "GTNN":   "Giao tiếp – Ngôn ngữ",
    "VDTHO":  "Vận động thô",
    "VDTINH": "Vận động tinh",
    "BCHOC":  "Bắt chước và học",
    "CNXH":   "Cá nhân – Xã hội",
    "DHBT":   "Dấu hiệu bất thường khác",
}

# Năm lĩnh vực "chính" theo cách phân loại của QĐ 2254/QĐ-BYT (Phiếu 2).
# DHBT không nằm trong nhóm này: nó có điều kiện kích hoạt riêng ở §7.2.
MAIN_DOMAINS = ["GTNN", "VDTHO", "VDTINH", "BCHOC", "CNXH"]

# Lĩnh vực không gắn với mốc tuổi (§3.3): LAG luôn = 0.
AGELESS_DOMAINS = ["DHBT"]


# =====================================================================
# ĐẶC TRƯNG (§2)
# =====================================================================

LAG_MAX = 3.0                 # §2.3 — trần của chỉ số trễ, đơn vị σ
MISS_THRESHOLD_DOM = 0.30     # §2.5 — bỏ trống > 30% trong lĩnh vực → d_i = None
MISS_THRESHOLD_ALL = 0.30     # §2.5 / INV-4 — bỏ trống > 30% toàn phiếu → từ chối

REL_W_MISS = 0.50             # §2.4
REL_W_INCONS = 0.40
REL_W_SPEED = 0.10
SPEED_SEC_PER_ITEM = 20

# Cặp câu dùng kiểm tra tính nhất quán (§2.4): cùng lĩnh vực và
# a_{khó} > a_{dễ} + INCONS_MIN_GAP_DAYS.
INCONS_MIN_GAP_DAYS = 60


# ---------------------------------------------------------------------
# [REVIEW-P3] MỞ RỘNG: đặc trưng dạng KHOẢNG khi dữ liệu chỉ ở mức lĩnh vực
# ---------------------------------------------------------------------
# Dữ liệu thật hiện có chỉ ghi "số câu Có / số câu Không" theo từng lĩnh vực,
# không ghi câu nào. Khi đó ta biết lĩnh vực có đúng k trong n câu mang nguy cơ
# nhưng không biết là những câu nào, nên (d_i, LAG_i) không xác định duy nhất.
#
# Cách xử lý: liệt kê VÉT CẠN mọi tổ hợp C(n, k) khả dĩ (n ≤ 5 nên tối đa 10
# tổ hợp / lĩnh vực), tính risk_i cho từng tổ hợp, rồi lấy:
#     risk_lo  = min,  risk_hi = max,  risk_point = ước lượng điểm
# Cách này chính xác tuyệt đối, tránh được "bài toán phụ thuộc" của số học
# khoảng (interval arithmetic) vì d_i và LAG_i vốn phụ thuộc nhau.
#
# Bề rộng khoảng thu được đi thẳng vào khoảng bất định §7.3 qua số hạng
# UNC_GAMMA. Khi có dữ liệu từng câu, bề rộng tự động bằng 0 và mọi công thức
# quay về đúng đặc tả gốc — không cần nhánh mã riêng.

# "mean"  : trung bình trên mọi tổ hợp — giả định tiên nghiệm đều về việc câu
#           nào chưa đạt. Minh bạch, cần nêu rõ giả định trong luận văn.
# "mid"   : trung điểm (lo + hi) / 2 — không giả định gì về phân phối.
AGG_POINT_ESTIMATE = "mean"

# Với hồ sơ nhập ở mức lĩnh vực, coi mọi câu của lĩnh vực là đã được hỏi
# (phiếu được phát nguyên bộ), bỏ qua lọc tuổi INV-3 ở mức câu.
AGG_ASSUME_ALL_APPLICABLE = True


# ---------------------------------------------------------------------
# [REVIEW-P5] Lĩnh vực DHBT bằng 0 ở cả 26/26 ca thật
# ---------------------------------------------------------------------
# Khó tin rằng 22 trẻ đã được kết luận có nguy cơ RLPTK mà không trẻ nào có
# hành vi lặp lại hay nhạy cảm giác quan. Nhiều khả năng mục này KHÔNG ĐƯỢC HỎI
# khi thu thập. Nếu vậy thì đây là DỮ LIỆU THIẾU, không phải giá trị 0 — và
# điền 0 chính là lỗi mà §2.5 gọi là "cấm tuyệt đối".
#
# True  : coi DHBT = 0 trên toàn bộ hồ sơ là thiếu dữ liệu → d_DHBT = None,
#         chuẩn hoá lại φ theo CH-5. An toàn hơn về mặt sàng lọc.
# False : tin vào giá trị 0 như đã ghi.
# Đặt lại thành False sau khi xác minh được với nơi thu thập dữ liệu.
DHBT_ZERO_AS_MISSING = True


# =====================================================================
# PHẠM VI TUỔI (§1.1, INV-3)  [REVIEW-P4]
# =====================================================================
# Quyết định: giữ cả 26 ca thật, báo cáo tách nhóm theo tuổi.
# AGE_SCOPE_DAYS là phạm vi CHÍNH DANH của đề tài (0–36 tháng);
# AGE_HARD_MAX_DAYS là giới hạn kỹ thuật để không loại bỏ hồ sơ khỏi phân tích.
AGE_SCOPE_DAYS = (0, 1095)        # 0–36 tháng — phạm vi tuyên bố của đề tài
AGE_HARD_MAX_DAYS = 1825          # 60 tháng — giới hạn kỹ thuật của bộ câu hỏi

# [REVIEW-P7.10] Tuổi nhỏ nhất mà CẢ SÁU lĩnh vực đều có ít nhất một câu hỏi áp
# dụng được. Dưới mốc này một số lĩnh vực bị loại và φ phải chuẩn hoá lại theo
# CH-5 — vẫn chấm được nhưng hồ sơ 6 lĩnh vực không đầy đủ.
# Trẻ < 28 ngày dùng Phiếu 1 của QĐ 2254, không dùng bộ 20 câu này (§7.2).
# Ca nhỏ tuổi nhất trong dữ liệu thật là 286 ngày nên ràng buộc này không cản
# trở phân tích hiện tại; nó tồn tại để cảnh báo khi mở rộng thu thập.
AGE_BANK_MIN_DAYS = 180

# Nhãn nhóm tuổi dùng khi báo cáo tách nhóm ở §9.
AGE_STRATA = {
    "trong_pham_vi": (0, 1095),   # ≤ 36 tháng — n = 18 trên dữ liệu thật
    "ngoai_pham_vi": (1096, AGE_HARD_MAX_DAYS),   # > 36 tháng — n = 8
}


# =====================================================================
# FIS LỚP 1 (§3)
# =====================================================================

T_NORM = "product"        # §3.1 — KHÔNG dùng "min" (nguyên nhân của V1)
S_NORM = "bounded_sum"   # [REVIEW-P12] — xem MF_D; "prob_sum" để lại vi phạm 4,97
DEFUZZ = "centroid"

RES_D = 0.01              # độ phân giải universe
RES_LAG = 0.01
RES_RISK = 0.5

# [REVIEW-P12] Hàm thuộc đầu vào chuyển sang PHÂN HOẠCH ĐƠN VỊ (Ruspini).
#
# Bộ ở Phụ lục A có Σ_k μ_k(x) tụt xuống 0,50 (với d) và 0,30 (với LAG) trong
# vùng chồng lấn. MF-4 của đặc tả chỉ yêu cầu Σ > 0 nên KHÔNG bắt được lỗi này.
# Hệ quả: tổng trọng số kích hoạt tụt theo, và risk_i GIẢM khi d TĂNG — tổng mức
# giảm đo được 78,2 điểm theo d và 100,7 điểm theo LAG trên lưới quét.
# "Thiếu hụt nặng hơn nhưng điểm nguy cơ thấp hơn" là lỗi logic chứ không phải
# lựa chọn thiết kế — cùng loại với vi phạm đơn điệu CH-2 mà §4.3 đã bác bỏ.
#
# Cách sửa tối thiểu: GIỮ NGUYÊN LÕI (điểm mà μ = 1, nơi mang ý nghĩa lâm sàng),
# chỉ kéo vai của mỗi tập tới đúng lõi của tập kề bên. Lõi trước/sau:
#     KHONG_THIEU_HUT  0,00        →  0,00        (không đổi)
#     THIEU_HUT_NHE    0,45        →  0,45        (không đổi)
#     THIEU_HUT_RO     [0,80; 1,0] →  [0,80; 1,0] (không đổi)
# Giá trị biên của FIS cũng không đổi: risk(0,0) = 11,50; risk(1,3) = 88,50.
# Kết hợp với S_NORM = "bounded_sum", vi phạm đơn điệu về đúng 0,000.
MF_D = {
    "KHONG_THIEU_HUT": ("trimf",  [0.00, 0.00, 0.45]),
    "THIEU_HUT_NHE":   ("trimf",  [0.00, 0.45, 0.80]),
    "THIEU_HUT_RO":    ("trapmf", [0.45, 0.80, 1.00, 1.00]),
}

MF_LAG = {
    "DUNG_HAN": ("trimf",  [0.00, 0.00, 1.00]),
    "TRE_NHE":  ("trimf",  [0.00, 1.00, 2.50]),
    "TRE_NANG": ("trapmf", [1.00, 2.50, 3.00, 3.00]),
}

MF_D_V1 = {        # nguyên văn Phụ lục A — giữ cho nghiên cứu loại trừ §9.7
    "KHONG_THIEU_HUT": ("trimf",  [0.00, 0.00, 0.40]),
    "THIEU_HUT_NHE":   ("trimf",  [0.20, 0.45, 0.70]),
    "THIEU_HUT_RO":    ("trapmf", [0.55, 0.80, 1.00, 1.00]),
}

MF_LAG_V1 = {
    "DUNG_HAN": ("trimf",  [0.00, 0.00, 0.80]),
    "TRE_NHE":  ("trimf",  [0.50, 1.00, 1.80]),
    "TRE_NANG": ("trapmf", [1.50, 2.50, 3.00, 3.00]),
}

# [REVIEW-P7.1] Giữ NGUYÊN VĂN đặc tả v1.0 — đã kiểm chứng bằng số.
#
# Hiện tượng: giải mờ centroid của trimf[0, 0, 35] cho 11,50 nên trẻ hoàn toàn
# bình thường (d = 0, LAG = 0) vẫn nhận risk_i = 11,50 thay vì 0. Tương tự,
# trần trên là centroid của trimf[65, 100, 100] = 88,50 chứ không phải 100.
# Thang điểm thực tế của hệ là [11,5 ; 88,5], không phải [0 ; 100].
#
# Đề xuất ban đầu của bản review là đổi THAP thành trapmf[0, 0, 10, 35]; đo lại
# cho thấy centroid TĂNG lên 12,27 — TỆ HƠN, vì đuôi phải dài mới là thứ kéo
# trọng tâm chứ không phải phần đỉnh. Muốn hạ xuống dưới 10 phải rút support
# còn khoảng [0, 24], và khi đó vùng chồng lấn với TRUNG_BINH gần như biến mất
# (MF-1 lung lay). Không đáng đánh đổi.
#
# Đây là đặc tính đã biết của giải mờ trọng tâm Mamdani: đầu ra không bao giờ
# chạm hai đầu mút của universe. Cách xử lý đúng KHÔNG phải sửa hàm thuộc mà là:
#   • Ghi rõ thang thực tế [11,5 ; 88,5] khi trình bày kết quả.
#   • §7.6 chuyển sang xác suất hậu nghiệm bằng Se/Sp chứ không dùng điểm thô,
#     nên phép hiệu chỉnh Bayes không bị ảnh hưởng.
#   • §9.4 dùng hiệu chỉnh Platt hoặc hồi quy đẳng hướng — vốn đã được đặc tả
#     yêu cầu — để đường calibration không bị lệch hệ thống.
# Việc phân loại KHÔNG bị ảnh hưởng: 11,50 vẫn cho μ_BINH_THUONG = 1,0.
MF_RISK_DOMAIN = {
    "THAP":       ("trimf", [0, 0, 35]),
    "TRUNG_BINH": ("trimf", [25, 50, 75]),
    "CAO":        ("trimf", [65, 100, 100]),
}

# Thang điểm thực tế của FIS lớp 1 sau giải mờ centroid (đo bằng test).
RISK_FLOOR_MAMDANI = 11.50
RISK_CEIL_MAMDANI = 88.50

# Bảng luật §3.3 — lưới đầy đủ 3×3, phủ kín 100%.
RULES_DOMAIN = [
    ("KHONG_THIEU_HUT", "DUNG_HAN", "THAP"),
    ("KHONG_THIEU_HUT", "TRE_NHE",  "THAP"),
    ("KHONG_THIEU_HUT", "TRE_NANG", "TRUNG_BINH"),
    ("THIEU_HUT_NHE",   "DUNG_HAN", "THAP"),
    ("THIEU_HUT_NHE",   "TRE_NHE",  "TRUNG_BINH"),
    ("THIEU_HUT_NHE",   "TRE_NANG", "CAO"),
    ("THIEU_HUT_RO",    "DUNG_HAN", "TRUNG_BINH"),
    ("THIEU_HUT_RO",    "TRE_NHE",  "CAO"),
    ("THIEU_HUT_RO",    "TRE_NANG", "CAO"),
]

# [REVIEW-P7.9] Với LAG_DHBT ≡ 0, bảng luật rút gọn chặn risk_DHBT ở 50 ngay cả
# khi trẻ có đủ 5/5 dấu hiệu bất thường — khó biện minh cho một lĩnh vực được
# xem là dấu hiệu trực tiếp của RLPTK. Bảng riêng dưới đây ánh xạ trực tiếp
# mức thiếu hụt sang mức nguy cơ, không qua trục LAG.
RULES_AGELESS = [
    ("KHONG_THIEU_HUT", "THAP"),
    ("THIEU_HUT_NHE",   "TRUNG_BINH"),
    ("THIEU_HUT_RO",    "CAO"),
]

# Đặt True để dùng đúng phương án 1 của §3.3 (bảng luật đầy đủ với LAG = 0).
DHBT_USE_LAG_TABLE = False

# [REVIEW-P10.3] FIS lớp 1 CHỈ được dùng khi có chỉ số trễ tuổi.
#
# Khi LAG ≡ 0, ánh xạ d → risk nén toàn bộ d ∈ [0; 0,33] vào [11,5; 13,3] rồi
# bão hoà ở 50 — mất gần hết độ phân giải ở đúng vùng chứa phần lớn năng lực
# phân biệt. Đo trên đoàn hệ Ba Lan: đưa d qua FIS rồi mới tổng hợp làm AUC tụt
# từ 0,9048 xuống 0,7593 (−0,092, p < 0,001) trên bộ 25 mục, và từ 0,9404 xuống
# 0,8716 (−0,069, p < 0,001) trên bộ 10 mục.
#
# FIS lớp 1 sinh ra để KẾT HỢP hai biến (d, LAG). Chạy nó với một biến bị ghim
# ở 0 là dùng ngoài miền thiết kế. Khi không có mốc tuổi kỳ vọng (bộ câu hỏi
# không phải dạng mốc phát triển), tầng này phải được BỎ QUA và d_i đi thẳng vào
# tầng tổng hợp §4.
FIS_BYPASS_WHEN_NO_LAG = True


# =====================================================================
# TỔNG HỢP — TÍCH PHÂN CHOQUET (§4)
# =====================================================================

# [REVIEW-P10.1] φ_DHBT hạ từ 0,16 xuống 0,08 — căn cứ từ kiểm thử ngoài.
#
# Trên đoàn hệ lâm sàng Ba Lan (n = 252, 18–24 tháng), nhóm 11 mục hành vi lặp
# lại chỉ đạt ROC-AUC = 0,5856 [0,5145; 0,6534] — khoảng tin cậy vừa đủ vượt
# mức đoán mò — trong khi nhóm giao tiếp – xã hội đạt 0,9125 [0,8739; 0,9456].
# Quét độ nhạy cho thấy AUC tăng ĐƠN ĐIỆU khi hạ φ_DHBT: 0,9048 (φ = 0,195)
# → 0,9173 (φ = 0).
#
# KHÔNG đặt về 0, vì ba lý do:
#   • Đoàn hệ Ba Lan chỉ phủ 18–24 tháng; hành vi lặp lại biểu hiện rõ hơn ở
#     tuổi lớn hơn, mà đề tài phủ tới 36 tháng.
#   • Các mục DHBT của Q-CHAT không trùng bộ 5 câu DHBT của phiếu BYT; riêng
#     Q20 (thoái lui kỹ năng) là dấu hiệu cảnh báo đỏ mà Q-CHAT không có.
#   • Chọn đúng giá trị tối ưu trên MỘT đoàn hệ là quá khớp, không phải hiệu chỉnh.
#
# Chọn 0,08 = một nửa giá trị cũ, diễn đạt mệnh đề: giá trị CHẨN ĐOÁN của tiêu
# chí B trong DSM-5 KHÁC giá trị SÀNG LỌC ở 0–36 tháng. Toàn bộ đường cong độ
# nhạy phải được báo cáo để lựa chọn này minh bạch.
SHAPLEY = {
    "GTNN":   0.263,   # trục cốt lõi DSM-5
    "CNXH":   0.263,   # trục cốt lõi DSM-5
    "BCHOC":  0.197,   # chú ý chung
    "DHBT":   0.080,   # hành vi lặp lại — xem P10.1
    "VDTINH": 0.110,
    "VDTHO":  0.087,
}

SHAPLEY_V1 = {         # nguyên văn Phụ lục A, giữ cho nghiên cứu loại trừ
    "GTNN": 0.24, "CNXH": 0.24, "BCHOC": 0.18,
    "DHBT": 0.16, "VDTINH": 0.10, "VDTHO": 0.08,
}

# [REVIEW-P10.2] CHỈ SỐ TƯƠNG TÁC ĐẶT VỀ 0 — giả thuyết đã kiểm và BỊ BÁC BỎ.
#
# Bộ giá trị lâm sàng dưới (INTERACTION_CLINICAL) được kiểm trên đoàn hệ Ba Lan
# và làm GIẢM năng lực phân biệt: −0,0100, CI95 [−0,0192; −0,0012], p = 0,029.
# Bỏ dần từng số hạng thì AUC tăng đều:
#     đầy đủ 0,8948 → bỏ I(CNXH,DHBT) 0,8968 → chỉ giữ I(GTNN,CNXH) 0,8999
#     → không tương tác 0,9048
# Mọi số hạng đều gây hại, không riêng số hạng liên quan DHBT.
#
# Với I = 0, tích phân Choquet suy biến ĐÚNG BẰNG trung bình có trọng số
# Shapley, nên khung lý thuyết §4 giữ nguyên; cái thay đổi là tham số, và nó
# được chọn bằng bằng chứng chứ không bằng trực giác. Đây là một KẾT QUẢ của
# luận văn, không phải sự rút lui.
#
# Lưu ý: kiểm thử ngoài KHÔNG phủ được cặp (VDTHO, VDTINH) vì Q-CHAT không có
# mục vận động nào. Số hạng dư thừa −0,15 vì vậy chưa được kiểm chứng — chưa
# được xác nhận mà cũng chưa bị bác bỏ. Phải nêu rõ điều này ở Chương 5.
INTERACTION: dict[tuple[str, str], float] = {}

INTERACTION_CLINICAL = {   # bộ trực giác lâm sàng, đã hiệu chỉnh k = 0.5 (§4.3)
    ("GTNN",  "CNXH"):   +0.18,   # cụm khiếm khuyết giao tiếp xã hội — cộng hưởng
    ("GTNN",  "BCHOC"):  +0.10,
    ("CNXH",  "BCHOC"):  +0.10,
    ("CNXH",  "DHBT"):   +0.08,
    ("VDTHO", "VDTINH"): -0.15,   # chậm vận động chung — dư thừa, CHƯA kiểm chứng
}

OWA_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.06, 0.04]

MEASURE_TOL = 1e-9
MEASURE_MC_SAMPLES = 10_000


# =====================================================================
# HỌC MÁY (§5, §6)
# =====================================================================

FKNN_K_GRID = [1, 3, 5, 7]
FKNN_K = 3
FKNN_M = 2.0
FKNN_MIN_LABELS = 10

# [REVIEW-P7.5] FKNN_MIN_LABELS một mình chưa đủ: tập 26 ca thật có 22 dương /
# 4 âm, thoả 26 ≥ 10 nhưng mọi láng giềng vẫn gần như luôn là ca dương.
# Điều kiện theo lớp thiểu số mới là điều kiện chặn thật.
FKNN_MIN_PER_CLASS = 5

FCM_C = 4
FCM_C_GRID = [2, 3, 4, 5]
FCM_M = 2.0
FCM_MAX_ITER = 300
FCM_TOL = 1e-5
FCM_MIN_LABELED_PER_CLUSTER = 3   # §6.4
# [P7 / N6] Chặn phân hoạch suy biến. Với m = 2 trên mô phỏng 6 chiều, FCM hội tụ
# về nghiệm tầm thường (mọi tâm cụm ≈ trung bình chung, PC = 1/c, PC chuẩn hoá
# = 0) và nhánh 3 cho AUC 0,30 trên TEST. Nhánh chỉ được tham gia hợp nhất khi
# PC chuẩn hoá (PC − 1/c)/(1 − 1/c) đạt mức tối thiểu dưới đây. Đây là chốt chặn
# SUY BIẾN, không phải ngưỡng chất lượng: 0,05 loại nghiệm tầm thường mà không
# chạm các phân hoạch có cấu trúc thật.
FCM_MIN_NPC = 0.05


# =====================================================================
# HỢP NHẤT (§7)
# =====================================================================

FUSION_WEIGHTS = {"rule": 0.50, "fknn": 0.30, "fcm": 0.20}

SAFETY_FLOOR = 65.0     # §7.2 — làm tròn lên từ điểm giao CTD/NN = 62,857
BYT_DOMAIN_RATIO = 2 / 3        # "≥ 2/3 câu Không trong một lĩnh vực chính"
BYT_DHBT_MIN_SIGNS = 1          # "≥ 1 dấu hiệu bất thường"
BYT_NEWBORN_AGE_DAYS = 28       # Phiếu 1 áp dụng cho trẻ < 28 ngày
BYT_NEWBORN_MIN_RISK_ITEMS = 1

UNC_W_MAX = 20.0        # §7.3
UNC_BETA = 0.5
# [REVIEW-P3] hệ số cho bề rộng khoảng sinh ra từ dữ liệu mức lĩnh vực.
# = 1.0 nghĩa là bất định do không biết câu nào chưa đạt được cộng nguyên vẹn
# vào khoảng bất định, không bị chiết khấu.
UNC_GAMMA = 1.0

CONF_THRESHOLD = 0.30   # §9.5 — ranh giới giữa tầng CHẮC CHẮN và RANH GIỚI
REL_WARN_THRESHOLD = 0.50

MF_OUTPUT = {
    "BINH_THUONG":  ("trapmf", [0,  0,  25, 42]),
    "CAN_THEO_DOI": ("trimf",  [35, 50, 68]),
    "NGHI_NGO":     ("trapmf", [58, 75, 100, 100]),
}

OUTPUT_LEVEL_ORDER = ["BINH_THUONG", "CAN_THEO_DOI", "NGHI_NGO"]

MF_TREND = {
    "CAI_THIEN": ("trapmf", [-100, -100, -8, -2]),
    "ON_DINH":   ("trimf",  [-5, 0, 5]),
    "XAU_DI":    ("trapmf", [2, 8, 100, 100]),
}

RULES_TREND = [
    ("XAU_DI",    "CAN_THEO_DOI", "CHUYEN_KHAM_SOM"),
    ("ON_DINH",   "NGHI_NGO",     "CHUYEN_KHAM"),
    ("CAI_THIEN", "CAN_THEO_DOI", "TIEP_TUC_THEO_DOI"),
    ("XAU_DI",    "BINH_THUONG",  "TANG_TAN_SUAT_SANG_LOC"),
]

PREVALENCE = {
    "cong_dong":   0.010,   # sàng lọc đại trà
    "nguy_co_cao": 0.050,   # gia đình đã nghi ngờ, chủ động đi khám
}


# =====================================================================
# MÔ PHỎNG (§8)
# =====================================================================

DELTA = {
    "GTNN":   1.8,
    "CNXH":   1.8,
    "DHBT":   1.5,
    "BCHOC":  1.2,
    "VDTINH": 0.5,
    "VDTHO":  0.3,
}

P_NOISE_BASE = 0.05
P_NOISE_ABSTRACT = 0.10

INFORMANT_BIAS = {"lac_quan": -0.15, "trung_lap": 0.00, "lo_lang": +0.15}
INFORMANT_PROB = {"lac_quan": 0.30, "trung_lap": 0.50, "lo_lang": 0.20}

MISS_MCAR_RATE = 0.03
MISS_FATIGUE_RECORD_RATE = 0.10
MISS_FATIGUE_ITEMS = (1, 3)
SPEED_ANOMALY_RATE = 0.05

# [REVIEW-P8.1] Đặc tả §8.6 ghi các tỷ lệ này mà không nói rõ "tỷ lệ trên cái gì".
# Nếu hiểu là tỷ lệ trên TOÀN BỘ tập thì `asd_van_dong_binh_thuong = 0.05` sẽ lớn
# hơn cả tỷ lệ hiện mắc 0.03 — vô lý. Cách hiểu duy nhất nhất quán là tỷ lệ CÓ
# ĐIỀU KIỆN trên nhóm tương ứng:
#     nhóm asd_*  → tỷ lệ trong các ca z = 1
#     nhóm cham_* → tỷ lệ trong các ca z = 0
# Cần ghi rõ cách hiểu này trong luận văn.
ATYPICAL = {
    "asd_van_dong_binh_thuong": 0.05,   # trên z = 1
    "asd_bieu_hien_muon":       0.05,   # trên z = 1
    "cham_ngon_ngu_don_thuan":  0.08,   # trên z = 0
    "cham_toan_dien":           0.04,   # trên z = 0
}
ATYPICAL_ON_POSITIVE = ["asd_van_dong_binh_thuong", "asd_bieu_hien_muon"]
ATYPICAL_ON_NEGATIVE = ["cham_ngon_ngu_don_thuan", "cham_toan_dien"]

# Tham số của từng nhóm ca không điển hình (§8.6).
ATYPICAL_PARAMS = {
    "asd_van_dong_binh_thuong": {"normal_domains": ["VDTHO", "VDTINH"]},
    "cham_ngon_ngu_don_thuan":  {"shift": {"GTNN": -1.5}},
    "cham_toan_dien":           {"shift_all": -1.2},
    "asd_bieu_hien_muon":       {"delta_scale": 0.5, "age_below_days": 500},
}

# DHBT không có mốc tuổi nên không dùng được công thức σ(θ + (age − a)/σ).
# Thay bằng: P(có dấu hiệu_j) = σ(−θ_DHBT + DHBT_BASE_LOGIT).
# Với θ ~ N(0,1) (z = 0) cho tỷ lệ nền ≈ 12 %; với θ ~ N(−1.5,1) (z = 1) cho ≈ 38 %,
# tức trung bình khoảng 1,9/5 dấu hiệu ở nhóm dương tính — hợp lý về lâm sàng.
DHBT_BASE_LOGIT = -2.0

# Thời gian làm bài mô phỏng (giây / câu).
SIM_SEC_PER_ITEM_MEAN = 35.0
SIM_SEC_PER_ITEM_SD = 8.0
SIM_SEC_PER_ITEM_ANOMALY = (5.0, 19.0)   # kích hoạt speed_flag (ngưỡng 20 s/câu)

SIM_CONFIG = {
    "n_train":     4000,
    "n_test":      1000,
    "prevalence":  0.03,
    "random_seed": 20260826,
    # [REVIEW-P7.10] Đặc tả ghi 28 ngày, nhưng ngân hàng 20 câu chỉ phủ đủ 6 lĩnh
    # vực từ AGE_BANK_MIN_DAYS = 180 trở lên. Sinh từ 28 ngày sẽ tạo ra hàng loạt
    # hồ sơ thiếu lĩnh vực, làm nhiễu việc đánh giá mà không nói lên điều gì về
    # mô hình. Giữ trần 1095 ngày = phạm vi chính danh 0–36 tháng của đề tài;
    # 8 ca thật vượt mốc này được báo cáo tách nhóm (xem AGE_STRATA).
    "age_min_days": 180,
    "age_max_days": 1095,
}

SANITY_PREVALENCE_TOL = 0.20    # SC-1, dung sai tương đối
SANITY_BYT_MAX_ACC = 0.99       # SC-5 — luật BYT không được đạt 100%


# =====================================================================
# ĐÁNH GIÁ (§9)
# =====================================================================

WILSON_Z = 1.96
N_SENSITIVITY = 200
PERTURB_PCT = 0.10
CALIB_BINS = 10
CALIB_BINS_REAL = 5

# [REVIEW-P1] A0′ là baseline THẬT trên dữ liệu thật: lưới an toàn BYT chạy một
# mình đã cho Se = 100%, Sp = 75%, Acc = 96,15% trên 26 ca. Vì lưới chỉ được
# NÂNG điểm, không cấu hình nào từ A1 trở đi có thể vượt con số đó. Bảng
# ablation bắt buộc phải mở đầu bằng A0′, nếu không toàn bộ Chương 5 vô nghĩa.
ABLATION_CONFIGS = [
    "A0_tong_diem",
    "A0p_luoi_an_toan_BYT",
    "A1_fis_7_luat",
    "A2_d_lien_tuc",
    "A3_them_LAG",
    "A4_them_choquet",
    "A5_them_fknn",
    "A6_mo_hinh_day_du",
    "A6p_owa_thay_choquet",
]


# =====================================================================
# ĐƯỜNG DẪN
# =====================================================================

import os as _os

BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
DATA_DIR = _os.path.join(BASE_DIR, "data")
SYNTHETIC_DIR = _os.path.join(DATA_DIR, "synthetic")
QUESTIONS_CSV = _os.path.join(DATA_DIR, "questions.csv")
RECORDS_JSONL = _os.path.join(DATA_DIR, "records.jsonl")
OUT_DIR = _os.path.join(_os.path.dirname(BASE_DIR), "out")
FIGURES_DIR = _os.path.join(OUT_DIR, "figures")
REPORTS_DIR = _os.path.join(OUT_DIR, "reports")


# =====================================================================
# VĂN BẢN BẮT BUỘC HIỂN THỊ (§12.1)
# =====================================================================

DISCLAIMER = (
    "Đây là công cụ SÀNG LỌC BAN ĐẦU, không phải chẩn đoán y khoa. "
    "Kết quả chỉ mang tính gợi ý và không thay thế đánh giá của bác sĩ "
    "chuyên khoa Phục hồi chức năng Nhi hoặc chuyên gia tâm lý lâm sàng. "
    "Mô hình được kiểm thử trên tập dữ liệu hạn chế và chưa được "
    "thẩm định lâm sàng."
)

SAFETY_NET_NOTICE = (
    "Kết luận này được xác lập theo ngưỡng quy định tại QĐ 2254/QĐ-BYT."
)
