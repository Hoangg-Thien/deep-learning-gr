# BTL01 Function Approximation Implementation Plan

**Goal:** Triển khai, chạy và kiểm chứng toàn bộ BTL01 theo PDF chính thức.
**Architecture:** Các module dữ liệu, MLP, training, metrics, visualization và experiments độc lập; CLI `main.py` điều phối. Test chỉ mở trong giai đoạn đánh giá cuối, sau khi khóa lựa chọn bằng validation.
**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, pytest; pypdf/BeautifulSoup để đọc nguồn; ReportLab và python-pptx để xuất tài liệu.
**Spec:** `MLP_Exercise - Yeu Cau.pdf`, 4 trang; sau đó yêu cầu chi tiết của người dùng.

## Ràng buộc và quyết định
- Giữ nguyên bốn tài liệu gốc; SHA256 ở `source_materials/manifest.json`.
- Gaussian chuẩn hóa 2D: mean [0,0], covariance [[0.3,-0.5],[-0.5,2]], miền [-2,2] x [-4,4].
- Dữ liệu lấy mẫu uniform độc lập: train pool 2000, validation 400, test 100; seed dữ liệu 2026 và SeedSequence tách ba luồng.
- A: prefix 200:200:2000 của train pool, width=10, depth=1. B: width 1..10, N=2000, depth=1. C: depth 1..4, N=2000, width=10/tầng.
- Tác giả triển khai chọn baseline Tanh, output linear, Adam lr=0.01, batch=256, max=600 epochs, patience=80, min_delta=1e-8; seeds khởi tạo 11/29/47, một CPU thread. Không phải tham số cố định của đề.
- Đầu vào được chia theo nửa độ rộng miền [2,4], không fit scaler trên validation/test. Nhãn giữ nguyên đơn vị mật độ xác suất.
- Giữ nguyên số epoch tối đa và batch size; A lớn có nhiều optimizer updates/epoch hơn. Ghi rõ đây là so sánh theo ngân sách epoch, không theo số update.
- Các cấu hình trùng baseline giữa A/B/C được huấn luyện riêng, cùng seed phải cho cùng kết quả. Mỗi run có thư mục/bằng chứng riêng.
- Lựa chọn chung dùng validation MSE trung bình qua seeds; seed đại diện lấy validation tốt nhất trong cấu hình đó, khóa JSON trước khi mở test.
- Test MSE/MAE/RMSE cho mọi cấu hình được báo cáo hậu nghiệm, không dùng sửa hyperparameters. Không cam kết xu hướng đơn điệu hoặc ngưỡng MSE mà đề không đặt ra.
- Báo cáo mẫu chỉ tham khảo bố cục: công thức chuẩn hóa in ở trang 2 và covariance trong code mẫu không thay thế PDF chính thức; không kế thừa việc dùng test làm validation hoặc nhãn “accuracy” cho MSE.
- Thông tin tên/MSSV/nhóm được để trống theo yêu cầu người dùng; thuyết trình và trả lời trực tiếp thuộc công việc của người nộp.

## Các phase và gate
- [x] 1. Đọc toàn bộ 4 tài liệu; trích xuất 25 trang PDF và nội dung HTML, ghi hash nguồn.
- [x] 2. Thiết kế module/config/CLI và ma trận yêu cầu; lựa chọn có ghi rõ ở trên.
- [x] 3. `src/data.py`: gaussian_2d(x, mean, covariance), generate_splits(seed), save/load CSV; tests tâm/đối xứng/tích phân/SPD/shape/range/độc lập/seed. Gate: 11 data tests PASS.
- [x] 4. `src/model.py`, `src/training.py`: MLP(hidden_sizes, activation), fit(train, validation, config, checkpoint); tests backward, giảm loss, early stopping, best checkpoint, seed. Gate: 29 tests PASS tổng cộng.
- [x] 5. `src/experiments.py`, `configs/official.json`: 24 cấu hình x 3 seeds, lưu cấu hình và history mỗi run; tests dải A/B/C và biến kiểm soát. Gate: 31 tests PASS; smoke 6 runs.
- [x] 6. `src/metrics.py`, `src/visualization.py`: MSE/MAE/RMSE; scatter/surface/histogram/loss/comparison/error; tests metrics thủ công. Gate: smoke audit 6 checkpoints PASS; đã xem hình dữ liệu thật.
- [x] 7. Chạy pytest (RED rồi GREEN), smoke E2E trong thư mục riêng; full E2E 72 runs. Gate: 35 tests PASS, 72/72 runs complete, smoke 6 runs PASS.
- [x] 8. Đọc CSV thực tế, phân tích xu hướng/độ lệch seeds/train-validation gap/thời gian; xác minh checkpoint tái tạo prediction. Gate: audit tính lại 72 checkpoints/24 summary configurations PASS; selection C_0004_seed11 theo validation.
- [x] 9. README tiếng Việt, IMPLEMENTATION_REPORT A-H, báo cáo DOCX/PDF 10–20 trang, slide PPTX 10–15 trang; mọi bảng/hình có chú thích. Gate: PDF16 trang; PPTX/PDF14 slides; DOCX11 tables/11 figures; đã xem PDF contact sheets và trang chi tiết.
- [x] 10. Audit từng yêu cầu PDF, kiểm hash gốc/artifact/metrics/history/test results; công cụ đóng ZIP sạch có gate audit và kiểm hash từng thành phần. Gate: audit JSON PASS; không venv/cache/smoke trong archive.

## Review focus
1. Không cho phép covariance không đối xứng/SPD, đầu vào NaN hoặc sai shape.
2. Metrics từ chối shape mismatch để tránh NumPy broadcasting âm thầm.
3. Checkpoint lưu bản clone trạng thái tốt nhất, không tham chiếu trọng số đang cập nhật.
4. Validation/test không ảnh hưởng gradient, scaler hay chọn hyperparameters bằng test.
5. Audit phải phát hiện run thiếu, cấu hình thiếu, CSV/history/checkpoint không nhất quán; smoke không được gộp với kết quả chính thức.

## Lệnh kiểm chứng
`python -m pytest -q --junitxml=results/verification/pytest.xml`
`python main.py --config configs/smoke.json --output results/smoke all`
`python main.py --config configs/official.json all`
`python scripts/build_deliverables.py`
`python scripts/audit.py`

Không tự commit/push repository cha; chỉ sửa trong thư mục bài tập được ủy quyền.

## Ledger quyết định và review cuối

- Không thay baseline hoặc siêu tham số sau khi chạy test; nguồn `src/` không đổi trong suốt 72 runs.
- Review độc lập xác nhận Gaussian/checkpoint/validation-only selection không có blocker; phát hiện hai lỗ hổng audit, đã sửa trong một lượt và thêm 4 regression cases RED→GREEN.
- Audit dữ liệu ưu tiên bản CSV trong checkout/output đang kiểm, không lấy nhầm checkout cũ. Config mỗi run được đối chiếu trực tiếp với experiment spec và training config chính thức.
- Cố định số CPU threads khi audit để khớp môi trường inference đã lưu. Giảm font/spacing vùng text có ảnh trong slide; kiểm mọi shape nằm trong slide.
- Không thực hiện trực tiếp: CUDA tùy chọn, render Word/PowerPoint, thuyết trình tại lớp. Danh tính/đóng góp để trống theo yêu cầu người dùng.
