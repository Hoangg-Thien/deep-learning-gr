# Requirement Traceability Matrix

Nguồn ưu tiên: `MLP_Exercise - Yeu Cau.pdf` (4 trang), sau đó yêu cầu người dùng. Trạng thái kỹ thuật được xác nhận bằng `results/verification/audit.json`; trình bày trực tiếp và thông tin cá nhân do người nộp thực hiện.

| ID | Nguồn | Yêu cầu | Triển khai | Kiểm chứng | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| R01 | PDF §1–2, tr.1 | Đúng bài toán hồi quy Gaussian và PyTorch | src/data.py; src/model.py | Gaussian analytic/integral tests; checkpoint audit | Đã kiểm chứng |
| R02 | PDF §2–3.1, tr.1 | μ=[0,0], C=[[.3,-.5],[-.5,2]], miền đúng | src/data.py | 11 data tests; data_statistics.csv | Đã kiểm chứng |
| R03 | PDF §2, tr.1 | Test độc lập đúng 100; validation riêng | src/data.py; configs/official.json | audit: shape/hash/no duplicate | Đã kiểm chứng |
| R04 | PDF §3.1, tr.1 | Seed, dữ liệu cố định, chia nhất quán | src/data.py; configs/official.json | SeedSequence + reproducibility tests | Đã kiểm chứng |
| R05 | PDF §3.1, tr.1 | Test không chọn siêu tham số | src/experiments.py | selection.json trước test; unit test bỏ qua test metrics | Đã kiểm chứng |
| R06 | PDF §3.1, tr.1 | Scatter 3D, surface thật, mô tả phân bố | src/visualization.py | data_scatter/gaussian_surface/data_distributions.png + CSV stats | Đã sinh và kiểm tra |
| R07 | PDF §3.2, tr.1 | nn.Module, nn.Linear, width/depth/activation tùy chỉnh | src/model.py | 12 forward/backward cases + invalid architecture tests | Đã kiểm chứng |
| R08 | PDF §3.2, tr.1 | Tensor, Dataset/DataLoader, MSELoss, optimizer | src/training.py | training tests + 72 runs | Đã kiểm chứng |
| R09 | PDF §3.2, tr.1 | Tự train/validation; train/eval/no_grad đúng | src/training.py | source audit + loss/history/checkpoint tests | Đã kiểm chứng |
| R10 | PDF §3.2, tr.1 | Loss mỗi epoch; best checkpoint validation | src/training.py; results/logs | audit 72 history minima và checkpoint predictions | Đã kiểm chứng |
| R11 | PDF §3.2, tr.1 | CPU; GPU tùy chọn | src/training.py | 72 CPU runs; CUDA unavailable | CPU đã kiểm chứng; GPU chưa chạy |
| R12 | PDF §3.2, tr.1 | MSE và MAE/RMSE; y–prediction; surface | src/metrics.py; src/visualization.py | manual metrics test + recompute 72 models + figures | Đã kiểm chứng |
| R13 | PDF §3.2, tr.1 | Không TensorFlow/Keras/AutoML thay thế | src/*.py; requirements.txt | source review + runtime torch metadata | Đã kiểm chứng |
| R14 | PDF §4, tr.2 | Baseline seed/batch/epochs/optimizer/lr/activation/stopping | configs/official.json | 72 *_config.json và checkpoint config audit | Đã kiểm chứng |
| R15 | PDF §4 A, tr.2 | A=200:200:2000, bảng/đồ thị/nhận xét | src/experiments.py; configs/official.json | 30 runs; experiment_A.csv; A_comparison.png; báo cáo E | Đã kiểm chứng |
| R16 | PDF §4 B, tr.2 | B=1..10; 1 tầng; under/overfitting | src/experiments.py | 30 runs; experiment_B.csv; B_comparison.png; báo cáo E | Đã kiểm chứng |
| R17 | PDF §4 C, tr.2 | C=1..4; test error/loss/thời gian | src/experiments.py | 12 runs; C_comparison/convergence/training_time.png | Đã kiểm chứng |
| R18 | PDF §4, tr.2 | Mỗi nhóm thay một yếu tố, nhãn splits rõ | configs/official.json; src/visualization.py | test controlled variables; audit config; labeled figures | Đã kiểm chứng |
| R19 | PDF §4, tr.2 | Lưu tham số, kết quả và giải thích từng cấu hình | results/logs; results/tables; báo cáo | 72 runs/configs/histories + bảng nhận xét 24 cấu hình | Đã kiểm chứng |
| R20 | PDF §4 khuyến khích | Ít nhất 3 seeds; mean ± SD | configs/official.json; src/experiments.py | 72 runs; summary.csv ddof=1; audit tính lại | Đã thực hiện |
| R21 | PDF §5, tr.2 | Module rõ; requirements, README, lệnh A/B/C | src; requirements.txt; README.md | smoke/full CLI chạy thực tế; docs review | Đã kiểm chứng |
| R22 | PDF §5, tr.2 | CSV logs, hình, trọng số .pt; không chỉ HTML | results; data; src | audit kiểm file và nạp 72 .pt | Đã kiểm chứng |
| R23 | PDF §5, tr.2 | Chạy lại không phụ thuộc đường dẫn cá nhân | main.py; scripts; README.md | Path(__file__) roots; manifest hash; E2E smoke/full | Đã kiểm chứng trên máy hiện tại |
| R24 | PDF §5, tr.2 | Tự triển khai PyTorch, giải thích đoạn quan trọng | src; reports/IMPLEMENTATION_REPORT.md | walkthrough C + train equations | Đã thực hiện |
| R25 | PDF §6, tr.2–3 | Báo cáo DOCX và PDF; đề xuất 10–20 trang | scripts/build_deliverables.py | ACADEMIC_REPORT.docx/.pdf; audit số trang | Đã sinh; PDF được kiểm tra |
| R26 | PDF §6.1–6.6 | Giới thiệu, dữ liệu, model, thực nghiệm, thảo luận, kết luận/tham khảo | reports/ACADEMIC_REPORT.* | 16 phần có bảng/hình đánh số và references | Đã thực hiện |
| R27 | PDF §7, tr.3 | Slide PPTX khoảng 10–15, đủ mục tiêu/model/A/B/C/kết luận | slides/BTL01_PRESENTATION.pptx | audit slide count/content; ZIP XML check | Đã sinh; cần người dùng mở PowerPoint rà soát |
| R28 | PDF §7 tùy chọn | PDF slide được khuyến khích | slides/BTL01_PRESENTATION.pdf | pypdf page count; cùng nội dung PPTX | Đã xuất bổ sung |
| R29 | PDF §7, rubric §9 | Thuyết trình 10–15 phút, hỏi đáp và demo tại lớp | slides/PRESENTATION_NOTES.md | demo commands và câu hỏi ôn tập | Người nộp thực hiện trực tiếp |
| R30 | PDF §8, tr.3 | ZIP tên BTL01_Nhom_HoTen_MSSV; thành viên/nhiệm vụ/tổng 100% | scripts/package_submission.py; reports/SUBMISSION_INFO.md | ZIP sạch; mẫu thông tin trống | Để trống theo yêu cầu người dùng; tự điền/đổi tên |
| U01 | User §3–4 | CSV dữ liệu; validation shape/range/leakage | src/data.py; data | data tests và audit | Đã kiểm chứng |
| U02 | User §3–4 | Early stopping; Gaussian/data/model/backward/checkpoint/metrics/seed tests | tests; src/training.py | pytest XML và early-stop test | Đã kiểm chứng |
| U03 | User §3 Evaluation | Predicted/error surface; under/overfitting, không tune test | results/figures; báo cáo E/G | selected surfaces + train/validation gaps | Đã thực hiện |
| U04 | User §6 | README tiếng Việt đủ 13 nội dung | README.md | hướng dẫn Windows/VS Code/venv/rerun/limitations | Đã thực hiện |
| U05 | User §6 | IMPLEMENTATION_REPORT tiếng Việt A–H | reports/IMPLEMENTATION_REPORT.md | generated từ CSV/log/config thật | Đã thực hiện |
| U06 | User §7–10 | Chạy thực tế, giữ tài liệu gốc, không giả số liệu, không venv/cache trong source ZIP | scripts/audit.py; scripts/package_submission.py | hash 4 gốc; 72 checkpoint recompute; archive contents | Đã kiểm chứng kỹ thuật |
