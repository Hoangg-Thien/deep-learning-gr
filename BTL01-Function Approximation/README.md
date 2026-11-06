# BTL01 – Xấp xỉ hàm Gaussian 2D bằng MLP PyTorch

Bài tập nghiên cứu khả năng xấp xỉ mật độ Gaussian hai chiều bằng mạng MLP, khảo sát số mẫu huấn luyện (A), số neurons (B) và số tầng ẩn (C). Nguồn chính thức là **MLP_Exercise - Yeu Cau.pdf**; tài liệu mẫu chỉ dùng tham khảo bố cục và ý tưởng.

Gaussian có mean `[0,0]`, covariance `[[0.3,-0.5],[-0.5,2]]`; lấy mẫu đầu vào uniform trong `[-2,2] × [-4,4]`. Test độc lập đúng 100 mẫu. Toàn bộ model, backpropagation và optimizer dùng PyTorch.

## Kết quả đã chạy

Đã chạy **72/72 runs** trên CPU, **35 tests PASS**; audit nạp lại 72 checkpoints và tính lại metrics/summary PASS. Báo cáo PDF có **16 trang**, slide có **14 trang**. Chi tiết bằng chứng ở `results/verification/audit.json`.

| Nhóm | Cấu hình được chọn bằng validation trong nhóm | Test MSE mean ± SD (3 seeds) |
| --- | --- | --- |
| A | 2000 mẫu; 1 tầng ×10 neurons | 1.200e-4 ± 3.324e-5 |
| B | 10 neurons; 1 tầng; 2000 mẫu | 1.200e-4 ± 3.324e-5 |
| C | 4 tầng ×10 neurons; 2000 mẫu | 1.039e-6 ± 2.300e-7 |

Model đại diện `C_0004_seed11` được chọn bằng validation, best epoch 592: test **MSE=7.965e-7, MAE=6.430e-4, RMSE=8.925e-4**. Các số này là kết quả thực tế, không phải mục tiêu cam kết cho mọi seed/máy. A/B có cùng baseline nên kết quả trùng là kiểm tra tái lập có chủ ý. 9/72 runs dừng sớm; nhiều runs còn chạm giới hạn epoch.

## Công nghệ và cấu trúc

Python 3.13, PyTorch, NumPy, Matplotlib, pytest. ReportLab/python-docx/python-pptx xuất tài liệu; pypdf/BeautifulSoup đọc tài liệu gốc. `requirements.txt` khóa các phiên bản đã dùng; `reports/environment_freeze.txt` ghi môi trường thực tế.

```text
main.py                    CLI chạy A/B/C hoặc toàn bộ pipeline
src/data.py                Gaussian, sinh độc lập, validate, CSV
src/model.py               nn.Module MLP tùy chỉnh width/depth/activation
src/training.py            DataLoader, train/validation, early stop, checkpoint
src/metrics.py             MSE/MAE/RMSE (đơn vị gốc)
src/experiments.py          Điều phối, lựa chọn validation, đánh giá cuối
src/visualization.py        Biểu đồ tĩnh cho nghiên cứu
tests/                     Automated tests
configs/official.json      24 cấu hình × 3 seeds
configs/smoke.json         6 runs ngắn để kiểm tra đường chạy
data/                      train.csv, validation.csv, test.csv
results/logs/              history và config mỗi run
results/checkpoints/       72 best checkpoints .pt
results/tables/            all_runs, summary, A/B/C và predictions
results/figures/           scatter, surface, histogram, comparison, loss
results/verification/      JUnit XML, test output, audit JSON
results/smoke/             Kết quả smoke, không đưa vào bảng chính thức
reports/                   RTM, báo cáo giải thích, DOCX/PDF học thuật
reports/source_materials/  Bản trích xuất và SHA256 tài liệu gốc
slides/                    Slide thuyết trình PPTX
scripts/                   Đọc nguồn, xuất tài liệu, audit, đóng gói
submission/                ZIP sạch, không chứa venv/cache/smoke
```

Thêm `scripts/` để tách công cụ xuất tài liệu và audit khỏi mô hình. Không cần notebook, server hay đường dẫn máy cá nhân để chạy pipeline.

## Cài đặt Windows + VS Code

1. Cài Python 3.13 bản 64-bit và VS Code cùng extension **Python** của Microsoft.
2. Mở thư mục bài tập bằng **File → Open Folder**, mở PowerShell terminal tại thư mục đó.
3. Kiểm tra Python rồi tạo môi trường cục bộ:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Trong VS Code chọn **Ctrl+Shift+P → Python: Select Interpreter → .venv\Scripts\python.exe**. Có thể kích hoạt môi trường bằng `.\.venv\Scripts\Activate.ps1`, nhưng các lệnh dùng đường dẫn đầy đủ dưới đây không cần activation và không cần thay đổi execution policy toàn cục.

Nếu `python` trỏ vào WindowsApps và không chạy, dùng đường dẫn tuyệt đối tới Python đã cài để thực hiện `-m venv`. Không cần gỡ/cài lại Python toàn hệ thống. Máy tạo kết quả có `.venv` cũ lỗi WindowsApps nên dùng **`.venv-run`** mới và tái sử dụng PyTorch CPU sẵn có bằng `--system-site-packages`. Trên máy này, thay `.venv` bằng `.venv-run` trong các lệnh bên dưới; môi trường của lần chạy nằm trong `reports/environment_freeze.txt`.

Kiểm tra CPU/GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Mặc định cấu hình dùng CPU và 1 thread để giảm nhiễu thời gian cho MLP nhỏ. Có hỗ trợ `device: "auto"`/`"cuda"` nếu có bản PyTorch CUDA phù hợp; lưu config khác và chạy ở output khác. GPU chưa được kiểm chứng trên máy CPU này.

## Chạy từng thực nghiệm

```powershell
.\.venv\Scripts\python.exe main.py A --config configs/official.json
.\.venv\Scripts\python.exe main.py B --config configs/official.json
.\.venv\Scripts\python.exe main.py C --config configs/official.json
```

Mỗi nhóm chạy 3 seeds 11/29/47. A chạy 200,400,…,2000 mẫu với MLP 2–10–1. B chạy 1,…,10 neurons, một tầng ẩn, 2000 mẫu. C chạy 1,…,4 tầng, mỗi tầng 10 neurons, 2000 mẫu. `selection.json` phản ánh các nhóm đã chạy trong output; để lấy lựa chọn trên đủ A/B/C, chạy `all`.

## Chạy E2E và tests

```powershell
# Smoke độc lập, KHÔNG phải kết quả chính thức
.\.venv\Scripts\python.exe main.py all --config configs/smoke.json --output results/smoke
.\.venv\Scripts\python.exe scripts/audit.py --output results/smoke --no-documents

# Huấn luyện và đánh giá toàn bộ 72 runs
.\.venv\Scripts\python.exe main.py all --config configs/official.json

# Kiểm thử có bằng chứng JUnit
.\.venv\Scripts\python.exe -m pytest -q --junitxml=results/verification/pytest.xml

# Tái tạo tài liệu từ các kết quả đã chạy, rồi audit toàn bộ
.\.venv\Scripts\python.exe scripts/build_deliverables.py
.\.venv\Scripts\python.exe scripts/audit.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

`all` bao gồm sinh CSV → kiểm dữ liệu → vẽ dữ liệu → train/validation A/B/C → lưu best model → khóa lựa chọn theo validation → test cuối → metrics/figures/tables. Công cụ build tài liệu đọc CSV, không huấn luyện lại. Audit độc lập nạp lại từng model, tính lại cả 3 metrics, đối chiếu mọi bảng và kiểm số trang/slide/test XML. Chạy audit trước khi kết luận hoàn thành.

## Tái lập và bảo toàn kết quả

- Seed dữ liệu 2026; `SeedSequence.spawn(3)` tách train/validation/test. Dữ liệu giữ cố định qua tất cả model seeds; A dùng các prefix lồng nhau của một train pool.
- Cố định Python/NumPy/PyTorch seeds, DataLoader generator, deterministic algorithms; `num_workers=0`. Không fit preprocessing trên validation/test; chia đầu vào theo `[2,4]` lấy từ miền đã biết, nhãn không scale.
- Baseline: Tanh, output tuyến tính, Adam lr=0.01, batch 256, tối đa 600 epochs, patience 80, min_delta=1e-8. Lưu checkpoint tại validation minimum chính xác; min_delta chỉ điều khiển early stopping.
- Lựa chọn cấu hình theo **mean validation MSE** qua seeds, rồi chọn seed đại diện theo validation thấp nhất. Test được đánh giá hậu nghiệm cho tất cả cấu hình, không dùng để thay đổi cấu hình hay quyết định dừng.
- Không bảo đảm bitwise giống nhau giữa các phiên bản PyTorch, CPU và CUDA. Trong cùng môi trường CPU, tests kiểm train history và prediction trùng hoàn toàn cùng seed.
- Chạy lại cùng output/config sẽ thay thế các artifact được sinh với cùng tên. Dùng `--output results/new_run` để giữ nguyên bằng chứng cũ; dữ liệu của output riêng nằm trong `results/new_run/data`.
- Output đã có config khác sẽ bị từ chối. Smoke và bộ chính thức phải dùng thư mục khác nhau. Pipeline không tự resume một run bị ngắt.

## Đọc kết quả

- `results/tables/summary.csv`: mean và sample SD (`ddof=1`) của MSE/MAE/RMSE, thời gian và epoch cho từng cấu hình.
- `results/tables/all_runs.csv`: metrics train/validation/test của checkpoint tốt nhất, 72 dòng.
- `results/logs/*_config.json`, `results/logs/*.csv`: tham số, loss từng epoch; `optimization_loss` là loss các mini-batch đang cập nhật, `train_loss` đo lại sau epoch.
- `results/checkpoints/*.pt`: state_dict, kiến trúc, config, best epoch/validation loss; nạp an toàn bằng `load_checkpoint` và `weights_only=True`.
- `results/tables/predictions/`: 100 ground-truth/predictions và residual mỗi run.
- `results/selection.json`: lựa chọn được khóa trước test; `results/manifest.json`: môi trường, hash dữ liệu/source và trạng thái chạy.
- `reports/IMPLEMENTATION_REPORT.md`: giải thích module, toán học, kết quả thực tế và hạn chế.
- `reports/ACADEMIC_REPORT.docx` và `.pdf`: báo cáo học thuật; `slides/BTL01_PRESENTATION.pptx`: slide; `reports/REQUIREMENT_TRACEABILITY.md`: đối chiếu PDF chính thức.
- `results/verification/audit.json`: kết quả audit thực tế cuối cùng; không coi việc chỉ có code/tests là hoàn thành E2E.

## Hạn chế

Chỉ 100 điểm test theo đề nên test metrics có biến động; 3 seeds chỉ thay đổi khởi tạo/shuffle, không phải 3 tập dữ liệu độc lập. Ngân sách cố định theo epoch khiến số optimizer updates của A tăng theo N; độ sâu làm tăng số parameters. Không phải mọi cấu hình đều hội tụ đầy đủ hoặc có xu hướng đơn điệu. Output tuyến tính có thể dự đoán mật độ âm nhỏ; không clamp để tránh che sai số. Không thực hiện tuning tiếp sau khi xem test. CUDA và giao diện Word/PowerPoint chưa được kiểm thử trực tiếp trên máy này. Tên, MSSV, thành viên và đóng góp được để trống theo yêu cầu; người nộp tự điền và trình bày/demo trực tiếp.
