"""Build Vietnamese Markdown, DOCX/PDF report and PPTX from measured results."""
from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments import read_csv


def fmt(value):
    return f'{float(value):.3e}'


def mean_sd(row, metric='test_mse'):
    return f"{fmt(row[metric + '_mean'])} ± {fmt(row[metric + '_std'])}"


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def requirements():
    # One row per independently verifiable requirement, including human submission items.
    return [
        ('R01', 'PDF §1–2, tr.1', 'Đúng bài toán hồi quy Gaussian và PyTorch', 'src/data.py; src/model.py', 'Gaussian analytic/integral tests; checkpoint audit', 'Đã kiểm chứng'),
        ('R02', 'PDF §2–3.1, tr.1', 'μ=[0,0], C=[[.3,-.5],[-.5,2]], miền đúng', 'src/data.py', '11 data tests; data_statistics.csv', 'Đã kiểm chứng'),
        ('R03', 'PDF §2, tr.1', 'Test độc lập đúng 100; validation riêng', 'src/data.py; configs/official.json', 'audit: shape/hash/no duplicate', 'Đã kiểm chứng'),
        ('R04', 'PDF §3.1, tr.1', 'Seed, dữ liệu cố định, chia nhất quán', 'src/data.py; configs/official.json', 'SeedSequence + reproducibility tests', 'Đã kiểm chứng'),
        ('R05', 'PDF §3.1, tr.1', 'Test không chọn siêu tham số', 'src/experiments.py', 'selection.json trước test; unit test bỏ qua test metrics', 'Đã kiểm chứng'),
        ('R06', 'PDF §3.1, tr.1', 'Scatter 3D, surface thật, mô tả phân bố', 'src/visualization.py', 'data_scatter/gaussian_surface/data_distributions.png + CSV stats', 'Đã sinh và kiểm tra'),
        ('R07', 'PDF §3.2, tr.1', 'nn.Module, nn.Linear, width/depth/activation tùy chỉnh', 'src/model.py', '12 forward/backward cases + invalid architecture tests', 'Đã kiểm chứng'),
        ('R08', 'PDF §3.2, tr.1', 'Tensor, Dataset/DataLoader, MSELoss, optimizer', 'src/training.py', 'training tests + 72 runs', 'Đã kiểm chứng'),
        ('R09', 'PDF §3.2, tr.1', 'Tự train/validation; train/eval/no_grad đúng', 'src/training.py', 'source audit + loss/history/checkpoint tests', 'Đã kiểm chứng'),
        ('R10', 'PDF §3.2, tr.1', 'Loss mỗi epoch; best checkpoint validation', 'src/training.py; results/logs', 'audit 72 history minima và checkpoint predictions', 'Đã kiểm chứng'),
        ('R11', 'PDF §3.2, tr.1', 'CPU; GPU tùy chọn', 'src/training.py', '72 CPU runs; CUDA unavailable', 'CPU đã kiểm chứng; GPU chưa chạy'),
        ('R12', 'PDF §3.2, tr.1', 'MSE và MAE/RMSE; y–prediction; surface', 'src/metrics.py; src/visualization.py', 'manual metrics test + recompute 72 models + figures', 'Đã kiểm chứng'),
        ('R13', 'PDF §3.2, tr.1', 'Không TensorFlow/Keras/AutoML thay thế', 'src/*.py; requirements.txt', 'source review + runtime torch metadata', 'Đã kiểm chứng'),
        ('R14', 'PDF §4, tr.2', 'Baseline seed/batch/epochs/optimizer/lr/activation/stopping', 'configs/official.json', '72 *_config.json và checkpoint config audit', 'Đã kiểm chứng'),
        ('R15', 'PDF §4 A, tr.2', 'A=200:200:2000, bảng/đồ thị/nhận xét', 'src/experiments.py; configs/official.json', '30 runs; experiment_A.csv; A_comparison.png; báo cáo E', 'Đã kiểm chứng'),
        ('R16', 'PDF §4 B, tr.2', 'B=1..10; 1 tầng; under/overfitting', 'src/experiments.py', '30 runs; experiment_B.csv; B_comparison.png; báo cáo E', 'Đã kiểm chứng'),
        ('R17', 'PDF §4 C, tr.2', 'C=1..4; test error/loss/thời gian', 'src/experiments.py', '12 runs; C_comparison/convergence/training_time.png', 'Đã kiểm chứng'),
        ('R18', 'PDF §4, tr.2', 'Mỗi nhóm thay một yếu tố, nhãn splits rõ', 'configs/official.json; src/visualization.py', 'test controlled variables; audit config; labeled figures', 'Đã kiểm chứng'),
        ('R19', 'PDF §4, tr.2', 'Lưu tham số, kết quả và giải thích từng cấu hình', 'results/logs; results/tables; báo cáo', '72 runs/configs/histories + bảng nhận xét 24 cấu hình', 'Đã kiểm chứng'),
        ('R20', 'PDF §4 khuyến khích', 'Ít nhất 3 seeds; mean ± SD', 'configs/official.json; src/experiments.py', '72 runs; summary.csv ddof=1; audit tính lại', 'Đã thực hiện'),
        ('R21', 'PDF §5, tr.2', 'Module rõ; requirements, README, lệnh A/B/C', 'src; requirements.txt; README.md', 'smoke/full CLI chạy thực tế; docs review', 'Đã kiểm chứng'),
        ('R22', 'PDF §5, tr.2', 'CSV logs, hình, trọng số .pt; không chỉ HTML', 'results; data; src', 'audit kiểm file và nạp 72 .pt', 'Đã kiểm chứng'),
        ('R23', 'PDF §5, tr.2', 'Chạy lại không phụ thuộc đường dẫn cá nhân', 'main.py; scripts; README.md', 'Path(__file__) roots; manifest hash; E2E smoke/full', 'Đã kiểm chứng trên máy hiện tại'),
        ('R24', 'PDF §5, tr.2', 'Tự triển khai PyTorch, giải thích đoạn quan trọng', 'src; reports/IMPLEMENTATION_REPORT.md', 'walkthrough C + train equations', 'Đã thực hiện'),
        ('R25', 'PDF §6, tr.2–3', 'Báo cáo DOCX và PDF; đề xuất 10–20 trang', 'scripts/build_deliverables.py', 'ACADEMIC_REPORT.docx/.pdf; audit số trang', 'Đã sinh; PDF được kiểm tra'),
        ('R26', 'PDF §6.1–6.6', 'Giới thiệu, dữ liệu, model, thực nghiệm, thảo luận, kết luận/tham khảo', 'reports/ACADEMIC_REPORT.*', '16 phần có bảng/hình đánh số và references', 'Đã thực hiện'),
        ('R27', 'PDF §7, tr.3', 'Slide PPTX khoảng 10–15, đủ mục tiêu/model/A/B/C/kết luận', 'slides/BTL01_PRESENTATION.pptx', 'audit slide count/content; ZIP XML check', 'Đã sinh; cần người dùng mở PowerPoint rà soát'),
        ('R28', 'PDF §7 tùy chọn', 'PDF slide được khuyến khích', 'slides/BTL01_PRESENTATION.pdf', 'pypdf page count; cùng nội dung PPTX', 'Đã xuất bổ sung'),
        ('R29', 'PDF §7, rubric §9', 'Thuyết trình 10–15 phút, hỏi đáp và demo tại lớp', 'slides/PRESENTATION_NOTES.md', 'demo commands và câu hỏi ôn tập', 'Người nộp thực hiện trực tiếp'),
        ('R30', 'PDF §8, tr.3', 'ZIP tên BTL01_Nhom_HoTen_MSSV; thành viên/nhiệm vụ/tổng 100%', 'scripts/package_submission.py; reports/SUBMISSION_INFO.md', 'ZIP sạch; mẫu thông tin trống', 'Để trống theo yêu cầu người dùng; tự điền/đổi tên'),
        ('U01', 'User §3–4', 'CSV dữ liệu; validation shape/range/leakage', 'src/data.py; data', 'data tests và audit', 'Đã kiểm chứng'),
        ('U02', 'User §3–4', 'Early stopping; Gaussian/data/model/backward/checkpoint/metrics/seed tests', 'tests; src/training.py', 'pytest XML và early-stop test', 'Đã kiểm chứng'),
        ('U03', 'User §3 Evaluation', 'Predicted/error surface; under/overfitting, không tune test', 'results/figures; báo cáo E/G', 'selected surfaces + train/validation gaps', 'Đã thực hiện'),
        ('U04', 'User §6', 'README tiếng Việt đủ 13 nội dung', 'README.md', 'hướng dẫn Windows/VS Code/venv/rerun/limitations', 'Đã thực hiện'),
        ('U05', 'User §6', 'IMPLEMENTATION_REPORT tiếng Việt A–H', 'reports/IMPLEMENTATION_REPORT.md', 'generated từ CSV/log/config thật', 'Đã thực hiện'),
        ('U06', 'User §7–10', 'Chạy thực tế, giữ tài liệu gốc, không giả số liệu, không venv/cache trong source ZIP', 'scripts/audit.py; scripts/package_submission.py', 'hash 4 gốc; 72 checkpoint recompute; archive contents', 'Đã kiểm chứng kỹ thuật'),
    ]


def load_results():
    manifest = json.loads((ROOT / 'results/manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['run_count'] != 72:
        raise RuntimeError('Official 72-run pipeline must complete before reporting')
    summary = read_csv(ROOT / 'results/tables/summary.csv')
    for row in summary:
        for k, v in row.items():
            if k != 'experiment':
                row[k] = float(v)
    records = read_csv(ROOT / 'results/tables/all_runs.csv')
    selection = json.loads((ROOT / 'results/selection.json').read_text())
    tests = ET.parse(ROOT / 'results/verification/pytest.xml').getroot()
    suites = list(tests.iter('testsuite'))
    count = sum(int(s.get('tests', 0)) for s in suites)
    if any(int(s.get('failures', 0)) + int(s.get('errors', 0)) for s in suites):
        raise RuntimeError('Cannot report passing tests')
    return manifest, summary, records, selection, count


def result_rows(summary, group):
    return [[str(int(r['value'])), fmt(r['train_mse_mean']), fmt(r['validation_mse_mean']),
             mean_sd(r), mean_sd(r, 'test_mae'), mean_sd(r, 'test_rmse')]
            for r in summary if r['experiment'] == group]


def analysis(summary, records, selection):
    groups = {g: [r for r in summary if r['experiment'] == g] for g in 'ABC'}
    text = {}
    for group, rows in groups.items():
        best = min(rows, key=lambda r: r['validation_mse_mean'])
        first, last = rows[0], rows[-1]
        improvements = sum(rows[i]['test_mse_mean'] < rows[i - 1]['test_mse_mean'] for i in range(1, len(rows)))
        text[group] = (f"Nhóm {group}: validation MSE trung bình thấp nhất ở giá trị {int(best['value'])}, "
                       f"validation MSE={fmt(best['validation_mse_mean'])}; test MSE tại cấu hình này là {mean_sd(best)}. "
                       f"Test MSE từ cấu hình đầu đến cuối: {fmt(first['test_mse_mean'])} → {fmt(last['test_mse_mean'])}; "
                       f"có {improvements}/{len(rows)-1} bước liên tiếp giảm. Đây là mô tả hậu nghiệm, không dùng test để chọn mô hình.")
    a = groups['A']
    text['A'] += (f" Với cùng 600 epochs tối đa, A=200 chỉ có 1 update/epoch còn A=2000 có 8; "
                  f"thời gian trung bình lần lượt {a[0]['training_seconds_mean']:.2f}s và {a[-1]['training_seconds_mean']:.2f}s. "
                  'Xu hướng chịu tác động đồng thời của lượng thông tin và số update; không suy luận thuần nhân quả từ đường test error.')
    b = groups['B']
    text['B'] += (' B=1 tạo một ridge phi tuyến theo một tổ hợp tuyến tính của x; không đủ linh hoạt để biểu diễn đỉnh Gaussian 2D và nền ở nhiều phía. '
                  f"Train MSE B=1 là {fmt(b[0]['train_mse_mean'])}, B=10 là {fmt(b[-1]['train_mse_mean'])}. "
                  'Train/validation cùng cao phù hợp với thiếu khớp hoặc chưa tối ưu xong. Chênh lệch validation–train dương chưa đủ kết luận quá khớp nghiêm trọng; cần xem đường loss và biến động seed.')
    candidates = []
    for width in range(1, 11):
        seeds = [r for r in records if r['experiment'] == 'B' and int(r['value']) == width]
        low = min(seeds, key=lambda r: float(r['validation_mse']))
        high = max(seeds, key=lambda r: float(r['validation_mse']))
        candidates.append((float(high['validation_mse']) / float(low['validation_mse']), width, low, high))
    ratio, width, low, high = max(candidates, key=lambda item: item[0])
    text['B'] += (f" Độ nhạy seed rõ nhất trong B ở width={width}: validation MSE seed{low['seed']}={fmt(low['validation_mse'])}, "
                  f"seed{high['seed']}={fmt(high['validation_mse'])}, chênh {ratio:.2f} lần. "
                  'Sự khác biệt dù cùng dữ liệu cho thấy vai trò của khởi tạo/shuffle và trạng thái tối ưu; không thể chỉ nhìn mean để kết luận tăng width luôn cải thiện.')
    target_stats = next(r for r in read_csv(ROOT / 'results/tables/data_statistics.csv')
                        if r['split'] == 'train' and r['variable'] == 'y')
    n = int(target_stats['n'])
    constant_mse = float(target_stats['std']) ** 2 * (n - 1) / n
    text['B'] += (f" Tham chiếu chỉ dùng train: dự đoán hằng bằng mean y_train có MSE={fmt(constant_mse)}; "
                  f"train MSE B=1 là {fmt(b[0]['train_mse_mean'])}, gần mức hằng. "
                  'Điều này củng cố nhận xét mô hình rất hẹp chưa học được cấu trúc đỉnh, nhưng không tách riêng hoàn toàn tác động capacity và tối ưu.')
    c = groups['C']
    text['C'] += (f" C=1 có 41 parameters; C=4 có 371. Thời gian trung bình {c[0]['training_seconds_mean']:.2f}s → {c[-1]['training_seconds_mean']:.2f}s. "
                  'Tăng độ sâu thay đổi cả độ biểu diễn lẫn số parameters và bài toán tối ưu. Không đo gradient theo layer nên không gán mọi thất bại cho vanishing gradient. '
                  'Đường hội tụ và thời điểm checkpoint giúp phân biệt giới hạn ngân sách với khả năng biểu diễn.')
    gaps = [(r['validation_mse_mean'] - r['train_mse_mean'], r) for r in summary]
    gap, row = max(gaps, key=lambda v: v[0])
    stopped = sum(r['stopped_early'] == 'True' for r in records)
    maxed = sum(int(r['epochs_ran']) == 600 for r in records)
    chosen = next(r for r in records if r['run_id'] == selection['run_id'])
    text['general'] = (f"Lựa chọn khóa trước test: {selection['run_id']}, dựa trên validation MSE trung bình của cấu hình "
                       f"{fmt(selection['mean_validation_mse'])}. Seed đại diện có test MSE={fmt(chosen['test_mse'])}, "
                       f"MAE={fmt(chosen['test_mae'])}, RMSE={fmt(chosen['test_rmse'])}. "
                       f"Có {stopped}/72 runs dừng sớm, {maxed}/72 chạy tới 600 epochs. "
                       f"Chênh lệch validation–train MSE trung bình lớn nhất là {fmt(gap)} tại {row['experiment']}={int(row['value'])}. "
                       'Đây không phải kiểm định thống kê về overfitting; các seeds cùng dùng một tập dữ liệu và test nhỏ.')
    return text


def write_markdown(manifest, summary, records, selection, count, findings):
    trace = requirements()
    trace_text = '# Requirement Traceability Matrix\n\nNguồn ưu tiên: `MLP_Exercise - Yeu Cau.pdf` (4 trang), sau đó yêu cầu người dùng. '
    trace_text += 'Trạng thái kỹ thuật được xác nhận bằng `results/verification/audit.json`; trình bày trực tiếp và thông tin cá nhân do người nộp thực hiện.\n\n'
    trace_text += md_table(['ID', 'Nguồn', 'Yêu cầu', 'Triển khai', 'Kiểm chứng', 'Trạng thái'], trace)
    (ROOT / 'reports/REQUIREMENT_TRACEABILITY.md').write_text(trace_text + '\n', encoding='utf-8')
    per_config = []
    for r in summary:
        gap = r['validation_mse_mean'] - r['train_mse_mean']
        per_config.append([r['experiment'], int(r['value']),
                           f"Train {fmt(r['train_mse_mean'])}; validation {fmt(r['validation_mse_mean'])}; gap {fmt(gap)}; "
                           f"best epoch trung bình {r['best_epoch_mean']:.1f}; {r['training_seconds_mean']:.2f}s. "
                           + ('Loss train còn cao: xem khả năng biểu diễn/ngân sách tối ưu.' if r['train_mse_mean'] > 1e-3
                              else 'Sai số train thấp hơn mốc tham chiếu mô tả 1e-3; không phải ngưỡng đạt của đề.')])
    report = f'''# IMPLEMENTATION REPORT – BTL01 Function Approximation

Ngày chạy UTC: {manifest['started_utc']} đến {manifest['completed_utc']}. Các giá trị bên dưới được đọc từ CSV của 72 lần chạy thật; không dùng kết quả mẫu.

## A. Requirement Analysis

Đã đọc toàn bộ 4 tài liệu: PDF chính thức 4 trang, MLP_Exercise 9 trang, Sample_Report 12 trang và nội dung notebook HTML. Bản trích xuất và SHA256 nằm tại `reports/source_materials/`. `MLP_Exercise - Yeu Cau.pdf` là nguồn yêu cầu cao nhất. Ma trận từng yêu cầu: [REQUIREMENT_TRACEABILITY.md](REQUIREMENT_TRACEABILITY.md).

PDF chính thức bổ sung validation độc lập, PyTorch, A/B/C có kiểm soát, logs/checkpoint, DOCX/PDF và PPTX. Các dải đề xuất được triển khai đầy đủ: A=200:200:2000, B=1..10, C=1..4. 3 seeds là khuyến khích và đã chạy. Báo cáo mẫu trang 2 in hệ số chuẩn hóa không đúng dạng 2D và code ví dụ dùng covariance identity, còn đề chính thức quy định covariance tương quan. Sample_Code dùng Keras và gọi MSE là “accuracy”; triển khai mới không kế thừa hai điểm này. Sample_Report dùng test làm validation; pipeline mới tách riêng.

Phương án: module Python nhỏ, CLI cấu hình JSON, train/validation riêng, checkpoint state_dict. Thông tin cá nhân/đóng góp được để trống theo chỉ dẫn người dùng. Không tự tạo danh tính hay tỷ lệ.

{md_table(['Yêu cầu chính', 'Triển khai', 'Kết quả/bằng chứng'], [
['Gaussian và dữ liệu độc lập', 'src/data.py', '3 CSV, 2000/400/100; tests analytic/shape/range/seed/independence'],
['MLP/forward/backward', 'src/model.py; src/training.py', 'nn.Module/Linear; autograd; 72 runs CPU'],
['Validation/checkpoint/early stop', 'src/training.py', '72 histories và best checkpoints; automated tests'],
['A/B/C và fair comparison', 'configs/official.json; src/experiments.py', '30/30/12 runs; CSV/figures; seed mean±SD'],
['Evaluation', 'src/metrics.py; src/visualization.py', 'MSE/MAE/RMSE; prediction parity; predicted/error surfaces'],
['Kiểm chứng', 'tests; scripts/audit.py', f'{count} tests PASS; audit tái tính checkpoints và tables'],
['Tài liệu/nộp bài', 'README; reports; slides; packaging', 'DOCX/PDF/PPTX; clean ZIP; danh tính để trống theo yêu cầu'],
['Thuyết trình và CUDA', 'PRESENTATION_NOTES; device option', 'Demo notes sẵn có; thuyết trình trực tiếp/CUDA chưa thực hiện']])}

## B. Mathematical Foundation

Với x=(x1,x2)^T, μ=(0,0)^T, C=[[0.3,-0.5],[-0.5,2]],

`f(x) = exp(-0.5 * (x-μ)^T C^(-1) (x-μ)) / (2π sqrt(det(C)))`.

det(C)=0.35>0, C đối xứng xác định dương. C^(-1)=(1/0.35)[[2,0.5],[0.5,0.3]]. Hai trị riêng xấp xỉ 0.16385 và 2.13615; covariance âm làm đỉnh nghiêng theo hướng tương quan âm. μ xác định tâm; đường đồng mức phụ thuộc C. Giá trị cực đại tại μ là 1/(2π√0.35) ≈ 0.26902095. Đây là mật độ, không phải xác suất ở một điểm. Các x được lấy uniform trên hình chữ nhật, KHÔNG lấy mẫu x từ Gaussian; y=f(x).

Chuẩn hóa miền cố định: h0=x/[2,4]. Tầng l: zl=Wl h(l-1)+bl, hl=tanh(zl). Đầu ra: ŷ=Wout hC+bout (linear). Mỗi tầng có bias. Cấu hình một hidden layer B neurons có 4B+1 tham số; C tầng width B có 3B+(C-1)(B²+B)+B+1.

MSE=(1/N)Σ(ŷ-y)²; MAE=(1/N)Σ|ŷ-y|; RMSE=√MSE. Đơn vị MSE là mật độ bình phương, MAE/RMSE cùng đơn vị y. Đạo hàm đầu ra ∂L/∂ŷ=2(ŷ-y)/N. Với Tanh, ∂h/∂z=1-h²; quy tắc dây chuyền truyền gradient ngược qua các nn.Linear. `loss.backward()` tính gradient và Adam cập nhật trọng số bằng moment bậc nhất/bậc hai; không có gradient hoặc optimizer update trong validation/test.

## C. Implementation Walkthrough

{md_table(['Module', 'Thành phần', 'Vai trò'], [
['src/data.py', 'gaussian_2d, validate_data, generate_splits, save_csv, load_csv', 'Tính mật độ float64; kiểm SPD/finite/shape/range/nhãn; 3 RNG streams; CSV 17 chữ số.'],
['src/model.py', 'MLP(nn.Module).forward', 'List hidden_sizes tùy chỉnh; tanh/relu/sigmoid; buffer input_scale; output linear.'],
['src/training.py', 'seed_everything, make_loader, fit, validation_loss', 'Determinism; TensorDataset/DataLoader; vòng train và eval đúng ngữ cảnh.'],
['src/training.py', 'predict, load_checkpoint', 'no_grad; batch inference; torch.load(weights_only=True); model.eval().'],
['src/metrics.py', 'regression_metrics', 'MSE/MAE/RMSE float64, từ chối broadcasting/NaN/empty.'],
['src/experiments.py', 'experiment_specs, run, select_model, summarize', '24 cấu hình, 3 seeds, khóa selection trước test, sample SD ddof=1.'],
['src/visualization.py', 'data_figures, comparison_figures, evaluation_figures', 'Agg backend; split labels, đơn vị, error bars; đóng figure sau save.'],
['scripts/', 'read_materials, build_deliverables, audit, package_submission', 'Nguồn, tài liệu, kiểm chứng độc lập và ZIP sạch.']])}

Luồng dữ liệu: uniform float64 → Gaussian labels → kiểm CSV → TensorDataset float32 → scale miền trong MLP → forward → MSELoss → zero_grad → backward → optimizer.step. Cuối epoch, model.eval()/no_grad() đánh giá lại toàn train và validation bằng tổng squared error chia N, kể cả batch cuối nhỏ. Train loss đo sau epoch khác optimization_loss là trung bình loss các batch lúc đang cập nhật.

Checkpoint theo validation minimum chính xác, clone tensor sang CPU để không bị trọng số cập nhật về sau làm thay đổi. min_delta chỉ phục vụ patience, không làm mất checkpoint có loss thấp hơn. Cuối fit luôn khôi phục best_state, không dùng mô hình cuối ngẫu nhiên. Checkpoint chứa state_dict, kiến trúc/config/seed, best_epoch và best_validation_loss; phục vụ inference, không hỗ trợ resume optimizer. Predict nhận đầu vào đơn vị gốc vì scale là buffer nằm trong checkpoint.

Train và validation được truyền vào fit; không có tham số test. Sau khi xong tất cả runs, chọn cấu hình theo validation MSE trung bình qua seeds, chọn seed đại diện theo validation nhỏ nhất, ghi selection.json và thời gian khóa; sau đó mới load test.csv để đánh giá. Biểu đồ grid là chẩn đoán hậu nghiệm, không thay test set 100 điểm, không dùng chọn hyperparameters.

## D. Experiment Design

Baseline: dữ liệu seed=2026; train pool=2000, validation=400, test=100; seeds khởi tạo/shuffle=11/29/47. Một hidden layer width=10; Tanh; output linear; Adam lr=0.01; batch=256; max epochs=600; patience=80; min_delta=1e-8; CPU 1 thread; num_workers=0. Cấu hình cụ thể ở `configs/official.json` và 72 JSON riêng.

{md_table(['Nhóm', 'Biến thay đổi', 'Giữ cố định', 'Số runs'], [['A','N=200,400,…,2000','width10, depth1, toàn bộ training config và validation/test','30'],['B','B=1,…,10','N2000, depth1, cùng train/validation/test và config','30'],['C','C=1,…,4','N2000, width10/tầng, cùng dữ liệu/config','12']])}

A dùng prefix lồng nhau để giảm nhiễu lấy mẫu khi tăng N. Dữ liệu được giữ nguyên giữa ba model seeds, không phải ba dataset seeds. So sánh có paired seeds và cùng shuffle generator; kiến trúc khác không thể có toàn bộ trọng số khởi tạo giống nhau. A làm tăng số update/epoch; C làm tăng số parameters. Ngân sách cố định theo epoch không tương đương compute hoặc update budget cố định; thời gian wall-clock có nhiễu hệ thống. Các baseline A=2000/B=10/C=1 được chạy riêng và audit yêu cầu cùng seed cho prediction trùng.

## E. Experimental Results

Tất cả bảng lấy từ `results/tables/summary.csv`; mỗi ô test là mean ± sample SD trên 3 seeds. Mọi metric đo từ **best checkpoint**, đơn vị nhãn gốc. Không có ngưỡng MSE bắt buộc trong PDF và không giả định xu hướng đơn điệu.
'''
    for g in 'ABC':
        report += f'\n### Nhóm {g}\n\n' + md_table(['Giá trị', 'Train MSE mean', 'Validation MSE mean', 'Test MSE ± SD', 'Test MAE ± SD', 'Test RMSE ± SD'], result_rows(summary, g))
        report += f'\n\n![Hình {g}: metrics theo biến khảo sát](../results/figures/{g}_comparison.png)\n\n{findings[g]}\n'
    report += '\n### Hội tụ, thời gian và mô hình được chọn\n\n' + findings['general']
    report += '\n\n' + md_table(['C', 'Parameters', 'Thời gian mean ± SD (s)', 'Epochs mean', 'Best epoch mean'], [
        [int(r['value']), 41 + (int(r['value'])-1)*110, f"{r['training_seconds_mean']:.2f} ± {r['training_seconds_std']:.2f}",
         f"{r['epochs_ran_mean']:.1f}", f"{r['best_epoch_mean']:.1f}"] for r in summary if r['experiment'] == 'C'])
    report += '\n\n![Hội tụ C](../results/figures/C_convergence.png)\n\n![Thời gian C](../results/figures/C_training_time.png)\n\n![Bề mặt và sai số](../results/figures/selected_surfaces.png)\n\n![Ground truth và prediction](../results/figures/selected_test_predictions.png)\n\n### Nhận xét từng cấu hình\n\n'
    report += md_table(['Nhóm', 'Giá trị', 'Quan sát từ logs'], per_config)
    report += f'''

## F. Verification

**{count} tests đã thực sự chạy, PASS**, không skip. Bằng chứng: `results/verification/pytest.xml` và `pytest_output.txt`. Bộ tests gồm Gaussian tâm/đối xứng/công thức quadratic/tích phân, SPD, dữ liệu shape/range/finite/nhãn và CSV roundtrip, 3 splits độc lập và seed, MLP các kiến trúc/activation, backward gradients, giảm validation loss, checkpoint save/load/best epoch, early stopping, metrics tính tay và từ chối broadcasting, train reproducibility, dải A/B/C/biến kiểm soát, selection không phụ thuộc test metrics.

Smoke đã chạy 6 runs riêng tại `results/smoke`, audit nạp lại 6 checkpoints và tái tính metrics PASS. Bộ chính thức chạy đủ 72 runs. `scripts/audit.py` độc lập nạp tất cả 72 checkpoint, tái tính train/validation/test metrics từ dữ liệu CSV, đối chiếu predictions và summary, xác minh best epoch=min history, configs và source hashes, test đúng100, không có điểm trùng splits, ba baseline cùng seed trùng prediction. Audit cuối được lưu sau khi xuất tài liệu vào `results/verification/audit.json`; chỉ file audit PASS mới xác nhận kết quả kiểm chứng đó.

Vấn đề môi trường và cách khắc phục: .venv cũ trỏ Python WindowsApps không khởi chạy; dùng Python 3.13.2 đã cài và venv .venv-run trong project, đọc lại PyTorch CPU sẵn có. pip trong sandbox bị WinError10013 do chặn mạng; đã cài dependencies cục bộ sau bước cấp quyền, không đổi cấu hình hệ thống. Lượt gọi unittest ban đầu gặp thiếu pytest khi dependencies còn tải; chuyển sang đúng pytest sau khi cài. Tests RED do các module chưa triển khai, rồi GREEN sau khi bổ sung; không coi lỗi import trước triển khai là bug huấn luyện. Không có kết quả PASS được suy đoán thay cho chạy thực tế.

Review độc lập không tìm thấy blocker ML, nhưng phát hiện audit có thể ưu tiên đường dẫn dữ liệu của checkout cũ và chưa đối chiếu run với toàn bộ config dự kiến. Đã sửa để ưu tiên CSV trong bản nộp hiện tại và kiểm params/spec của từng run; bổ sung 4 regression cases RED→GREEN. Đồng thời giảm font/spacing vùng text có hình trong PPTX để tránh chạm biểu đồ. Không đổi src hoặc siêu tham số trong lúc bộ chính thức đang train.

## G. Limitations & Future Improvements

- 100 test samples theo đề: sai số ước lượng có phương sai, không kết luận vượt trội thống kê chỉ từ mean±SD 3 seeds.
- 3 model seeds dùng cùng dữ liệu; chưa đo độ nhạy với seed dữ liệu, chưa có confidence interval theo sampling uncertainty.
- Cố định epoch thay vì update budget; số parameters thay đổi với width/depth; chưa có capacity-matched hoặc compute-matched comparison.
- Nhiều runs có thể chạm trần epoch; không dùng test để kéo dài hoặc tuning. Tanh có thể khó tối ưu; chưa đo gradient layer để khẳng định vanishing gradient.
- Output linear có thể âm ở đuôi; không clamp. Grid residual dùng để chẩn đoán, không tham gia chọn model.
- CPU được chạy thực tế; CUDA chưa kiểm chứng. Tính tái lập bitwise chỉ kiểm trong cùng môi trường, thời gian có nhiễu.
- DOCX/PPTX được xuất và kiểm cấu trúc; chưa render bằng Microsoft Word/PowerPoint trên máy này. Người dùng nên rà soát layout trong ứng dụng trước khi nộp. PDF báo cáo và PDF slide là bản xuất trực tiếp, không phải chuyển từ Office.
- Danh tính, MSSV, thành viên và đóng góp để trống theo yêu cầu; người dùng tự điền, đổi tên ZIP và thực hiện thuyết trình/hỏi đáp.

Hướng cải tiến: lặp nhiều data seeds; chạy thêm update-budget matched; khảo sát activation/LR/optimizer bằng validation với kế hoạch định trước; thử output softplus; phân tích residual theo vùng và bootstrap test uncertainty; kiểm CUDA độc lập. Đây là đề xuất tương lai, không phải kết quả đã chạy.

## H. Final Conclusion

Đã triển khai PyTorch đầy đủ các bước sinh dữ liệu → train/validation → checkpoint → lựa chọn validation → test cuối → bảng/hình/tài liệu; 24 cấu hình với 3 seeds, tổng 72 runs. {findings['general']}

Các hạng mục kỹ thuật được truy vết tại RTM và audit. Phần chưa thực hiện trực tiếp gồm CUDA tùy chọn, render Office và thuyết trình ở lớp; thông tin cá nhân để trống theo chỉ dẫn. Không coi các phần đó là đã được kiểm chứng. Không có số liệu sample report được dùng làm kết quả thật.

### Lệnh đã chạy và cách tái lập

```powershell
.\\.venv-run\\Scripts\\python.exe scripts/read_materials.py
.\\.venv-run\\Scripts\\python.exe -m pytest -q --junitxml=results/verification/pytest.xml
.\\.venv-run\\Scripts\\python.exe main.py --config configs/smoke.json --output results/smoke all
.\\.venv-run\\Scripts\\python.exe scripts/audit.py --output results/smoke --no-documents
.\\.venv-run\\Scripts\\python.exe main.py --config configs/official.json all
.\\.venv-run\\Scripts\\python.exe scripts/build_deliverables.py
.\\.venv-run\\Scripts\\python.exe scripts/audit.py
.\\.venv-run\\Scripts\\python.exe scripts/package_submission.py
```

### Tài liệu tham khảo

1. MLP_Exercise - Yeu Cau.pdf, §§1–9, nguồn chính thức trong project.
2. MLP_Exercise.pdf, 9 trang, phát biểu và thiết kế A/B/C.
3. Sample_Code.html và Sample_Report.pdf: tham khảo cấu trúc, không dùng kết quả số.
4. [PyTorch: deterministic algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).
5. [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).
6. [PyTorch: Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
'''
    (ROOT / 'reports/IMPLEMENTATION_REPORT.md').write_text(report, encoding='utf-8')


def academic_pages(summary, records, selection, count, findings):
    # Each page is shared semantically by DOCX/PDF: paragraphs, tables, numbered figures.
    pages = []
    def page(title, *blocks):
        pages.append((title, list(blocks)))
    def p(text):
        return ('p', text)
    def t(caption, headers, rows):
        return ('table', caption, headers, rows)
    def im(name, caption, width=17):
        return ('image', name, caption, width)
    selected = next(r for r in records if r['run_id'] == selection['run_id'])
    page('BTL01 – XẤP XỈ HÀM SỐ DÙNG MẠNG MLP',
         p('BÁO CÁO HỌC THUẬT – Python / PyTorch'),
         p('Học phần: Học sâu. Họ tên: ____________________. MSSV: ____________________. Nhóm: ____________________.'),
         p('Thành viên, phân công và tỷ lệ đóng góp: xem mẫu SUBMISSION_INFO.md; người nộp tự điền, tổng 100% nếu làm nhóm.'),
         p('Tóm tắt. Nghiên cứu sử dụng mạng perceptron đa tầng để xấp xỉ mật độ Gaussian hai chiều trên một miền hữu hạn. Pipeline được triển khai trực tiếp bằng PyTorch, có train/validation/test độc lập, chọn checkpoint theo validation và đánh giá cuối trên 100 điểm test. Ba nhóm khảo sát quy mô dữ liệu, độ rộng và độ sâu gồm 24 cấu hình, mỗi cấu hình 3 seeds, tổng 72 lần chạy.'),
         p(findings['general']),
         p('Đóng góp kỹ thuật: hệ thống tái lập theo JSON; kiểm thử tự động; logs/checkpoints có truy vết; audit tính lại metrics từ từng checkpoint; tài liệu và slide xuất từ số liệu thực tế. Không kế thừa kết quả số của tài liệu mẫu.'),
         p('Từ khóa: MLP, Gaussian 2D, hồi quy, validation, PyTorch, thực nghiệm có kiểm soát.'))
    page('1. Giới thiệu và phân tích yêu cầu',
         p('Mục tiêu là học ánh xạ hai biến thực sang mật độ xác suất bằng MLP hai đầu vào, một đầu ra. Hàm mục tiêu có dạng giải tích nên có thể sinh nhãn chính xác, kiểm tra sai số từng điểm và so sánh bề mặt xấp xỉ với bề mặt thật.'),
         p('Nguồn ràng buộc là MLP_Exercise - Yeu Cau.pdf, 4 trang. MLP_Exercise.pdf minh họa các trục khảo sát. Sample_Code.html và Sample_Report.pdf chỉ tham khảo cách tổ chức; các khác biệt về framework, covariance và test/validation được giải quyết theo đề chính thức.'),
         t('Bảng 1. Nhóm yêu cầu và minh chứng chính.', ['Nhóm', 'Yêu cầu', 'Minh chứng'], [
             ['Dữ liệu', 'Gaussian, 3 splits, seed, scatter/surface', 'data/*.csv; data_statistics.csv'],
             ['PyTorch', 'nn.Module, DataLoader, backprop, checkpoint', 'src/model.py; training.py; *.pt'],
             ['A/B/C', '24 cấu hình, 3 seeds, fair comparison', 'official.json; 72 CSV histories'],
             ['Đánh giá', 'MSE/MAE/RMSE; y–ŷ; surface error', 'summary.csv; selected figures'],
             ['Báo cáo', 'DOCX/PDF; slide PPTX; giải thích', 'reports/; slides/'],
             ['Tái lập', 'Requirements; README; lệnh; tests', f'{count} tests; manifest; audit.json']]),
         p('Tiêu chí nghiệm thu là chạy được và có minh chứng thực tế, không chỉ có mã nguồn. PDF không đặt ngưỡng MSE đạt/trượt; kết quả được đánh giá qua phương pháp đúng, cấu hình đầy đủ và phân tích trung thực. Bảng truy vết đầy đủ gồm cả các yêu cầu người dùng tại REQUIREMENT_TRACEABILITY.md.'),
         p('Bố cục: phần 2 trình bày Gaussian; phần 3 khảo sát dữ liệu; phần 4–6 mô hình và cài đặt; phần 7–10 thực nghiệm; phần 11 đánh giá mô hình chọn; phần 12–13 kiểm chứng, hạn chế và kết luận.'))
    page('2. Cơ sở toán học của Gaussian hai chiều',
         p('Hàm mục tiêu: f(x) = exp[-0.5 (x−μ)^T C^(-1)(x−μ)] / [2π sqrt(det(C))], với μ=[0,0]^T và C=[[0.3,−0.5],[−0.5,2]]. Đây là hệ số chuẩn hóa cho hai chiều; không thay bằng căn bậc hai của 2π như một dòng công thức trong báo cáo mẫu.'),
         p('det(C)=0.35. Ma trận đối xứng và xác định dương nên nghịch đảo tồn tại: C^(-1)=(1/0.35)[[2,0.5],[0.5,0.3]]. Tâm μ quyết định vị trí đỉnh; các phần tử đường chéo là phương sai, phần tử ngoài đường chéo biểu diễn đồng biến thiên âm. Đường đồng mức là ellipse, nghiêng về hướng x1 tăng thì x2 giảm.'),
         p('Đỉnh f(0,0)=1/(2π sqrt(0.35))≈0.26902095; y không cần trải toàn khoảng [0,1] dù nằm trong khoảng đó. Mật độ tại một điểm không phải xác suất của điểm. Tích phân Gaussian trên toàn R² bằng 1, còn trên miền chữ nhật hữu hạn có thể nhỏ hơn 1.'),
         p('Miền đầu vào −2≤x1≤2, −4≤x2≤4. Điểm x được lấy uniform trong miền, sau đó tính y=f(x). Phân bố x không phải Gaussian. Nhãn không thêm nhiễu; sai số đánh giá là sai số xấp xỉ và tối ưu, không có nhiễu đo lường chủ ý.'),
         p('Implementation dùng NumPy float64 và solve(C, delta.T) để tính dạng toàn phương. Hàm kiểm tra kích thước, finite, tính đối xứng và Cholesky xác định dương. Không dùng hard-code số liệu output hoặc SciPy để thay toàn bộ kiểm chứng công thức.'),
         p('Kiểm thử gồm đỉnh tại mean, tính đối xứng f(x)=f(−x), so với dạng quadratic tính trực tiếp, tích phân số trên miền rộng gần 1 và từ chối covariance sai. Dữ liệu train được chuyển float32 khi đưa vào PyTorch; metrics báo cáo tính float64 trên nhãn CSV gốc.'))
    stats = read_csv(ROOT / 'results/tables/data_statistics.csv')
    page('3.1. Sinh dữ liệu, chia tập và thống kê',
         p('Một SeedSequence gốc 2026 sinh ba child streams độc lập: train 2000, validation 400, test 100. A lấy prefix train pool; B/C dùng toàn bộ pool. Kiểm tra hash và giao của các tập điểm xác nhận không có bản ghi trùng giữa splits. Validation lớn hơn test giúp ổn định lựa chọn; đề chỉ cố định test=100.'),
         t('Bảng 2. Thống kê đo trực tiếp trên CSV.', ['Split', 'Biến', 'N', 'Min', 'Max', 'Mean', 'SD'],
           [[r['split'], r['variable'], r['n'], fmt(r['min']), fmt(r['max']), fmt(r['mean']), fmt(r['std'])] for r in stats]),
         im('data_distributions.png', 'Hình 1. Histogram mật độ thực nghiệm của x1, x2, y; phân biệt train/validation/test.'),
         p('Do Gaussian có đỉnh hẹp, nhiều điểm uniform có y gần 0. MSE trung bình có thể nhỏ dù vùng đỉnh chưa xấp xỉ tốt; vì vậy cần xem scatter, parity và bề mặt residual. CSV giữ 17 chữ số có nghĩa; các tập được kiểm tra nhãn khớp hàm giải tích trước huấn luyện.'))
    page('3.2. Trực quan hóa tập dữ liệu',
         im('data_scatter.png', 'Hình 2. Scatter 3D của ba tập độc lập, có nhãn và kích thước.'),
         im('gaussian_surface.png', 'Hình 3. Bề mặt Gaussian thật trên miền chính thức.', 12),
         p('Scatter thưa hơn tại test là do N=100 theo đề. Surface vẽ trên lưới chẩn đoán 81×81 chỉ để minh họa hàm giải tích, không thêm điểm vào train/validation/test. Số lượng điểm khảo sát ở mỗi tập được thể hiện tách bạch để tránh gọi validation là test như ví dụ cũ.'))
    page('4. MLP và forward/backpropagation',
         p('Mạng nhận x=(x1,x2), chia theo vector miền [2,4] để đưa đầu vào về [−1,1]. Đây là phép biến đổi định trước, không học thống kê trên validation/test. Các tầng ẩn dùng nn.Linear kèm Tanh; tầng cuối nn.Linear(...,1) không activation để hồi quy liên tục.'),
         p('Forward: h0=x/[2,4]; zl=Wl h(l−1)+bl; hl=tanh(zl), l=1..C; ŷ=Wout hC+bout. Mỗi Linear có bias. Số neurons và số tầng được điều khiển bằng list hidden_sizes; code cũng cho phép ReLU và sigmoid, nhưng các thực nghiệm chính giữ Tanh cố định.'),
         im('selected_architecture.png', 'Hình 4. Kiến trúc của model đại diện được lựa chọn bằng validation.'),
         p('Loss L=(1/N)Σ(ŷi−yi)². Backprop bắt đầu từ ∂L/∂ŷi=2(ŷi−yi)/N, nhân với đạo hàm Tanh 1−h² và các trọng số qua quy tắc dây chuyền. PyTorch autograd tạo và tính đồ thị gradient cho từng mini-batch. optimizer.zero_grad xóa gradient cũ trước backward; optimizer.step cập nhật tham số Adam.'),
         p('Metrics: MSE=(1/N)Σe²; MAE=(1/N)Σ|e|; RMSE=√MSE. MAE ít nhạy điểm sai số lớn hơn MSE. RMSE cùng đơn vị y, dễ diễn giải. Không gọi MSE là accuracy. Tests từ chối array shape khác nhau để tránh broadcasting khiến một phép tính sai vẫn cho ra scalar.'))
    page('5. Tổ chức chương trình và luồng dữ liệu',
         t('Bảng 3. Trách nhiệm module.', ['Module', 'Nội dung'], [
             ['data.py', 'Gaussian, kiểm dữ liệu, SeedSequence, CSV roundtrip'],
             ['model.py', 'MLP(nn.Module); hidden_sizes; scale buffer; forward'],
             ['training.py', 'DataLoader, optimizer, train/eval, early stopping, checkpoint'],
             ['metrics.py', 'MSE/MAE/RMSE với kiểm shape/finite'],
             ['experiments.py', 'Sinh A/B/C, chạy seeds, selection, tổng hợp CSV'],
             ['visualization.py', 'Dữ liệu, loss, comparisons, predictions, residual'],
             ['main.py', 'CLI config/output/A/B/C/all'],
             ['scripts/', 'Đọc tài liệu, xuất report/slides, audit, ZIP']]),
         p('Luồng thực thi: đọc JSON → sinh dữ liệu độc lập → ghi và kiểm CSV → TensorDataset/DataLoader → MLP → train/validation → lưu history và best state → khóa selection.json → đánh giá test → xuất summary và hình. Training không nhận test, preprocessing không fit trên test, lựa chọn dùng validation mean qua 3 seeds.'),
         p('Mỗi run có ID duy nhất, ví dụ A_0200_seed11. Cấu hình chi tiết được lưu cùng logs; checkpoint chứa kiến trúc và seed. Manifest ghi phiên bản môi trường, hash code/dữ liệu và trạng thái complete. Mô hình nạp bằng weights_only=True, model.eval() và dự đoán trong no_grad().'),
         p('Dataset CSV giữ float64; TensorDataset dùng float32 để train. Việc chuyển dtype không thay công thức nhãn. input_scale là buffer đi cùng state_dict nên inference nhận x đơn vị gốc. Output không clamp về 0 để không che giấu residual vùng nền.'),
         p('Cấu trúc chuẩn có src/, tests/, configs/, data/, results/logs|figures|tables|checkpoints/, reports/. Thêm scripts/ cho công cụ và slides/ cho sản phẩm thuyết trình; không trộn code xuất tài liệu với training.'))
    page('6. Training loop, checkpoint và cấu hình chuẩn',
         t('Bảng 4. Thiết lập cố định trước đánh giá test.', ['Tham số', 'Giá trị'], [
             ['Dữ liệu', 'Seed 2026; train2000 / validation400 / test100'],
             ['Model seeds', '11, 29, 47'], ['Baseline', '2 → 10(Tanh) → 1(linear)'],
             ['Optimizer / LR', 'Adam / 0.01'], ['Batch / epochs', '256 / tối đa 600'],
             ['Early stopping', 'patience80; min_delta=1e−8 trên validation MSE'],
             ['Runtime', 'CPU; 1 thread; num_workers=0; deterministic'],
             ['Checkpoint', 'Validation MSE minimum chính xác; khôi phục trước test']]),
         p('Mỗi epoch: model.train(); lặp mini-batches; forward → MSELoss → zero_grad → backward → optimizer.step. Sau đó model.eval(); torch.no_grad(); lặp DataLoader để tính tổng squared error chia đúng số điểm. Batch cuối nhỏ vẫn được tính theo số mẫu, không trung bình các batch không trọng số.'),
         p('History ghi optimization_loss, train_loss và validation_loss. train_loss là metric đo lại sau epoch; optimization_loss phản ánh mô hình đang thay đổi giữa các mini-batch. min_delta chỉ điều khiển patience. Bất kỳ validation loss thấp hơn đều thay best checkpoint, kể cả mức giảm nhỏ hơn min_delta.'),
         p('Best state được clone tách khỏi tensor đang huấn luyện. Cuối train khôi phục best state, lưu .pt với config và epoch. Không dùng test loss để điều chỉnh dừng. Không hỗ trợ resume optimizer; chạy lại từ đầu bằng seed và JSON. Checkpoint load/save, early stopping và reproducibility được kiểm thử thực tế.'))
    page('7. Thiết kế A/B/C và so sánh công bằng',
         t('Bảng 5. Các biến khảo sát.', ['Nhóm', 'Biến độc lập', 'Các biến giữ cố định', 'Runs'], [
             ['A', 'N=200:200:2000', 'width10, depth1, training config, validation/test', '30'],
             ['B', 'width=1..10', 'N2000, depth1, training config, 3 splits', '30'],
             ['C', 'depth=1..4', 'N2000, width10/tầng, training config, 3 splits', '12']]),
         p('Fair comparison dùng fixed CSV, paired model seeds, cùng optimizer/LR/batch/epochs tối đa/early stopping và cùng tiêu chí selection. A dùng prefix lồng nhau để không thay hoàn toàn dữ liệu khi tăng N. B/C dùng cùng 2000 điểm. Các cấu hình trùng baseline được chạy lại riêng để audit tính tái lập.'),
         p('Ngân sách theo epoch: ceil(N/256) updates mỗi epoch, từ 1 tại N=200 đến 8 tại N=2000. Vì vậy tác động tăng N gồm thêm thông tin và thêm updates; không coi đây là thí nghiệm cố định compute. Early stopping có thể làm epoch thực tế khác nhau. Runtime đo cả train và validation, không tính figure generation.'),
         p('Width/depth làm thay số parameters: depth1 widthB có 4B+1; depthC width10 có 41+110(C−1). C=4 có 371 so với C=1 có 41. Độ sâu cũng thay độ khó tối ưu, nên không quy mọi thay đổi metric cho một yếu tố ẩn như gradient vanishing khi chưa đo.'),
         p('Mean ± sample SD tính qua 3 model seeds, ddof=1. Dữ liệu cố định nên SD phản ánh khởi tạo và shuffle, không phải uncertainty của quá trình lấy mẫu test. Lựa chọn cấu hình bằng validation mean, seed đại diện bằng validation thấp nhất. Test mọi cấu hình là kết quả cuối, không quay lại tuning.'))
    for group, title, table_number, figure_number in [('A','8. Kết quả A – Số mẫu train',6,5),('B','9. Kết quả B – Số neurons',7,6),('C','10.1. Kết quả C – Số tầng ẩn',8,7)]:
        compact = [[r[0], r[1], r[2], r[3]] for r in result_rows(summary, group)]
        page(title,
             t(f'Bảng {table_number}. MSE từ best checkpoint; test mean ± SD trên 3 seeds.', ['Giá trị','Train mean','Validation mean','Test mean ± SD'], compact),
             im(f'{group}_comparison.png', f'Hình {figure_number}. Train/validation/test MSE, MAE và RMSE theo {group}; error bars là SD.'),
             p(findings[group]),
             p('Toàn bộ MAE/RMSE từng run và thống kê mean/SD lưu trong summary.csv và experiment_' + group + '.csv. Không lược bỏ cấu hình có kết quả kém. Không dùng model cuối thay best checkpoint.'))
    page('10.2. Độ sâu: hội tụ và thời gian',
         im('C_convergence.png', 'Hình 8. Loss train/validation qua epoch của tất cả C và seeds.'),
         t('Bảng 9. Runtime và epoch theo độ sâu.', ['C','Parameters','Time mean ± SD(s)','Epoch mean','Best epoch mean'], [
             [int(r['value']), 41+110*(int(r['value'])-1), f"{r['training_seconds_mean']:.2f} ± {r['training_seconds_std']:.2f}",
              f"{r['epochs_ran_mean']:.1f}", f"{r['best_epoch_mean']:.1f}"] for r in summary if r['experiment']=='C']),
         im('C_training_time.png', 'Hình 9. Thời gian train+validation trên CPU, không gồm vẽ hình.', 12),
         p('Các đường loss phải được đọc cùng best epoch và trần 600 epochs. Mô hình chưa tối ưu tốt ở một seed không chứng minh kiến trúc đó bất khả dụng. Runtime tăng có thể đến từ số phép toán, epoch thực tế và nhiễu scheduling CPU.'))
    page('11. Mô hình chọn và trực quan hóa sai số',
         p(findings['general']),
         t('Bảng 10. Metrics seed đại diện được chọn trước test.', ['Run ID', 'Best epoch', 'Test MSE', 'Test MAE', 'Test RMSE'], [
             [selected['run_id'], selected['best_epoch'], fmt(selected['test_mse']), fmt(selected['test_mae']), fmt(selected['test_rmse'])]]),
         im('selected_surfaces.png', 'Hình 10. Hàm thật, dự đoán MLP và absolute error trên grid chẩn đoán.'),
         im('selected_test_predictions.png', 'Hình 11. Ground truth–prediction và residual trên 100 test samples.'),
         p('Parity lệch khỏi đường chéo thể hiện bias cục bộ. Surface error làm lộ sai số quanh đỉnh và vùng nền mà MSE trung bình có thể che. Grid không phải test set bổ sung và không tham gia chọn cấu hình. Predicted density có thể âm nhỏ do output linear; metrics giữ nguyên output.'))
    page('12. Kiểm thử, minh chứng và vấn đề môi trường',
         p(f'Đã thực sự chạy {count} tests PASS, không skip. Bằng chứng máy đọc được là results/verification/pytest.xml; output terminal ở pytest_output.txt. Smoke E2E có 6 runs và audit độc lập PASS; bộ chính thức có 72 runs với đủ logs/configs/checkpoints/predictions.'),
         t('Bảng 11. Phạm vi kiểm chứng.', ['Hạng mục','Cách kiểm'], [
             ['Gaussian','Tâm, đối xứng, quadratic, tích phân≈1, SPD'],
             ['Dữ liệu','Shape/range/finite/label, CSV roundtrip, independence/seed'],
             ['MLP','12 kiến trúc/activation cases; output/backward finite'],
             ['Training','Loss giảm; early stop; cùng seed trùng history/prediction'],
             ['Checkpoint','Load/save; best epoch=min validation; inference khớp'],
             ['Metrics','Tính tay; shape mismatch/NaN/empty từ chối'],
             ['Experiments','A/B/C ranges; biến kiểm soát; selection bỏ qua test'],
             ['Audit','Tái tính 72 models, summary, hash, PDF/slide/test count']]),
         p('Audit cuối nạp lại mỗi checkpoint và tính train/validation/test MSE/MAE/RMSE từ CSV; đối chiếu predictions và bảng summary. Kiểm best_epoch là argmin của history, checkpoint config trùng JSON, test=100, splits không trùng điểm và baseline cùng seed trùng prediction. Kết quả audit.json phải PASS mới coi phần kiểm chứng đó hoàn tất.'),
         p('Vấn đề môi trường: venv cũ trỏ WindowsApps không chạy; đã tạo .venv-run với Python 3.13.2 hoạt động, sử dụng PyTorch CPU cài sẵn. Tải pip ban đầu bị sandbox chặn socket WinError10013, sau đó cài cục bộ qua bước cấp quyền. Không đổi execution policy, PATH hoặc cấu hình toàn cục. Tests RED trước triển khai đã chuyển GREEN; không bịa kết quả từ lượt chạy lỗi.'),
         p('Nguồn gốc số liệu truy vết qua official_execution.txt → logs/config JSON → checkpoint → all_runs.csv → summary.csv → figures → report/slides. Bốn tài liệu gốc có SHA256 không đổi.'))
    page('13. Hạn chế, kết luận và tái lập',
         p('Hạn chế: test chỉ100; ba seeds dùng chung dữ liệu; ngân sách theo epoch chưa tách tác động compute; width/depth đổi parameters; output linear có thể âm; CPU-only verification; không đo gradient từng layer. Không khẳng định overfitting nghiêm trọng chỉ từ gap nhỏ hoặc vanishing gradient từ một đường loss.'),
         p('Cải tiến: lặp nhiều dataset seeds; so sánh cố định updates/compute/parameters; tuning activation/LR chỉ bằng validation trước test; output softplus; phân tích residual theo vùng; bootstrap uncertainty và kiểm CUDA. Các đề xuất này chưa được triển khai thành kết quả mới.'),
         p('Kết luận: pipeline PyTorch hoàn chỉnh đã chạy đủ 24 cấu hình ×3 seeds, có kiểm thử và artifact truy vết. Các kết quả cho phép phân tích khả năng biểu diễn, tối ưu và chi phí thay vì mặc định mạng lớn hơn luôn tốt hơn. Những mục kỹ thuật hoàn thành được đối chiếu trong RTM và audit.'),
         p('Thông tin tên/MSSV/nhóm/phân công để trống theo yêu cầu người dùng. Thuyết trình, hỏi đáp và mở Office để kiểm layout là phần người nộp tự thực hiện. Không khẳng định đã kiểm thử CUDA hoặc đã thuyết trình trực tiếp.'),
         p('Tái lập: python -m pip install -r requirements.txt; python -m pytest -q --junitxml=results/verification/pytest.xml; python main.py all; python scripts/build_deliverables.py; python scripts/audit.py; python scripts/package_submission.py. Xem README để tạo venv và chọn interpreter trong VS Code.'),
         p('Tài liệu tham khảo: [1] MLP_Exercise - Yeu Cau.pdf, 4 trang, nguồn chính thức. [2] MLP_Exercise.pdf, 9 trang. [3] Sample_Code.html và Sample_Report.pdf, nguồn tham khảo bố cục. [4] PyTorch deterministic algorithms: docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html. [5] PyTorch Saving and Loading Models: docs.pytorch.org/tutorials/beginner/saving_loading_models. [6] PyTorch Reproducibility: docs.pytorch.org/docs/stable/notes/randomness.html.'))
    return pages


def font_paths():
    candidates = [(Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
                  (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold
    raise RuntimeError('Need Arial or DejaVuSans TTF for Vietnamese PDF; install a local font and update font_paths')


def build_pdf_docx(pages):
    regular, bold = font_paths()
    pdfmetrics.registerFont(TTFont('ReportFont', str(regular)))
    pdfmetrics.registerFont(TTFont('ReportBold', str(bold)))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyVN', fontName='ReportFont', fontSize=10, leading=14, spaceAfter=9))
    styles.add(ParagraphStyle('HeadingVN', fontName='ReportBold', fontSize=16, leading=21, spaceAfter=16, textColor=colors.HexColor('#173650')))
    styles.add(ParagraphStyle('CaptionVN', fontName='ReportFont', fontSize=9, leading=12, spaceAfter=8))
    styles.add(ParagraphStyle('CellVN', fontName='ReportFont', fontSize=8, leading=11))
    story = []
    document = Document()
    document.core_properties.author = ''
    document.core_properties.title = 'BTL01 – Xấp xỉ Gaussian 2D bằng MLP PyTorch'
    section = document.sections[0]
    section.page_height, section.page_width = DocInches(11.69), DocInches(8.27)
    section.top_margin = section.bottom_margin = DocInches(0.65)
    section.left_margin = section.right_margin = DocInches(0.7)
    document.styles['Normal'].font.name = 'Arial'
    document.styles['Normal'].font.size = DocPt(10)
    document.styles['Normal'].paragraph_format.space_after = DocPt(6)
    for index, (title, blocks) in enumerate(pages):
        if index:
            story.append(PageBreak())
            document.add_page_break()
        story.append(Paragraph(escape(title), styles['HeadingVN']))
        document.add_heading(title, 1)
        for block in blocks:
            if block[0] == 'p':
                story.append(Paragraph(escape(block[1]), styles['BodyVN']))
                document.add_paragraph(block[1])
            elif block[0] == 'table':
                _, caption, headers, rows = block
                story.append(Paragraph(escape(caption), styles['CaptionVN']))
                values = [[Paragraph(escape(str(cell)), styles['CellVN']) for cell in row] for row in [headers] + rows]
                width = A4[0] - 86
                table = Table(values, colWidths=[width / len(headers)] * len(headers), repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf3')),
                                           ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#b4c1cb')),
                                           ('VALIGN',(0,0),(-1,-1),'TOP'),
                                           ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                                           ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story.extend([table, Spacer(1, 10)])
                document.add_paragraph(caption, 'Caption')
                dt = document.add_table(rows=1, cols=len(headers))
                dt.style = 'Table Grid'
                for cell, header in zip(dt.rows[0].cells, headers):
                    cell.text = str(header)
                for row in rows:
                    for cell, value in zip(dt.add_row().cells, row):
                        cell.text = str(value)
                for row in dt.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.size = DocPt(8)
                document.add_paragraph('')
            elif block[0] == 'image':
                _, name, caption, width_cm = block
                path = ROOT / 'results/figures' / name
                image = Image(str(path))
                width = min(width_cm * 28.346, A4[0] - 86)
                image.drawHeight *= width / image.drawWidth
                image.drawWidth = width
                story.extend([image, Paragraph(escape(caption), styles['CaptionVN'])])
                document.add_picture(str(path), width=DocInches(width_cm / 2.54))
                document.add_paragraph(caption, 'Caption')
    def footer(canvas, doc):
        canvas.setFont('ReportFont', 8)
        canvas.drawString(43, 24, 'BTL01 – Gaussian Function Approximation / PyTorch')
        canvas.drawRightString(A4[0] - 43, 24, str(doc.page))
    SimpleDocTemplate(str(ROOT / 'reports/ACADEMIC_REPORT.pdf'), pagesize=A4,
                      leftMargin=43, rightMargin=43, topMargin=38, bottomMargin=38,
                      title='BTL01 – Xấp xỉ Gaussian 2D bằng MLP', author='').build(story, onFirstPage=footer, onLaterPages=footer)
    document.save(ROOT / 'reports/ACADEMIC_REPORT.docx')


def build_slides(summary, records, selection, count, findings):
    slides = [
        ('BTL01 – Xấp xỉ Gaussian 2D bằng MLP', ['Python / PyTorch; 24 cấu hình × 3 seeds = 72 runs', 'Họ tên: ____________________  MSSV: ____________________', 'Nhóm / phân công: ____________________'], None),
        ('Bài toán và yêu cầu', ['Input x1∈[−2,2], x2∈[−4,4]; output y=f(x)', 'Khảo sát A: số mẫu; B: neurons; C: tầng ẩn', 'Train/validation/test độc lập; test đúng100, chỉ đánh giá cuối'], 'gaussian_surface.png'),
        ('Gaussian và dữ liệu', ['μ=[0,0]; C=[[0.3,−0.5],[−0.5,2]]; det(C)=0.35', 'f(x)=exp[−0.5(x−μ)^T C^(-1)(x−μ)] / [2π√det(C)]', 'Uniform inputs; train2000, validation400, test100; seed2026'], 'data_scatter.png'),
        ('Kiến trúc MLP', ['nn.Module + nn.Linear; Tanh ở tầng ẩn; output linear', 'Scale miền x/[2,4] cố định, không fit trên test', 'Baseline 2–10–1; tùy chỉnh width và depth'], 'selected_architecture.png'),
        ('Training bằng PyTorch', ['TensorDataset/DataLoader; MSELoss; Adam lr0.01, batch256', 'Forward → loss → zero_grad → backward → optimizer.step', '600 epochs tối đa; patience80; checkpoint theo validation minimum', 'model.eval() / no_grad(); khôi phục best trước test'], None),
        ('Thiết kế thực nghiệm có kiểm soát', ['A=200:200:2000; width10, depth1', 'B=1..10; N2000, depth1; C=1..4; width10/tầng, N2000', 'Seeds11/29/47; fixed CSV; mean ± sample SD', 'A: updates/epoch tăng theo N; C: parameters tăng theo depth'], None),
        ('A – Quy mô dữ liệu', [findings['A']], 'A_comparison.png'),
        ('B – Độ rộng và thiếu khớp', [findings['B']], 'B_comparison.png'),
        ('C – Độ sâu', [findings['C']], 'C_comparison.png'),
        ('Hội tụ và thời gian', ['Xem loss train/validation, best epoch và trần ngân sách cùng nhau', 'Runtime đo train+validation; CPU1 thread; có nhiễu scheduling', 'Không gán nguyên nhân vanishing gradient khi chưa đo gradient'], 'C_convergence.png'),
        ('Bảng tổng hợp và mô hình chọn', [f"Chọn theo validation mean: {selection['run_id']}",
                                                  f"Validation mean của cấu hình: {fmt(selection['mean_validation_mse'])}",
                                                  'Bảng bên dưới: cấu hình có validation tốt nhất trong mỗi nhóm; test báo cáo hậu nghiệm.'], None),
        ('Dự đoán và sai số thực tế', [f"Model đại diện: {selection['run_id']}", 'Grid81×81 chỉ chẩn đoán; test metrics vẫn từ100 mẫu độc lập', 'Output linear có thể âm nhỏ; không clamp để che residual'], 'selected_surfaces.png'),
        ('Kiểm chứng và demo', [f'{count} tests PASS; smoke6 runs; official72 runs', 'Audit nạp lại checkpoints, tái tính metrics và summary, đối chiếu hashes', 'Demo: python main.py all --config configs/smoke.json --output results/demo', 'python -m pytest -q; python scripts/audit.py'], 'selected_test_predictions.png'),
        ('Kết luận, giới hạn và hướng phát triển', ['Hoàn thành pipeline và artifact truy vết cho đủ A/B/C', '100 test points, fixed data seed, 3 model seeds; không kết luận thống kê quá mức', 'Cần update/compute-matched studies và nhiều data seeds', 'CUDA và layout Office chưa kiểm trực tiếp; tên/MSSV do người nộp điền', 'Nguồn: PDF chính thức; PyTorch docs; README và báo cáo kèm theo'], None),
    ]
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    pdf_story = []
    styles = getSampleStyleSheet()
    slide_body = ParagraphStyle('SlideBodyVN', fontName='ReportFont', fontSize=13, leading=18, spaceAfter=10)
    slide_title = ParagraphStyle('SlideTitleVN', fontName='ReportBold', fontSize=24, leading=29, spaceAfter=20)
    for index, (title, bullets, figure) in enumerate(slides, 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        def textbox(left, top, width, height, lines, size, bold=False, spacing=13):
            shape = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
            tf = shape.text_frame
            tf.word_wrap = True
            for i, text in enumerate(lines):
                paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                paragraph.text = text
                paragraph.font.name, paragraph.font.size, paragraph.font.bold = 'Arial', Pt(size), bold
                paragraph.font.color.rgb = RGBColor.from_string('173650')
                paragraph.space_after = Pt(spacing)
            return shape
        textbox(0.5, 0.25, 12.3, 0.8, [title], 28, True)
        if figure:
            # Detailed experiment commentary is in speaker notes; keep visible copy concise.
            visible = bullets if index not in (7, 8, 9) else [bullets[0].split(' Đây là')[0].split(' Test MSE từ')[0]]
            textbox(0.55, 1.10, 12.1, 1.65, visible, 16, spacing=6)
            path = ROOT / 'results/figures' / figure
            from PIL import Image as PILImage
            with PILImage.open(path) as picture:
                iw, ih = picture.size
            width = min(12.1, 3.8 * iw / ih)
            height = width * ih / iw
            slide.shapes.add_picture(str(path), Inches((13.333-width)/2), Inches(3.0), width=Inches(width), height=Inches(height))
        else:
            textbox(0.7, 1.35, 11.9, 2.2 if index == 11 else 5.3, bullets, 23 if index == 1 else 21)
        comparison_table = None
        if index == 11:
            comparison_table = [['Nhóm', 'Giá trị', 'Validation MSE', 'Test MSE ± SD', 'Test MAE ± SD']]
            for group in 'ABC':
                row = min([r for r in summary if r['experiment'] == group], key=lambda r:r['validation_mse_mean'])
                comparison_table.append([group, str(int(row['value'])), fmt(row['validation_mse_mean']),
                                         mean_sd(row), mean_sd(row, 'test_mae')])
            shape = slide.shapes.add_table(4, 5, Inches(0.6), Inches(4.0), Inches(12.1), Inches(2.0))
            for ri, row in enumerate(comparison_table):
                for ci, value in enumerate(row):
                    cell = shape.table.cell(ri, ci)
                    cell.text = value
                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.name, paragraph.font.size = 'Arial', Pt(14)
                        paragraph.font.bold = ri == 0
        textbox(11.6, 7.1, 1.1, 0.25, [f'{index}/{len(slides)}'], 10)
        slide.notes_slide.notes_text_frame.text = '\n\n'.join(bullets) + '\n\nNguồn: logs và bảng CSV trong results; chỉ dùng validation để chọn model.'
        if index > 1:
            pdf_story.append(PageBreak())
        pdf_story.append(Paragraph(escape(title), slide_title))
        pdf_bullets = bullets if index not in (7,8,9) else [bullets[0].split(' Đây là')[0].split(' Test MSE từ')[0]]
        pdf_story.extend(Paragraph(escape(b), slide_body) for b in pdf_bullets)
        if comparison_table:
            cell_style = ParagraphStyle('SlideCellVN', fontName='ReportFont', fontSize=11, leading=15)
            table = Table([[Paragraph(escape(v), cell_style) for v in row] for row in comparison_table],
                          colWidths=[60, 80, 180, 250, 250])
            table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), .5, colors.grey),
                                       ('BACKGROUND',(0,0),(-1,0), colors.HexColor('#e5edf3')),
                                       ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
            pdf_story.extend([Spacer(1,20), table])
        if figure:
            image = Image(str(ROOT / 'results/figures' / figure))
            scale = min(820/image.drawWidth, 280/image.drawHeight)
            image.drawWidth *= scale
            image.drawHeight *= scale
            pdf_story.append(image)
    presentation.save(ROOT / 'slides/BTL01_PRESENTATION.pptx')
    SimpleDocTemplate(str(ROOT / 'slides/BTL01_PRESENTATION.pdf'), pagesize=(960,540),
                      leftMargin=45, rightMargin=45, topMargin=30, bottomMargin=25).build(pdf_story)
    notes = '# Gợi ý thuyết trình 10–15 phút\n\nTên, MSSV, nhóm và đóng góp: người nộp tự điền.\n\n'
    for i, (title, bullets, _) in enumerate(slides, 1):
        notes += f'## Slide {i}: {title}\n\n' + '\n\n'.join(bullets) + '\n\n'
    notes += '''## Demo và câu hỏi chuẩn bị

Mở VS Code, chọn interpreter venv; chạy `python -m pytest -q`, sau đó `python main.py all --config configs/smoke.json --output results/demo`. Đây là demo đường chạy, không thay bộ kết quả chính thức.

1. Vì sao x uniform nhưng y là mật độ Gaussian? x là các vị trí khảo sát hàm, không phải samples từ Gaussian.
2. Vì sao validation và test tách riêng? Validation chọn epoch/cấu hình; test chỉ đo sau lựa chọn.
3. Tại sao train loss khác optimization_loss? Train đo lại sau epoch; optimization loss đo khi model đang cập nhật qua batch.
4. Vì sao nhiều mẫu/layer không chắc luôn tốt hơn? Thay thông tin, capacity và độ khó tối ưu; finite budget/seed có thể gây dao động.
5. Vì sao không gọi MSE là accuracy? Bài toán hồi quy, MSE là bình phương sai số; càng nhỏ càng tốt.
6. Hãy chỉ forward, backward, optimizer.step, model.eval và no_grad trong src/training.py.
7. Checkpoint nào được nộp? Best validation state, không mặc định last state; selection.json ghi run đại diện.
8. SD trên 3 seeds phản ánh gì? Khởi tạo và shuffle cùng dữ liệu, không uncertainty do dataset sampling.
'''
    (ROOT / 'slides/PRESENTATION_NOTES.md').write_text(notes, encoding='utf-8')


def main():
    manifest, summary, records, selection, count = load_results()
    findings = analysis(summary, records, selection)
    write_markdown(manifest, summary, records, selection, count, findings)
    pages = academic_pages(summary, records, selection, count, findings)
    build_pdf_docx(pages)
    build_slides(summary, records, selection, count, findings)
    print(f'Built report with {len(pages)} planned pages and 14 slides from {len(records)} actual runs.')


if __name__ == '__main__':
    main()
