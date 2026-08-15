# Lab 16 — Kết quả Benchmark LightGBM (CPU flow)

## Hạ tầng đã triển khai
- **Region:** us-east-1
- **Compute Node:** `t3.medium` (2 vCPU / 4 GB RAM) — instance `i-0f20ad9cc85b4eb80`, private IP `10.0.10.96`
- **Bastion Host:** `t3.micro` — public IP `32.199.156.73`
- Dataset: [`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 giao dịch, 31 cột, tỷ lệ gian lận 0.173%)

## Bảng kết quả (Bước 4.4)

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.203 s |
| Thời gian training | 2.003 s |
| Best iteration | 1 |
| AUC-ROC | 0.95165 |
| Accuracy | 0.99895 |
| F1-Score | 0.72727 |
| Precision | 0.65574 |
| Recall | 0.81633 |
| Inference latency (1 row) | 1.606 ms |
| Inference throughput (1000 rows) | 520,060.8 rows/sec (1.923 ms cho 1000 dòng) |

Chi tiết đầy đủ (kèm thông tin môi trường): xem [`benchmark_result.json`](benchmark_result.json).

## Kiểm tra tài nguyên (Phần 5)
- **RAM:** `free -h` trên Compute Node → tổng 3.7 GiB, dùng 227 MiB lúc idle sau training, còn trống 1.7 GiB, buff/cache 1.8 GiB.
- **CPU:** CloudWatch `CPUUtilization` cho `i-0f20ad9cc85b4eb80` — mức nền ~0.1%, tăng vọt lên trung bình 8.4% (đỉnh 37.8%) trong cửa sổ 5 phút chứa lúc script `benchmark.py` chạy training.
- **Network:** `ip -s link` trên interface `ens5` → RX ~286 MB / 196,635 gói (chủ yếu do tải dataset 150 MB qua Kaggle CLI), TX ~0.96 MB / 11,525 gói.
- **Chi phí:** AWS Cost Explorer (`aws ce get-cost-and-usage`) cho ngày chạy lab trả về $0 — dữ liệu billing của AWS thường trễ 24h so với usage thực tế, nên số liệu chưa kịp cập nhật. Ước tính theo bảng giá công bố (Phần 5.2 của README): **~$0.10/giờ** (EC2 t3.medium + t3.micro + NAT Gateway + ALB) cho toàn bộ thời gian hạ tầng này chạy.

## Báo cáo ngắn (nhận xét)

Với `t3.medium` (2 vCPU, không GPU), toàn bộ vòng load dữ liệu + training trên 227,845 dòng chỉ mất khoảng 4.2 giây tổng cộng — LightGBM tận dụng tốt thuật toán histogram-based gradient boosting nên không cần phần cứng mạnh cho dữ liệu dạng bảng cỡ này. Mô hình dừng sớm ở vòng lặp thứ 1 (`best_iteration=1`, patience 30 vòng) cho thấy tập đặc trưng PCA (V1–V28) đã rất tách biệt giữa hai lớp, nên các vòng boosting sau gần như không cải thiện thêm AUC trên validation set. AUC-ROC đạt 0.952 là một baseline khá tốt nhưng vẫn thấp hơn mức ~0.97–0.98 mà các mô hình được tune kỹ trên cùng bộ dữ liệu này thường đạt được — có thể cải thiện bằng cách tăng `scale_pos_weight`/`is_unbalance` để xử lý mất cân bằng lớp (0.17% gian lận), tăng `stopping_rounds`, hoặc tune `num_leaves`/`learning_rate`. Precision 0.656 và Recall 0.816 phản ánh đúng đặc trưng bài toán fraud detection: ưu tiên bắt được nhiều giao dịch gian lận (recall cao) dù phải chấp nhận một số cảnh báo nhầm. Điểm ấn tượng nhất là tốc độ inference: độ trễ chỉ 1.6 ms cho một dòng và throughput hơn **520,000 dòng/giây** khi dự đoán hàng loạt — chỉ với 2 vCPU. Điều này minh chứng rõ ràng cho lý do LightGBM (và các mô hình gradient boosting nói chung) vẫn là lựa chọn phổ biến để phục vụ real-time fraud-detection API, vì không cần GPU mà vẫn đạt thông lượng cực cao với chi phí hạ tầng chỉ ~$0.04/giờ cho compute node.
