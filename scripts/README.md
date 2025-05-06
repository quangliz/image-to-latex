### Tổng quan: thư mục bao gồm:
- **Tiền xử lý dữ liệu**: tải, xử lý và chuẩn bị dữ liệu
- **Training**: train mô hình
- **Utils**: chứa các hàm hỗ trợ và các lớp sử dụng trong dự án
- **Callbacks**: custom callbacks
### Chi tiết:
#### `prepare_data.py`: 
- Tải dữ liệu thô và giải nén 
- Xử lý dữ liệu thô
- Làm sạch dữ liệu công thức LaTeX
- Xây dựng từ vựng cho token hóa
#### `data.py`:
- **BaseDataset**: triển khai của Pytorch Dataset để load ảnh và công thức
- **Tokenizer**: Xử lý chuyển đổi giữa token LaTeX và chỉ số(index)
- **Im2Latex**: Pytorch Lightning Datamodule quản lý việc load data và tiền xử lý
#### `train.py`:
- Thiết lập data module và model
- Gọi configs(checkpoints, early stopping, metrics)
- Khởi tạo trainer
- Thực thi quá trình train và test
#### `utils.py`:
- **TqdmUpto**: tạo thanh quá trình giúp dễ dàng quan sát
- **Image Processing**: hàm load và cắt ảnh
- `download_url()`: Tải dữ liệu từ nguồn
- `extract_tar_file()`: xuất dữ liệu nén
- `crop()`: cắt vùng thừa của ảnh thô
- `get_all_formulas()`: load công thức từ file
- `get_split()`: lấy dữ liệu được chia cụ thể(train/test/val)
#### `callbacks.py`:
- **MetricsCallback**: log các chỉ số cần thiết sau mỗi epoch
### Chi tiết quá trình xử lý dữ liệu
#### Xử lý ảnh:
- Ảnh được cắt và loại bỏ vùng viền trắng thừa
- Thêm padding xung quanh ảnh công thức
#### Xử lý công thức:
- Làm sạch công thức để đơn giản hóa 
- Thêm các token đặc biệt(SOS, EOS, PAD, UNK)
- Token hóa biến đổi công thức thành các chuỗi index