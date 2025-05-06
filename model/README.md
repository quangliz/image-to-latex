### Encoder: sử dụng kiến trúc của Resnet-18 với chút thay đổi
- **Backbone**: Mô hình sử dụng backbone là Resnet-18, dừng lại ở layer3(28x28 output) giúp giữ lại cấu trúc không gian tốt cho bước mã hóa vị trí 2D mà vẫn mang đặc trưng đủ mạnh. Resnet-18 biến đổi ảnh đầu vào (B, C, H, W) thành feature map (B, 256, H/32, W/32), khi mà số chiều không gian giảm nhưng chiều sâu các kênh tăng, giúp học được nhiều đặc trưng hơn.
- **Bottleneck**: Sau khi có feature map với 256 kênh, ta tối ưu số kênh bằng cách thêm 1 lớp bottleneck tích chập, giảm xuống còn 128 kênh, giúp giảm số tham số và độ phức tạp tính toán trong khi vẫn giữ được các thông tin không gian.
- **2D positional encoding**: Ta cần mã hóa vị trí không gian 2 chiều bằng cách kết hợp mã hóa vị trí không gian cho chiều ngang và chiều dọc. Công thức được nhắc đến trong bài báo ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762):
	- $PE_(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$ 
	- $PE_(pos, 2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})$ 
- **Feature Flattening**: làm phẳng feature map 2D (B, $d_{model}$, H, W) thành chuỗi (H$\times$W, B, $d_{model}$) để có thể được xử lý bởi Transformer
### Decoder: sử dụng mã hóa Transformer tiêu chuẩn
- **Token Embedding**: Khi bắt đầu, chuỗi đầu ra (các token LaTeX) được biểu diễn dưới dạng chỉ số nguyên. Để làm việc trong không gian liên tục, các token này được ánh xạ thành vector thông qua một **bảng embedding học được**, kích thước `(vocab_size, d_model)`. Embedding giúp mô hình hiểu được các token không chỉ là những con số rời rạc mà còn chứa thông tin ngữ nghĩa.
- **1D Positional Encoding**: Transformer không có khả năng tự nhận biết thứ tự chuỗi, do đó cần thêm thông tin vị trí. Các vector nhúng sau bước 1 được cộng với mã hóa vị trí (sin-cos) được tính cố định như ở phần 2D positional encoding. Nhờ vậy, mỗi vị trí token có thể được phân biệt, và mô hình hiểu được cấu trúc tuần tự của chuỗi.
- **Transformer Decoder Block**: Sau khi vector đã mang đủ thông tin token và vị trí, chúng đi qua nhiều tầng decoder giống nhau. Mỗi tầng gồm ba phần chính:
	- a. **Masked Self-Attention**
		- Token tại vị trí `t` chỉ được phép nhìn các token trước nó.
		- Giúp mô hình sinh từng ký tự LaTeX tuần tự, không “nhìn trước đáp án”
	- b. **Cross-Attention với Encoder Output**
		- Đầu ra từ encoder (ảnh) được sử dụng làm khóa (key) và giá trị (value).
		- Cho phép mô hình **liên kết token đầu ra với đặc trưng ảnh**, ví dụ: vị trí dấu cộng, ký hiệu phân số...
			- $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$
	- c. **Feed-Forward Network (FFN)**
		- Gồm 2 lớp tuyến tính với ReLU ở giữa:
			- $FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$
		- Tăng khả năng biểu diễn và tính phi tuyến của mô hình.
- **LayerNorm + Residual**: 
	- Mỗi khối attention hoặc FFN đều có:
	    - **Layer Normalization** để ổn định gradient.
	    - **Residual Connection** giúp truyền thông tin và gradient dễ dàng:
			- $x \leftarrow x + \text{Sublayer}(x)$
		- Kết nối tắt giúp mô hình sâu hơn mà vẫn dễ huấn luyện, tránh mất mát thông tin.
- **Final Linear Projection**
	- Sau các tầng decoder, mỗi token được biểu diễn bởi vector kích thước `d_model`.
	- Lớp tuyến tính cuối sẽ chiếu vector này về kích thước của từ vựng, tạo ra **logits**:
	- `Target Tokens → Embedding → 1D Positional Encoding → Transformer Decoder → Linear → Output Logits`
	-  Logits sau đó được dùng để dự đoán token tiếp theo thông qua softmax
#### Luồng dữ liệu tổng quát
```
Target Tokens
→ Token Embedding
→ 1D Positional Encoding
→ [Decoder Blocks]
   ↳ Masked Self-Attention
   ↳ Cross-Attention (với Encoder Output)
   ↳ Feed-Forward Network
   ↳ + Residual + LayerNorm
→ Final Linear Projection
→ Output Logits (Softmax)
```
### Phương pháp đánh giá
#### Character Error Rate:
- Là khoảng cách Levenshtein giữa chuỗi dự đoán và chuỗi mục tiêu, được chuẩn hóa theo độ dài của chuỗi mục tiêu
- Mô hình hoạt động càng tốt với chỉ số càng thấp
- **Ưu điểm**: Cung cấp phép đo chính xác ở cấp độ từng ký tự, cho biết mức độ sai lệch nhỏ
#### Exact Math:
- Tỷ lệ phần trăm các công thức được dự đoán chính xác hoàn toàn (tất cả các token đều khớp với nhãn gốc)
- Giá trị càng cao càng tốt, cho thấy mô hình tái hiện chính xác toàn bộ công thức
- **Ưu điểm**: Là thước đo nghiêm ngặt, yêu cầu dự đoán chính xác toàn bộ biểu thức
#### BLEU:
- Đo mức độ trùng khớp giữa các cụm từ (n-gram) trong chuỗi dự đoán và chuỗi tham chiếu
- Giá trị càng cao cho thấy độ tương đồng ngữ nghĩa càng tốt
- **Ưu điểm**: Nhạy với mức độ đúng một phần, ít bị ảnh hưởng bởi những sai khác nhỏ
#### Edit distance:
- Là khoảng cách Levenshtein thô – số lần chèn, xóa hoặc thay thế cần thiết để biến chuỗi dự đoán thành chuỗi đúng
- Giá trị càng thấp càng tốt
- **Ưu điểm**: Cung cấp một thước đo tuyệt đối về độ khác biệt giữa hai chuỗi.
### Tối ưu hóa Suy luận:
Nhiều kỹ thuật được sử dụng để cải thiện tốc độ và hiệu quả của quá trình suy luận:
- **Early Stopping**: Quá trình sinh chuỗi kết thúc khi tất cả các chuỗi trong batch đã sinh token kết thúc (EOS)
- **Post-Processing**: Gán padding cho tất cả các token xuất hiện sau EOS nhằm đảm bảo định dạng đầu ra nhất quán