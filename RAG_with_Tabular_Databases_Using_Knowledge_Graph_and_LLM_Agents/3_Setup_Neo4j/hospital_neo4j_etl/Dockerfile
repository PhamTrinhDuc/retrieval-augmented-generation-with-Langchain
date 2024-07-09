FROM python:3.11-slim
# Sử dụng image python:3.11-slim làm image cơ bản. 
# Đây là một phiên bản nhẹ (slim) của Python 3.11, giúp giảm kích thước của Docker image cuối cùng.
WORKDIR /app
# Thiết lập thư mục làm việc hiện tại trong container thành /app. 
# Tất cả các lệnh tiếp theo trong Dockerfile sẽ được thực thi trong thư mục này.
COPY ./src/ /app
# Sao chép toàn bộ nội dung của thư mục src từ máy chủ (host) vào thư mục /app trong container. 
# Điều này giúp chuyển mã nguồn của ứng dụng vào container.
COPY ./pyproject.toml /code/pyproject.toml
# Sao chép tệp pythonproject.toml từ máy chủ (host) vào thư mục /code trong container. 
# Tệp này có thể chứa thông tin cấu hình hoặc phụ thuộc của dự án.
RUN pip install /code/.
# Thực thi lệnh pip install để cài đặt các gói Python dựa trên thông tin trong thư mục /code. 
# Lệnh này thường được sử dụng để cài đặt các phụ thuộc được liệt kê trong tệp cấu hình như setup.py hoặc pyproject.toml.
CMD ["sh", "entrypoint.sh"]
# Thiết lập lệnh mặc định sẽ được chạy khi container khởi động. 
# Trong trường hợp này, container sẽ chạy tệp shell script entrypoint.sh.