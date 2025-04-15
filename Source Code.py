import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cài đặt font chữ hỗ trợ tiếng Việt (DejaVu Sans)
plt.rcParams['font.family'] = 'DejaVu Sans'  # Font hỗ trợ tiếng Việt tốt

# Đọc dữ liệu
file_path = r"TIMES_WorldUniversityRankings_2024.csv"
df = pd.read_csv(file_path) # tạo dataframe df từ file csv
# df = pd.read_csv(file_path, encoding='utf-8') nếu có lỗi về font chữ thì sử dụng dòng này

# Hàm chuyển đổi điểm số từ chuỗi dạng khoảng thành giá trị trung bình
def convert_score_range(score):
    # Kiểm tra xem biến score có phải là chuỗi và có chứa ký tự "–" hay không 
    # để xem score có biểu diễn một khoảng giá trị ví dụ như "7.5–8.5" hay không
    if isinstance(score, str) and "–" in score:
        # Sử dụng biểu thức chính quy (regex) để tìm tất cả các chuỗi con chứa các ký tự số và dấu chấm.
        parts = re.findall(r"[\d\.]+", score)
        # Nếu tìm được đúng 2 số (ví dụ: "7.5" và "8.5")
        if len(parts) == 2:
            # Chuyển các chuỗi số thành float, tính trung bình cộng và trả về giá trị trung bình
            return (float(parts[0]) + float(parts[1])) / 2
    try:
        # Nếu score không phải là chuỗi biểu diễn khoảng giá trị, cố gắng chuyển đổi trực tiếp sang số thực
        return float(score)
    except ValueError:
        # Nếu không thể chuyển đổi được ví dụ như score không phải là số hợp lệ thì trả về None
        return None

# Làm sạch dữ liệu: xác định danh sách các cột chứa điểm số cần xử lý
score_columns = [
    "scores_overall", "scores_teaching", "scores_research", 
    "scores_citations", "scores_industry_income", "scores_international_outlook"
]

# Áp dụng hàm convert_score_range cho từng cột điểm số trong dataframe
for col in score_columns:
    df[col] = df[col].apply(convert_score_range)  # Chuyển đổi từng giá trị score trong cột thành float (hoặc trung bình nếu là khoảng)

# Làm sạch cột số lượng sinh viên: loại bỏ dấu phẩy và chuyển đổi thành float
df["stats_number_students"] = df["stats_number_students"].str.replace(",", "").astype(float)

# Chuyển đổi cột tỉ lệ sinh viên trên giảng viên thành số, bắt lỗi nếu có giá trị không hợp lệ
df["stats_student_staff_ratio"] = pd.to_numeric(df["stats_student_staff_ratio"], errors="coerce")

# Chuyển đổi cột phần trăm sinh viên quốc tế: loại bỏ ký hiệu "%" và chuyển thành số
df["stats_pc_intl_students"] = pd.to_numeric(df["stats_pc_intl_students"].str.replace("%", ""), errors="coerce")

# Lưu bản sao dữ liệu gốc để so sánh sau này
df_before_cleaning = df.copy()

# Loại bỏ các dòng không có điểm tổng thể (scores_overall)
df_cleaned = df.dropna(subset=["scores_overall"])

# Thêm bước loại bỏ outliers bằng IQR cho tất cả các cột điểm số
for col in score_columns:
    # Tính toán các giá trị Q1 (phần tư thứ 1) và Q3 (phần tư thứ 3)
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    # Tính khoảng tứ phân vị (IQR)
    IQR = Q3 - Q1
    # Xác định giới hạn dưới và trên cho outlier
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Giữ lại những dòng mà giá trị trong cột nằm trong khoảng [lower_bound, upper_bound]
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

# Kiểm tra số lượng dòng trước và sau khi làm sạch
print(f"Số lượng trường đại học trước khi làm sạch: {len(df)}")
print(f"Số lượng trường đại học sau khi làm sạch: {len(df_cleaned)}")

# So sánh giá trị bị thiếu (missing) trong dữ liệu
missing_before = df_before_cleaning.isnull().sum()
missing_after = df_cleaned.isnull().sum()
print("\n Giá trị bị thiếu trước khi làm sạch:")
print(missing_before[missing_before > 0])
print("\n Giá trị bị thiếu sau khi làm sạch:")
print(missing_after[missing_after > 0])

# Thống kê mô tả cho các cột điểm số
print("\n Thống kê trước khi làm sạch:")
print(df_before_cleaning[score_columns].describe())
print("\n Thống kê sau khi làm sạch:")
print(df_cleaned[score_columns].describe())

# So sánh phân bố dữ liệu bằng Boxplot cho các cột điểm số
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
# Boxplot trước khi làm sạch
df_before_cleaning[score_columns].boxplot(ax=axes[0])
axes[0].set_title(" Phân bố điểm số của các trường đại học trước khi làm sạch", fontsize=14)
axes[0].set_ylabel("Giá trị điểm số", fontsize=12)
# Boxplot sau khi làm sạch và loại bỏ outliers
df_cleaned[score_columns].boxplot(ax=axes[1])
axes[1].set_title(" Phân bố điểm số của các trường đại học sau khi làm sạch và loại bỏ outliers", fontsize=14)
axes[1].set_ylabel("Giá trị điểm số", fontsize=12)
plt.tight_layout()
plt.show()

# Vẽ Histogram so sánh phân bố điểm số trước và sau khi làm sạch cho từng cột điểm số
fig, axes = plt.subplots(2, len(score_columns), figsize=(15, 6), sharex=True, sharey=True)
for i, col in enumerate(score_columns):
    # Vẽ histogram với kernel density (kde) cho dữ liệu trước khi làm sạch với màu đỏ
    sns.histplot(df_before_cleaning[col], ax=axes[0, i], kde=True, bins=20, color='red')
    axes[0, i].set_title(f"Trước ({col})", fontsize=10)
    # Vẽ histogram với kernel density (kde) cho dữ liệu sau khi làm sạch với màu xanh lam
    sns.histplot(df_cleaned[col], ax=axes[1, i], kde=True, bins=20, color='blue')
    axes[1, i].set_title(f"Sau ({col})", fontsize=10)
# Gán nhãn cho trục y cho từng hàng của histogram
axes[0, 0].set_ylabel("Tần suất (Trước khi làm sạch)", fontsize=12)
axes[1, 0].set_ylabel("Tần suất (Sau khi làm sạch)", fontsize=12)
plt.tight_layout()
plt.show()

# Phân tích tương quan giữa các cột điểm số và trực quan hóa bằng Heatmap
correlation_matrix = df_cleaned[score_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title(" Ma trận tương quan giữa các tiêu chí đánh giá", fontsize=14)
plt.xticks(rotation=30)
plt.show()

# Phát hiện Outlier bằng IQR cho cột "scores_overall" (sau khi loại bỏ ban đầu)
Q1 = df_cleaned['scores_overall'].quantile(0.25)
Q3 = df_cleaned['scores_overall'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Lọc ra các dòng có giá trị "scores_overall" nằm ngoài khoảng [lower_bound, upper_bound]
outliers = df_cleaned[(df_cleaned['scores_overall'] < lower_bound) | (df_cleaned['scores_overall'] > upper_bound)]
print(f"\n Số lượng trường đại học có điểm số bất thường (Outlier): {len(outliers)}")
if not outliers.empty:
    print("Các trường đại học có điểm số bất thường:")
    print(outliers[['name', 'scores_overall', 'location']])

# Trực quan hóa outlier cho "scores_overall" bằng Boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=df_cleaned['scores_overall'])
# plt.xlabel("Điểm tổng thể", fontsize=12)
# plt.title(" Phân bố điểm tổng thể của các trường đại học sau khi loại bỏ outliers", fontsize=14)
# plt.show()

# Phân cụm các trường đại học bằng thuật toán K-Means dựa trên các tiêu chí điểm số
cluster_features = ["scores_overall", "scores_teaching", "scores_research", 
                    "scores_citations", "scores_industry_income", "scores_international_outlook"]
scaler = StandardScaler()  # Chuẩn hóa dữ liệu để các cột có cùng đơn vị và tầm giá trị
df_scaled = scaler.fit_transform(df_cleaned[cluster_features])
# Áp dụng K-Means với 3 cụm, random_state để đảm bảo kết quả lặp lại được và n_init xác định số lần khởi tạo
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cleaned["cluster"] = kmeans.fit_predict(df_scaled)

# Trực quan hóa kết quả phân cụm bằng Scatterplot dựa trên "scores_overall" và "scores_research"
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_cleaned["scores_overall"], y=df_cleaned["scores_research"], 
                hue=df_cleaned["cluster"], palette="coolwarm", size=df_cleaned["stats_number_students"])
plt.xlabel("Điểm tổng thể", fontsize=12)
plt.ylabel("Điểm nghiên cứu", fontsize=12)
plt.title(" Phân cụm trường đại học theo tiêu chí điểm số", fontsize=14)
plt.legend(title="Cụm trường đại học")
plt.show()

# Đảm bảo df_cleaned là một bản sao độc lập trước khi gán lại giá trị của cột cluster
df_cleaned = df_cleaned.copy()
# Gán lại cột cluster (có thể làm lại để chắc chắn giá trị được tính đúng)
df_cleaned.loc[:, "cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(df_scaled)

# Hiển thị số lượng trường đại học trong mỗi cụm
print("\n📊 Số lượng trường đại học trong mỗi cụm:")
print(df_cleaned["cluster"].value_counts())

# Vẽ Scatterplot để phân tích mối quan hệ giữa các yếu tố khác nhau
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# So sánh Điểm tổng thể và Số lượng sinh viên
sns.scatterplot(ax=axes[0, 0], x=df_cleaned['scores_overall'], y=df_cleaned['stats_number_students'], alpha=0.5)
axes[0, 0].set_title("Điểm tổng thể vs Số lượng sinh viên")
# So sánh Điểm nghiên cứu và Điểm trích dẫn
sns.scatterplot(ax=axes[0, 1], x=df_cleaned['scores_research'], y=df_cleaned['scores_citations'], alpha=0.5)
axes[0, 1].set_title("Điểm nghiên cứu vs Điểm trích dẫn")
# So sánh Tỷ lệ sinh viên quốc tế và Điểm tổng thể
sns.scatterplot(ax=axes[1, 0], x=df_cleaned['stats_pc_intl_students'], y=df_cleaned['scores_overall'], alpha=0.5)
axes[1, 0].set_title("Tỷ lệ sinh viên quốc tế vs Điểm tổng thể")
# So sánh Tỷ lệ sinh viên/giảng viên và Điểm giảng dạy
sns.scatterplot(ax=axes[1, 1], x=df_cleaned['stats_student_staff_ratio'], y=df_cleaned['scores_teaching'], alpha=0.5)
axes[1, 1].set_title("Tỷ lệ sinh viên/giảng viên vs Điểm giảng dạy")
plt.tight_layout()
plt.show()
