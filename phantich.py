import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def convert_score_range(score):
    """Chuyển đổi điểm có dạng khoảng thành giá trị trung bình."""
    if isinstance(score, str) and "–" in score:
        parts = re.findall(r"[\d\.]+", score)
        if len(parts) == 2:
            return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(score)
    except ValueError:
        return None

def process_university_rankings(file_path):
    """Xử lý dữ liệu bảng xếp hạng đại học từ file CSV."""
    df = pd.read_csv(file_path)
    
    # Chuyển đổi dữ liệu
    df["scores_overall"] = df["scores_overall"].apply(convert_score_range)
    df["stats_number_students"] = df["stats_number_students"].str.replace(",", "").astype(float)
    df["stats_student_staff_ratio"] = pd.to_numeric(df["stats_student_staff_ratio"], errors="coerce")
    df["stats_pc_intl_students"] = pd.to_numeric(df["stats_pc_intl_students"].str.replace("%", ""), errors="coerce")

    # Loại bỏ những dòng không có scores_overall
    df_cleaned = df.dropna(subset=["scores_overall"])
    
    # Truy vấn dữ liệu
    top_10_universities = df_cleaned.nlargest(10, "scores_overall")[["name", "scores_overall", "location"]]
    top_countries = df_cleaned["location"].value_counts().head(10)
    top_countries_scores = df_cleaned.groupby("location")["scores_overall"].mean().sort_values(ascending=False).head(10)
    top_international = df_cleaned.nlargest(10, "stats_pc_intl_students")[["name", "stats_pc_intl_students", "location"]]
    top_student_staff_ratio = df_cleaned.nsmallest(10, "stats_student_staff_ratio")[["name", "stats_student_staff_ratio", "location"]]
    top_research = df_cleaned.nlargest(10, "scores_research")[["name", "scores_research", "location"]]
    top_citations = df_cleaned.nlargest(10, "scores_citations")[["name", "scores_citations", "location"]]
    top_industry_income = df_cleaned.nlargest(10, "scores_industry_income")[["name", "scores_industry_income", "location"]]
    top_international_outlook = df_cleaned.nlargest(10, "scores_international_outlook")[["name", "scores_international_outlook", "location"]]
    top_students = df_cleaned.nlargest(10, "stats_number_students")[["name", "stats_number_students", "location"]]
    
    return {
        "top_10_universities": top_10_universities,
        "top_countries": top_countries,
        "top_countries_scores": top_countries_scores,
        "top_international": top_international,
        "top_student_staff_ratio": top_student_staff_ratio,
        "top_research": top_research,
        "top_citations": top_citations,
        "top_industry_income": top_industry_income,
        "top_international_outlook": top_international_outlook,
        "top_students": top_students,
        "df_cleaned": df_cleaned
    }

def plot_data(data):
    """Vẽ các biểu đồ từ dữ liệu."""
    df_cleaned = data["df_cleaned"]
    
    # 1) Top 10 quốc gia có nhiều trường đại học nhất
    top_countries = data["top_countries"]
    if not top_countries.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_countries.values, y=top_countries.index, palette="viridis")
        plt.xlabel("Số lượng trường đại học")
        plt.ylabel("Quốc gia")
        plt.title("Top 10 quốc gia có nhiều trường đại học nhất")
        plt.show()
    
    # 2) Mối quan hệ giữa điểm tổng thể và số lượng sinh viên (scatter plot)
    if not df_cleaned.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df_cleaned['scores_overall'], y=df_cleaned['stats_number_students'], alpha=0.5)
        plt.xlabel("Điểm tổng thể")
        plt.ylabel("Số lượng sinh viên")
        plt.title("Mối quan hệ giữa điểm số và số lượng sinh viên")
        plt.show()
    
    # 3) Top 10 quốc gia có điểm trung bình tổng thể cao nhất
    top_countries_scores = data["top_countries_scores"]
    if not top_countries_scores.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top_countries_scores.values, y=top_countries_scores.index, palette="coolwarm")
        plt.xlabel("Điểm trung bình tổng thể")
        plt.ylabel("Quốc gia")
        plt.title("Top 10 quốc gia có điểm trung bình tổng thể cao nhất")
        plt.show()

    # 4) Top 10 trường đại học có điểm tổng cao nhất
    top_10_universities = data["top_10_universities"]
    if not top_10_universities.empty:
        plt.figure(figsize=(8, 6))
        # Sắp xếp để thanh xếp từ thấp đến cao
        top_10_universities_sorted = top_10_universities.sort_values("scores_overall")
        sns.barplot(data=top_10_universities_sorted, x="scores_overall", y="name", palette="magma")
        plt.xlabel("Điểm tổng thể")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có điểm tổng cao nhất")
        plt.show()

    # 5) Top 10 trường đại học có tỷ lệ sinh viên quốc tế cao nhất
    top_international = data["top_international"]
    if not top_international.empty:
        plt.figure(figsize=(8, 6))
        top_international_sorted = top_international.sort_values("stats_pc_intl_students")
        sns.barplot(data=top_international_sorted, x="stats_pc_intl_students", y="name", palette="Blues_r")
        plt.xlabel("Tỷ lệ % sinh viên quốc tế")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có tỷ lệ sinh viên quốc tế cao nhất")
        plt.show()

    # 6) Top 10 trường đại học có tỷ lệ sinh viên trên giảng viên thấp nhất (tốt nhất)
    top_student_staff_ratio = data["top_student_staff_ratio"]
    if not top_student_staff_ratio.empty:
        plt.figure(figsize=(8, 6))
        # Ở đây sắp xếp giảm dần (ascending=False) để trường có tỷ lệ cao nằm đầu
        # Nếu muốn trường có tỷ lệ thấp nằm trên cùng, có thể bỏ ascending=False
        top_student_staff_ratio_sorted = top_student_staff_ratio.sort_values("stats_student_staff_ratio", ascending=False)
        sns.barplot(data=top_student_staff_ratio_sorted, x="stats_student_staff_ratio", y="name", palette="GnBu_d")
        plt.xlabel("Tỷ lệ sinh viên/giảng viên (càng thấp càng tốt)")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có tỷ lệ SV/GV thấp nhất")
        plt.show()

    # 7) Top 10 trường đại học có điểm nghiên cứu cao nhất
    top_research = data["top_research"]
    if not top_research.empty:
        plt.figure(figsize=(8, 6))
        top_research_sorted = top_research.sort_values("scores_research")
        sns.barplot(data=top_research_sorted, x="scores_research", y="name", palette="YlOrBr")
        plt.xlabel("Điểm nghiên cứu")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có điểm nghiên cứu cao nhất")
        plt.show()

    # 8) Top 10 trường đại học có điểm trích dẫn cao nhất
    top_citations = data["top_citations"]
    if not top_citations.empty:
        plt.figure(figsize=(8, 6))
        top_citations_sorted = top_citations.sort_values("scores_citations")
        sns.barplot(data=top_citations_sorted, x="scores_citations", y="name", palette="Reds")
        plt.xlabel("Điểm trích dẫn")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có điểm trích dẫn cao nhất")
        plt.show()

    # 9) Top 10 trường đại học có điểm thu nhập từ ngành công nghiệp cao nhất
    top_industry_income = data["top_industry_income"]
    if not top_industry_income.empty:
        plt.figure(figsize=(8, 6))
        top_industry_income_sorted = top_industry_income.sort_values("scores_industry_income")
        sns.barplot(data=top_industry_income_sorted, x="scores_industry_income", y="name", palette="OrRd")
        plt.xlabel("Điểm thu nhập từ ngành công nghiệp")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có thu nhập từ ngành công nghiệp cao nhất")
        plt.show()

    # 10) Top 10 trường đại học có triển vọng quốc tế cao nhất
    top_international_outlook = data["top_international_outlook"]
    if not top_international_outlook.empty:
        plt.figure(figsize=(8, 6))
        top_international_outlook_sorted = top_international_outlook.sort_values("scores_international_outlook")
        sns.barplot(data=top_international_outlook_sorted, x="scores_international_outlook", y="name", palette="PuBuGn")
        plt.xlabel("Điểm triển vọng quốc tế")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có triển vọng quốc tế cao nhất")
        plt.show()

    # 11) Top 10 trường đại học có số lượng sinh viên lớn nhất
    top_students = data["top_students"]
    if not top_students.empty:
        plt.figure(figsize=(8, 6))
        top_students_sorted = top_students.sort_values("stats_number_students")
        sns.barplot(data=top_students_sorted, x="stats_number_students", y="name", palette="CMRmap")
        plt.xlabel("Số lượng sinh viên")
        plt.ylabel("Trường đại học")
        plt.title("Top 10 trường đại học có số lượng sinh viên lớn nhất")
        plt.show()

def print_university_list(df_cleaned):
    """Hiển thị danh sách các trường trong biểu đồ số 2 trên terminal."""
    selected_columns = df_cleaned[["name", "scores_overall", "stats_number_students"]]
    print("\nDanh sách trường đại học có trong biểu đồ số 2 (Điểm số vs Số lượng sinh viên):\n")
    print(selected_columns.to_string(index=False))

# Đường dẫn tới file CSV
file_path = "TIMES_WorldUniversityRankings_2024.csv"

# Gọi hàm xử lý dữ liệu
results = process_university_rankings(file_path)

print("Top 10 trường đại học có điểm tổng cao nhất:")
print(results["top_10_universities"])
print("\nTop 10 quốc gia có nhiều trường trong bảng xếp hạng:")
print(results["top_countries"])
print("\nTop 10 quốc gia có điểm tổng thể trung bình cao nhất:")
print(results["top_countries_scores"])
print("\nTop 10 trường đại học có tỷ lệ sinh viên quốc tế cao nhất:")
print(results["top_international"])
print("\nTop 10 trường đại học có tỷ lệ sinh viên trên giảng viên thấp nhất (tốt nhất):")
print(results["top_student_staff_ratio"])
print("\nTop 10 trường đại học có điểm nghiên cứu cao nhất:")
print(results["top_research"])
print("\nTop 10 trường đại học có điểm trích dẫn cao nhất:")
print(results["top_citations"])
print("\nTop 10 trường đại học có điểm thu nhập từ ngành công nghiệp cao nhất:")
print(results["top_industry_income"])
print("\nTop 10 trường đại học có điểm triển vọng quốc tế cao nhất:")
print(results["top_international_outlook"])
print("\nTop 10 trường đại học có số lượng sinh viên lớn nhất:")
print(results["top_students"])

print_university_list(results["df_cleaned"])

# Vẽ tất cả các biểu đồ
plot_data(results)
