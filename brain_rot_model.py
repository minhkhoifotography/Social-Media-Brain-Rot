import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. KẾT NỐI VÀ TẢI DỮ LIỆU (Thay ID của bạn vào đây)
SHEET_ID = 'THAY_ID_CỦA_BẠN_VÀO_ĐÂY'
SHEET_NAME = 'Data'
url = f'https://docs.google.com/spreadsheets/d/1v1UX9TMJ1sdiZVULXLs6HrM6QSmENtyrXHRA_MAiXnc/gviz/tq?tqx=out:csv&sheet=Data'

try:
    df = pd.read_csv(url)
    print("✅ Kết nối thành công. Đang xử lý dữ liệu...")
except Exception as e:
    print(f"❌ Lỗi: {e}")
    # Tạo dữ liệu giả lập để bạn chạy thử nếu chưa có đủ 50 mẫu
    data = {
        'UsageHours': np.random.uniform(0, 8, 100),
        'AvgReactionTime': np.random.uniform(400, 1200, 100),
        'Accuracy': np.random.uniform(60, 100, 100)
    }
    df = pd.DataFrame(data)

# 2. TIỀN XỬ LÝ DỮ LIỆU (Feature Engineering)
# Tính Focus Score: Chỉ số đo lường sự tập trung thực tế
df['FocusScore'] = (df['Accuracy'] / df['AvgReactionTime']) * 1000

# 3. MÔ HÌNH MACHINE LEARNING (K-Means Clustering)
# Chọn 2 đặc trưng chính để máy học: Thời gian dùng và Điểm tập trung
X = df[['UsageHours', 'FocusScore']]

# Chuẩn hóa dữ liệu để thuật toán chạy chính xác hơn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Khởi tạo K-Means với 3 cụm (Cần thiết, Nguy cơ, Brain Rot)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Định nghĩa tên cho các cụm dựa trên đặc điểm dữ liệu
# (Lưu ý: Thứ tự cluster 0, 1, 2 có thể thay đổi tùy dữ liệu, bạn cần kiểm tra trung bình cụm)
cluster_map = {0: "Tập trung tốt", 1: "Nguy cơ nhẹ", 2: "Brain Rot nặng"}
df['Status'] = df['Cluster'].map(cluster_map)

# 4. TRỰC QUAN HÓA KẾT QUẢ (Data Visualization)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df, x='UsageHours', y='FocusScore', hue='Status', 
                palette='viridis', s=100, alpha=0.8)

plt.title('PHÂN CỤM MỨC ĐỘ BRAIN ROT DỰA TRÊN DỮ LIỆU THỰC TẾ', fontsize=14)
plt.xlabel('Thời gian sử dụng mạng xã hội (Giờ/Ngày)', fontsize=12)
plt.ylabel('Chỉ số tập trung (Focus Score)', fontsize=12)
plt.legend(title='Trạng thái não bộ')
plt.grid(True, linestyle='--', alpha=0.6)

# Lưu biểu đồ để đưa vào báo cáo
plt.savefig('brain_rot_analysis.png')
plt.show()

print("\n--- THỐNG KÊ CHI TIẾT THEO NHÓM ---")
print(df.groupby('Status')[['UsageHours', 'FocusScore']].mean())
