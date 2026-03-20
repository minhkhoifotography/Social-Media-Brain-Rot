  import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. KẾT NỐI DỮ LIỆU
SHEET_ID = 'THAY_ID_CỦA_BẠN_VÀO_ĐÂY'
SHEET_NAME = 'Data' # Tên tab bạn đã đặt trong Apps Script
url = f'https://docs.google.com/spreadsheets/d/1v1UX9TMJ1sdiZVULXLs6HrM6QSmENtyrXHRA_MAiXnc/gviz/tq?tqx=out:csv&sheet=Data'

try:
    df = pd.read_csv(url)
    print("✅ Kết nối dữ liệu thành công!")
    print(df.head()) # Xem 5 dòng đầu tiên
except Exception as e:
    print(f"❌ Lỗi kết nối: {e}")

# 2. TIỀN XỬ LÝ (DATA CLEANING) - Bước quan trọng của dân chuyên Tin
# Loại bỏ các mẫu thử lỗi (Outliers)
# - Accuracy phải > 50% (loại bỏ người nhấn bừa)
# - AvgRT phải nằm trong khoảng [200ms, 2000ms] (loại bỏ bot hoặc người bị xao nhãng)
df_clean = df[
    (df['Accuracy'] >= 50) & 
    (df['AvgReactionTime'] >= 200) & 
    (df['AvgReactionTime'] <= 2000)
].copy()

# Tạo chỉ số Focus Score (Càng cao càng tập trung)
df_clean['FocusScore'] = (df_clean['Accuracy'] / df_clean['AvgReactionTime']) * 1000

print(f"\n📊 Đã làm sạch dữ liệu. Số mẫu hợp lệ: {len(df_clean)}/{len(df)}")

# 3. TRỰC QUAN HÓA SƠ BỘ
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x='UsageHours', y='AvgReactionTime', scatter_kws={'alpha':0.5})
plt.title('Mối tương quan giữa Thời gian dùng MXH và Tốc độ phản xạ')
plt.xlabel('Số giờ sử dụng (giờ/ngày)')
plt.ylabel('Thời gian phản xạ trung bình (ms)')
plt.grid(True)
plt.show()
