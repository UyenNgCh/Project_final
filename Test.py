import joblib
import pandas as pd

# Load dữ liệu
overview_companies = pd.read_excel('Overview_Companies.xlsx')
overview_reviews = pd.read_excel('Overview_Reviews.xlsx')
reviews = pd.read_excel('Reviews.xlsx')
data = reviews.merge(overview_reviews, on='id', how='left')
data = data.merge(overview_companies, on='id', how='left')

# Kiểm tra file HDBSCAN
try:
    hdb = joblib.load('hdbscan_fasttext.pkl')
    print("File hdbscan_fasttext.pkl tồn tại.")
    print("Có labels_:", hasattr(hdb, 'labels_'))
    if hasattr(hdb, 'labels_'):
        print("Số lượng nhãn:", len(hdb.labels_))
        print("Số hàng trong data:", len(data))
        print("Nhãn có khớp với số hàng dữ liệu:", len(hdb.labels_) == len(data))
except FileNotFoundError:
    print("Không tìm thấy file hdbscan_fasttext.pkl")
    