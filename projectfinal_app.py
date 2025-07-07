import streamlit as st
import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize
import joblib
import gensim.downloader as api
import hdbscan
from collections import Counter
from wordcloud import WordCloud

# Load dữ liệu
overview_companies = pd.read_excel('Overview_Companies.xlsx')
overview_reviews = pd.read_excel('Overview_Reviews.xlsx')
reviews = pd.read_excel('Reviews.xlsx')
data = reviews.merge(overview_reviews, on='id', how='left')
data = data.merge(overview_companies, on='id', how='left')

# Load dictionaries
def load_dict(filepath, sep='\t'):
    dct = {}
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line or sep not in line:
                continue
            key, value = line.split(sep, 1)
            dct[key.strip().lower()] = value.strip().lower()
    return dct

def load_list(filepath):
    with open(filepath, 'r', encoding='utf8') as f:
        return [line.strip() for line in f if line.strip()]

emoji_dict = load_dict('emojicon.txt')
teen_dict = load_dict('teencode.txt')
wrong_lst = load_list('wrong-word.txt')
stopwords_lst = load_list('vietnamese-stopwords.txt')
eng_vn_dict = load_dict('english-vnmese.txt')

custom_stopwords = ['công', 'ty', 'công ty', 'company', 'cty', 'công_ty', 'chúng tôi', 'chúng_tôi', 'có thể', 'có_thể', 'cái đó', 'cái_đó']
stopwords_lst += custom_stopwords

# Hàm xử lý văn bản
def normalize_negation(tokens, negation_words=['không', 'chưa', 'chẳng', 'chả', 'ít']):
    new_tokens = []
    skip = False
    for i, token in enumerate(tokens):
        if skip:
            skip = False
            continue
        if token in negation_words and i+1 < len(tokens):
            neg_token = token + '_' + tokens[i+1]
            new_tokens.append(neg_token)
            skip = True
        else:
            new_tokens.append(token)
    return new_tokens

def clean_text_for_lda(text, stopwords, emoji_dict=None, teen_dict=None, eng_vn_dict=None):
    if pd.isnull(text) or str(text).strip() == "":
        return []
    text = str(text).lower()
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    if emoji_dict:
        text = ' '.join(emoji_dict.get(word, word) for word in text.split())
    if teen_dict:
        text = ' '.join(teen_dict.get(word, word) for word in text.split())
    if eng_vn_dict:
        text = ' '.join(eng_vn_dict.get(word, word) for word in text.split())
    tokens = word_tokenize(text, format='text').split()
    tokens = normalize_negation(tokens)
    if eng_vn_dict:
        tokens = [eng_vn_dict.get(tok, tok) for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stopwords and len(tok) > 2]
    return tokens

# Tạo cột lda_tokens nếu chưa có
if 'lda_tokens' not in data.columns:
    data['all_text'] = data['What I liked'].fillna('') + ' ' + data['Suggestions for improvement'].fillna('')
    data['lda_tokens'] = data['all_text'].apply(
        lambda x: clean_text_for_lda(x, stopwords_lst, emoji_dict, teen_dict, eng_vn_dict)
    )

# Tạo cột cluster_hdbscan nếu chưa có
if 'cluster_hdbscan' not in data.columns:
    try:
        hdb = joblib.load('hdbscan_fasttext.pkl')
        if hasattr(hdb, 'labels_') and len(hdb.labels_) == len(data):
            data['cluster_hdbscan'] = hdb.labels_
        else:
            st.error("Mô hình HDBSCAN không có nhãn hợp lệ hoặc không khớp với dữ liệu.")
            st.stop()
    except FileNotFoundError:
        st.error("Không tìm thấy file hdbscan_fasttext.pkl")
        st.stop()

# Load models với caching
@st.cache_resource
def load_fasttext():
    return api.load('fasttext-wiki-news-subwords-300')

fasttext_model = load_fasttext()

# Load models
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer_balanced.pkl')
    svm_model = joblib.load('svm_model_balanced.pkl')
    hdb_model = joblib.load('hdbscan_fasttext.pkl')
except FileNotFoundError as e:
    st.error(f"Không tìm thấy file mô hình: {e}")
    st.stop()

# Hàm dự đoán sentiment
def predict_sentiment(comment):
    tokens = clean_text_for_lda(comment, stopwords_lst, emoji_dict, teen_dict, eng_vn_dict)
    text_str = ' '.join(tokens)
    vector = tfidf_vectorizer.transform([text_str])
    prediction = svm_model.predict(vector)[0]
    return prediction

# Hàm dự đoán cluster
def sentence_vector(tokens, model):
    vecs = [model[word] for word in tokens if word in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)

def predict_cluster(comment):
    tokens = clean_text_for_lda(comment, stopwords_lst, emoji_dict, teen_dict, eng_vn_dict)
    embedding = sentence_vector(tokens, fasttext_model)
    label, strength = hdbscan.approximate_predict(hdb_model, embedding.reshape(1, -1))
    return label[0], strength[0]

# Sidebar
st.sidebar.title("Navigator")
page1 = st.sidebar.selectbox("Page", ["Business Objective", "Sentiment Analysis", "Information Clustering"])

if page1 == "Sentiment Analysis":
    page2 = st.sidebar.selectbox("Details", ["Business Objective", "Build Project", "New Prediction"])
elif page1 == "Information Clustering":
    page2 = st.sidebar.selectbox("Details", ["Business Objective", "Build Project", "New Prediction"])
else:
    page2 = None

st.sidebar.markdown("---")
st.sidebar.subheader("Thành viên thực hiện")
st.sidebar.write("1. Nguyễn Trọng Hiến")
st.sidebar.write("tronghien97lx@gmail.com")
st.sidebar.write("2. Nguyễn Lê Châu Uyên")
st.sidebar.write("uyenngchau@gmail.com")

# Main content
if page1 == "Business Objective":
    st.image("cover_image.jpg", use_container_width=True)
    st.title("Sentiment Analysis & Information Clustering")
    st.subheader("Business Objective")
    st.markdown("""
    - **Mục tiêu dự án:** 
        + Phân loại đánh giá người dùng thành 3 cảm xúc: tích cực, trung lập, tiêu cực
        + So sánh độ chính xác giữa các mô hình học máy
        + Phân cụm đánh giá để khám phá chủ đề trong dữ liệu
        + Hỗ trợ doanh nghiệp nắm bắt về nhân sự, cải thiện tuyển dụng
    """)

elif page1 == "Sentiment Analysis":
    if page2 == "Business Objective":
        st.image("sentiment_cover.jpg", use_container_width=True)
        st.title("Sentiment Analysis")
        st.write("Sentiment analysis - phân tích tình cảm (hay còn gọi là phân tích quan điểm, phân tích cảm xúc, phân tính cảm tính) là cách sử dụng xử lý ngôn ngữ tự nhiên, phân tích văn bản, ngôn ngữ học tính toán, và sinh trắc học để nhận diện, trích xuất, định lượng và nghiên cứu các trạng thái tình cảm và thông tin chủ quan một cách có hệ thống.")
        st.subheader("Mục tiêu phân tích cảm xúc")
        st.markdown("""
        - **Phát hiện cảm xúc tích cực, tiêu cực, trung lập trong review của nhân viên về công ty.**
        - **Giúp doanh nghiệp hiểu rõ hơn về điểm mạnh/yếu trong trải nghiệm nhân viên.**
        """)
    elif page2 == "Build Project":
        st.title("Build Project: Sentiment Analysis")
        st.subheader("Số lượng nhãn cảm xúc")
        st.image('sentiment_label.jpg')
        st.markdown("""
        - **Nhận xét:** 
            + Positive có 6,028 đánh giá, chiếm tỷ lệ 73.76%, cho thấy hầu như người tham gia đánh giá phản hồi tích cực về các công ty IT
            + Neutral có 1,639 đánh giá, chiếm tỷ lệ 19.48%, cho thấy một bộ phận người tham gia đánh giá phản hồi trung lập
            + Negative có 570 đánh giá, chiếm tỷ lệ 6.77%, cho thấy đánh giá không hài lòng chiếm tỷ lệ khá thấp
        - **Đánh giá:** 
            + Dữ liệu mất cân bằng, nhóm Positive chiếm ưu thế rõ rệt → cần xử lý cân bằng khi huấn luyện mô hình
        """)  

        st.image('sentiment_confusion_matrix.jpg')

        st.subheader("**Các mô hình đã huấn luyện:** Logistic Regression, Naive Bayes, SVM")
        st.write("Bảng kết quả")
        st.image("sentiment_result.jpg", use_container_width=True)
        st.markdown("""
        - **Nhận xét:** 
            + Tất cả mô hình có Macro F1 > 80% → khả năng phân loại giữa 3 nhóm cảm xúc được cải thiện rõ rệt
            + Linear SVM vượt trội nhất ở mọi chỉ số (Accuracy, Macro F1, Weighted F1), kế đến là Logistic Regression cũng cho kết quả khá tốt
        """)    

    elif page2 == "New Prediction":
        st.title("Sentiment Analysis - New Prediction")
        comment = st.text_area("Nhập nhận xét:")
        if st.button("Dự đoán"):
            if comment:
                sentiment = predict_sentiment(comment)
                st.write(f"Dự đoán cảm xúc: **{sentiment.capitalize()}**")
            else:
                st.warning("Vui lòng nhập nhận xét!")

elif page1 == "Information Clustering":
    if page2 == "Business Objective":
        st.image("clustering_cover.jpg", use_container_width=True)
        st.title("Information Clustering")
        st.subheader("Mục tiêu phân cụm chủ đề")
        st.markdown("""
        - **Phân nhóm các review/công ty thành các chủ đề/cụm nội dung nổi bật.**
        - **Khám phá các khía cạnh quan trọng như lương thưởng, môi trường làm việc, đào tạo, chính sách,...**
        - **Hỗ trợ doanh nghiệp xác định điểm mạnh/yếu theo từng nhóm chủ đề cụ thể.**
        """)
    
    elif page2 == "Build Project":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dùng LDA để xác định số cụm = 3")
            st.markdown("""
            - **Tên các cụm:** 
                + 0: Môi trường làm việc & Đồng nghiệp
                + 1: Chính sách làm việc & Văn hóa công ty
                + 2: Lương & Phúc lợi
            """)
        with col2:
            st.image("lda_coherence.jpg", use_container_width=True)

        st.subheader("Trực quan từ khóa của mỗi cụm")
        st.image("clustering_wordcloud_0.jpg")
        st.image("clustering_wordcloud_1.jpg")
        st.image("clustering_wordcloud_2.jpg")
        st.subheader("**Các mô hình đã huấn luyện:** KMeans, Agglomerative Clustering, HDBSCAN")
        st.write("Bảng kết quả")
        st.image("clustering_result.jpg", use_container_width=True)
        st.image("clustering_visual.jpg", use_container_width=True)
        st.markdown("""
        - **Nhận xét:** 
            + HDBSCAN cho kết quả tốt nhất với Silhouette Score cao, thể hiện cụm phân biệt rõ ràng.
            + KMeans và Agglomerative có điểm số thấp, cho thấy phân cụm chưa thật sự hiệu quả.
        """)

    elif page2 == "New Prediction":
        st.title("Information Clustering - New Prediction")
        company_names = data['Company Name'].unique().tolist()
        company_name_input = st.selectbox("Chọn hoặc nhập tên công ty:", company_names)
        if st.button("Phân tích"):
            company_data = data[data['Company Name'] == company_name_input]
            if company_data.empty:
                st.error("Không tìm thấy công ty với tên đã nhập.")
            else:
                total_reviews = len(company_data)
                st.write(f"**Tên công ty:** {company_name_input}")
                st.write(f"**Số lượng đánh giá:** {total_reviews}")
                
                valid = company_data[company_data['cluster_hdbscan'] != -1]
                topic_names = {
                    0: "Môi trường & Đồng nghiệp",
                    1: "Chính sách & Phúc lợi",
                    2: "Lương & Dự án thực tế"
                }
                if not valid.empty:
                    cluster_label = valid['cluster_hdbscan'].mode()[0]
                    cluster_name = topic_names.get(cluster_label, f"Cụm {cluster_label}")
                    st.write(f"**Thuộc cụm HDBSCAN:** {cluster_label} – {cluster_name}")
                else:
                    st.write("Công ty không thuộc cụm HDBSCAN nào.")
                
                tokens = company_data['lda_tokens'].dropna().tolist()
                all_tokens = [t for lst in tokens if isinstance(lst, list) for t in lst]
                
                if all_tokens:
                    st.write("**Top 10 từ khóa phổ biến:**")
                    top_tokens = Counter(all_tokens).most_common(10)
                    for i, (kw, count) in enumerate(top_tokens, 1):
                        st.write(f"{i}. {kw} ({count} lần)")
                    
                    wordcloud_input = dict(Counter(all_tokens).most_common(20))
                    wordcloud = WordCloud(width=800, height=600, background_color='white', max_words=30).generate_from_frequencies(wordcloud_input)
                    wordcloud.to_file("wordcloud.png")
                    st.image("wordcloud.png", caption=f"Từ khóa nổi bật – {company_name_input}", use_container_width=True)
                else:
                    st.write("Không có từ khóa để hiển thị.")
                
                negative_data = company_data[company_data['sentiment'] == 'negative']
                neg_tokens = negative_data['lda_tokens'].dropna().tolist()
                neg_words = [t for tokens in neg_tokens if isinstance(tokens, list) for t in tokens]
                
                if neg_words:
                    st.write("**Đề xuất cải tiến dựa trên phản hồi tiêu cực:**")
                    top_neg = Counter(neg_words).most_common(5)
                    for i, (word, freq) in enumerate(top_neg, 1):
                        st.write(f"{i}. {word} ({freq} lần)")
                    
                    keyword_suggestions = {
                        'lương': 'Xem xét chính sách đãi ngộ và lương thưởng công bằng hơn.',
                        'sếp': 'Cải thiện kỹ năng quản lý và giao tiếp từ cấp trên.',
                        'quản_lý': 'Tăng cường minh bạch và hiệu quả trong cách điều hành.',
                        'áp_lực': 'Cân bằng khối lượng công việc và giảm áp lực không cần thiết.',
                        'môi_trường': 'Cải thiện môi trường làm việc và tạo không khí tích cực hơn.',
                        'đào_tạo': 'Tổ chức nhiều khóa đào tạo để nâng cao kỹ năng chuyên môn.',
                        'phúc_lợi': 'Bổ sung và cải thiện các chính sách phúc lợi hấp dẫn hơn.',
                        'chính_sách': 'Rà soát các chính sách nội bộ để đảm bảo công bằng và minh bạch.',
                        'công_việc': 'Rà soát lại phân công công việc để tránh quá tải và mơ hồ.',
                        'thăng_tiến': 'Thiết lập lộ trình phát triển và cơ hội thăng tiến rõ ràng.',
                        'giao_tiếp': 'Khuyến khích giao tiếp cởi mở giữa các phòng ban và quản lý.',
                        'giờ_làm': 'Xem xét lại thời gian làm việc và chính sách làm thêm giờ.',
                        'đánh_giá': 'Xây dựng quy trình đánh giá nhân viên công bằng và rõ ràng.',
                        'thiếu': 'Đảm bảo đủ nguồn lực, nhân sự và công cụ hỗ trợ cho công việc.',
                        'định_hướng': 'Hỗ trợ nhân viên về định hướng nghề nghiệp và mục tiêu cá nhân.',
                        'nội_bộ': 'Tăng tính minh bạch và hiệu quả trong truyền thông nội bộ.',
                        'văn_phòng': 'Nâng cấp cơ sở vật chất và không gian làm việc hiện đại hơn.',
                        'văn_hoá': 'Xây dựng văn hoá công ty tích cực và gắn kết.',
                        'phân_biệt': 'Xử lý nghiêm các hành vi phân biệt đối xử hoặc thiên vị.',
                        'cạnh_tranh': 'Tạo môi trường cạnh tranh lành mạnh thay vì áp lực gây mệt mỏi.',
                        'đồng_nghiệp': 'Thúc đẩy tinh thần đồng đội và hợp tác giữa các nhân viên.',
                        'chế_độ': 'Cập nhật chế độ nghỉ phép, thưởng lễ tết hợp lý hơn.',
                        'nghỉ_phép': 'Khuyến khích sử dụng quyền nghỉ phép để phục hồi năng lượng.',
                        'hỗ_trợ': 'Nâng cao khả năng hỗ trợ của quản lý khi nhân viên gặp khó khăn.',
                        'nội_quy': 'Đảm bảo nội quy rõ ràng và được phổ biến rộng rãi.',
                        'thái_độ': 'Xây dựng thái độ làm việc tích cực từ cấp trên và đồng nghiệp.'
                    }
                    suggested = set()
                    for word, _ in top_neg:
                        if word in keyword_suggestions and word not in suggested:
                            st.write(f"- {keyword_suggestions[word]}")
                            suggested.add(word)
                else:
                    st.write("Không có phản hồi tiêu cực để phân tích.")