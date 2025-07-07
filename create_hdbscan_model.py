import pandas as pd
import numpy as np
import joblib
import gensim.downloader as api
import hdbscan
from underthesea import word_tokenize
import re

# Load dictionaries
def load_dict(filepath, sep='\t'):
    dct = {}
    try:
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line or sep not in line:
                    continue
                key, value = line.split(sep, 1)
                dct[key.strip().lower()] = value.strip().lower()
    except FileNotFoundError:
        print(f"Không tìm thấy file từ điển: {filepath}")
    return dct

def load_list(filepath):
    try:
        with open(filepath, 'r', encoding='utf8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Không tìm thấy file danh sách: {filepath}")
        return []

emoji_dict = load_dict('emojicon.txt')
teen_dict = load_dict('teencode.txt')
wrong_lst = load_list('wrong-word.txt')
stopwords_lst = load_list('vietnamese-stopwords.txt')
eng_vn_dict = load_dict('english-vnmese.txt')
custom_stopwords = ['công', 'ty', 'công ty', 'company', 'cty', 'công_ty', 'chúng tôi', 'chúng_tôi', 'có thể', 'có_thể', 'cái đó', 'cái_đó']
stopwords_lst += custom_stopwords

# Text processing
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

# Load data
try:
    overview_companies = pd.read_excel('Overview_Companies.xlsx')
    overview_reviews = pd.read_excel('Overview_Reviews.xlsx')
    reviews = pd.read_excel('Reviews.xlsx')
    data = reviews.merge(overview_reviews, on='id', how='left')
    data = data.merge(overview_companies, on='id', how='left')
except FileNotFoundError as e:
    print(f"Không tìm thấy file dữ liệu: {e}")
    exit()

# Create lda_tokens
data['all_text'] = data['What I liked'].fillna('') + ' ' + data['Suggestions for improvement'].fillna('')
data['lda_tokens'] = data['all_text'].apply(
    lambda x: clean_text_for_lda(x, stopwords_lst, emoji_dict, teen_dict, eng_vn_dict)
)

# Create embeddings
print("Đang tải mô hình FastText, vui lòng đợi...")
fasttext_model = api.load('fasttext-wiki-news-subwords-300')
def sentence_vector(tokens, model):
    vecs = [model[word] for word in tokens if word in model]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
X_embed = np.vstack([
    sentence_vector(tokens, fasttext_model)
    for tokens in data['lda_tokens'].fillna('').apply(lambda x: x if isinstance(x, list) else [])
])

# Train HDBSCAN
print("Đang huấn luyện HDBSCAN...")
hdb = hdbscan.HDBSCAN(min_cluster_size=40)
labels_hdb = hdb.fit_predict(X_embed)

# Save HDBSCAN model
joblib.dump(hdb, 'hdbscan_fasttext.pkl')
print("Đã lưu mô hình HDBSCAN vào hdbscan_fasttext.pkl")

# Save data with cluster_hdbscan for debugging
data['cluster_hdbscan'] = labels_hdb
data.to_excel('data_with_clusters.xlsx', index=False)
print("Đã lưu dữ liệu với cột cluster_hdbscan vào data_with_clusters.xlsx")