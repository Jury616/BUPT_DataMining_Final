# data_preprocessing.py
import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import json
import numpy as np
from gensim.models import Word2Vec

def load_and_clean_data(data_dir):
    """
    加载数据并清洗
    :param data_dir: 数据文件所在文件夹路径
    :return: 文本列表和标签列表
    """
    texts, labels = [], []

    # 遍历目录下的所有JSON文件
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)

        # 确保是JSON文件
        if not file_name.endswith('.json'):
            continue

        try:
            # 打开并加载JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 遍历JSON文件中的每条记录
                for record in data:
                    text = record.get("案情描述", "")
                    label = record.get("案件类别", "")

                    # 清洗文本：去除特殊字符和多余空白
                    text = re.sub(r'<[^>]+>', '', text)  # 去除HTML标签
                    text = re.sub(r'\s+', ' ', text).strip()  # 去除多余空白

                    if text and label:  # 确保文本和标签均存在
                        texts.append(text)
                        labels.append(label)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return texts, labels


def tokenize_texts_jieba(texts):
    """
    使用 Jieba 对文本列表进行分词
    :param texts: 原始文本列表
    :return: 分词后的文本列表 (list of lists of tokens)
    """
    return [list(jieba.cut(text)) for text in texts]

def vectorize_texts(texts_joined, method="tfidf", max_features=5000, stop_words=None):
    """
    向量化已分词并用空格连接的文本 (主要用于 BoW 和 TF-IDF)
    :param texts_joined: 文本列表 (每个文本是空格连接的词)
    :param method: "count" (BoW) 或 "tfidf"
    :param max_features: 最大特征数
    :param stop_words: 停用词列表
    :return: 文本特征矩阵, 向量化器实例
    """
    if method == "count":
        vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    else:
        raise ValueError("Invalid method. Choose 'count' or 'tfidf'.")

    features = vectorizer.fit_transform(texts_joined)
    return features, vectorizer

def train_word2vec_model(tokenized_texts, vector_size=100, window=5, min_count=2, workers=4, sg=0):
    """
    训练 Word2Vec 模型
    :param tokenized_texts: 分词后的文本列表 (list of lists of tokens)
    :param vector_size: 词向量维度
    :param window: 上下文窗口大小
    :param min_count: 忽略总频率低于此的单词
    :param workers: 训练并行数
    :param sg: 训练算法 (0 for CBOW, 1 for Skip-gram)
    :return: 训练好的 Word2Vec 模型
    """
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=window,
                     min_count=min_count, workers=workers, sg=sg)
    return model

def create_document_vectors_word2vec(tokenized_texts, w2v_model):
    """
    使用预训练的 Word2Vec 模型为文档创建平均词向量表示
    :param tokenized_texts: 分词后的文本列表 (list of lists of tokens)
    :param w2v_model: 训练好的 Word2Vec 模型
    :return: 文档向量的 NumPy 数组
    """
    document_vectors = []
    vector_size = w2v_model.vector_size
    for tokens in tokenized_texts:
        valid_tokens = [token for token in tokens if token in w2v_model.wv]
        if not valid_tokens:
            # 如果文档中没有词在 Word2Vec 词汇表中，使用零向量
            document_vectors.append(np.zeros(vector_size))
        else:
            # 计算文档中所有有效词向量的平均值
            document_vectors.append(np.mean(w2v_model.wv[valid_tokens], axis=0))
    return np.array(document_vectors)