# lda_model.py
from sklearn.decomposition import LatentDirichletAllocation

def train_lda(features, n_topics=10, random_state=42):
    """
    训练 LDA 模型并提取主题分布
    :param features: 文本词频矩阵
    :param n_topics: 主题数量
    :param random_state: 随机种子
    :return: LDA 模型和文本的主题分布
    """
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, n_jobs=-1) # n_jobs for parallelism
    topic_distributions = lda.fit_transform(features)  # 每篇文档的主题分布
    return lda, topic_distributions

def print_top_words(lda, feature_names, n_top_words=10):
    """
    打印每个主题的前几个关键词
    :param lda: 训练好的 LDA 模型
    :param feature_names: 词汇表
    :param n_top_words: 每个主题的关键词数量
    """
    print("\n--- LDA Top Words per Topic ---")
    for topic_idx, topic in enumerate(lda.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print("--- End LDA Top Words ---\n")