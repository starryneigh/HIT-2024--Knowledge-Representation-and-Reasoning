from gensim.models import word2vec
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python word2vec.py <corpus_path> <model_path>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"{sys.argv[1]} not found.")
        sys.exit(1)
    corpus_path = sys.argv[1]
    model_path = sys.argv[2]
    # 加载语料
    sentences = word2vec.Text8Corpus(corpus_path)

    # 训练模型
    window = 3 # 窗口大小
    vector_size = 16  # 嵌入向量维度
    sg = 1 # 是否使用skip-gram 0表示否
    epochs = 10 # 训练轮次
    seed = 42 # 随机种子


    model = word2vec.Word2Vec(sentences, window=window, vector_size=vector_size, epochs=epochs, seed=seed, sg=sg)

    model.save(model_path)