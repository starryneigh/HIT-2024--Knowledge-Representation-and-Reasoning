from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontManager
import matplotlib
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python play.py <model_path>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"{sys.argv[1]} not found.")
        sys.exit(1)
    
    model_path = "./ckpt/my.model"
    matplotlib.rc("font",family='YouYuan')
    model = word2vec.Word2Vec.load(model_path)
    sim = model.wv.similarity("实事求是", "解放思想")
    print('\n'*3)
    print(f'sim(实事求是, 解放思想): {sim}')
    
    sim = model.wv.similarity("法国", "新年")
    print('\n'*3)
    print(f'sim(法国, 新年): {sim}')

    print('\n'*3)