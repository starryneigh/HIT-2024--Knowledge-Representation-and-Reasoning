import os
from TransE import TransE
from TransH import TransH
from TransR import TransR
from Config import FB15kConfig, WN18Config

def get_model_and_dataset():
    model_dict = {
        "TransE": TransE,
        "TransH": TransH,
        "TransR": TransR
    }
    dataset_dict = {
        "FB15k": FB15kConfig,
        "WN18": WN18Config
    }
    return model_dict, dataset_dict

def read_words(file_path):
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            head, tail, relation = line.strip().split()
            triples.append((head, relation, tail))
    return triples

def read_idxs(file_path):
    triples = []
    with open(file_path, 'r') as f:
        total = f.readline().strip().split()
        # print("total:", total)
        for line in f:
            head, tail, relation = line.strip().split()
            triples.append((int(head), int(relation), int(tail)))
    return triples

def words_to_ids(path):
    words = []
    word2id = {}
    with open(path, 'r') as f:
        for line in f:
            word, idx = line.strip().split()
            words.append(word)
            word2id[word] = int(idx)
    # print(len(words))
    return words, word2id

# 将三元组转换为索引表示
def convert_triples_to_index(triples, entity2id, relation2id):
    indexed_triples = []
    for head, relation, tail in triples:
        h_idx = entity2id[head]
        r_idx = relation2id[relation]
        t_idx = entity2id[tail]
        indexed_triples.append([h_idx, r_idx, t_idx])
    return indexed_triples

def read_dataset(folder):
    dataset = {}
    train_path = os.path.join(folder, 'train.txt')
    valid_path = os.path.join(folder, 'valid.txt')
    test_path = os.path.join(folder, 'test.txt')
    entity2id_path = os.path.join(folder, 'entity2id.txt')
    relation2id_path = os.path.join(folder, 'relation2id.txt')
    _1_1_path = os.path.join(folder, '1-1.txt')
    _1_n_path = os.path.join(folder, '1-n.txt')
    n_1_path = os.path.join(folder, 'n-1.txt')
    n_n_path = os.path.join(folder, 'n-n.txt')

    entities, entity2id = words_to_ids(entity2id_path)
    relations, relation2id = words_to_ids(relation2id_path)

    train = read_words(train_path)
    valid = read_words(valid_path)
    test = read_words(test_path)

    dataset['train'] = convert_triples_to_index(train, entity2id, relation2id)
    dataset['valid'] = convert_triples_to_index(valid, entity2id, relation2id)
    dataset['test'] = convert_triples_to_index(test, entity2id, relation2id)
    dataset['entity2id'] = entity2id
    dataset['relation2id'] = relation2id
    dataset['1-1'] = read_idxs(_1_1_path)
    dataset['1-n'] = read_idxs(_1_n_path)
    dataset['n-1'] = read_idxs(n_1_path)
    dataset['n-n'] = read_idxs(n_n_path)

    return dataset

if __name__ == '__main__':
    dataset = read_dataset('WN18')
    print('train:', dataset['train'][:5])
    print('valid:', dataset['valid'][:5])
    print('test:', dataset['test'][:5])
    print('1-1:', dataset['1-1'][:5])
    print('1-n:', dataset['1-n'][:5])
    print('n-1:', dataset['n-1'][:5])
    print('n-n:', dataset['n-n'][:5])
    