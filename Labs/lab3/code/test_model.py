import os
import json
import torch
import numpy as np
from config import NerConfig
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data_loader import NerDataset
from model import BertNer
from tqdm import tqdm
from seqeval.metrics import classification_report

def test_model(model, test_loader, output_dir, id2label, device):
    model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model_ner.bin")))
    model.eval()

    preds = []
    for step, batch_data in enumerate(tqdm(test_loader)):
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        logits = model(input_ids, attention_mask).logits

        attention_mask = attention_mask.detach().cpu().numpy()
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            length = sum(attention_mask[i])
            logit = [id2label[j] for j in logits[i][1:length]]
            preds.append(logit)

    return preds

def show_pred(test_data, pred):
    for i in range(len(pred)):
        print(test_data[i]['text'])
        print(pred[i][:-1])
    
    extracted_entities = []
    extracted_labels = []
    for i in range(len(pred)):
        entities = []
        labels = []
        current_entity = ""
        current_label = ""
        for j, label in enumerate(pred[i]):
            if label == 'O':
                continue
            elif label.startswith('B'):
                if current_entity:
                    entities.append(current_entity)
                    labels.append(current_label)
                current_entity = test_data[i]['text'][j]
                current_label = label[2:]
            elif label.startswith('I') and current_entity:
                current_entity += test_data[i]['text'][j]
                if j + 1 == len(pred[i]) or not pred[i][j + 1].startswith('I'):
                    entities.append(current_entity)
                    labels.append(current_label)
                    current_entity = ""
                    current_label = ""
        
        if current_entity:
            entities.append(current_entity)
            labels.append(current_label)
        
        extracted_entities.append(entities)
        extracted_labels.append(labels)

    for entities, labels in zip(extracted_entities, extracted_labels):
        for entity, label in zip(entities, labels):
            print(f"{entity}: {label}", end='\t')
        print()

def create_json_data(text):
    data = {}
    data['text'] = [char for char in text]
    data['labels'] = ['O' for _ in text]
    data['id'] = 0
    return data

if __name__ == "__main__":
    # flag = 'testfile'
    flag = 'teststr'
    data_name = "duie/"
    args = NerConfig(data_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    model = BertNer(args).to(device)

    if flag == 'testfile':
        test_data = []
        with open(os.path.join(args.data_path, "test.json"), "r", encoding="utf-8") as fp:
            for line in fp:
                data = json.loads(line)
                data = create_json_data(data['text'])
                test_data.append(data)
    elif flag == 'teststr':
        test_data = []
        str_list = ['《民航客运服务会话》是1995年中国民航出版社出版的图书，作者是周石田',
                    '再有之后的《半生缘》，蒋勤勤饰演的顾曼璐完全把林心如的曼桢衬得像是涉世未深的小姑娘，毫无半点风情',
                    '裴友生，男，汉族，湖北蕲春人，1957年12月出生，大专学历',
                    '吴君如演的周吉是电影《花田喜事》，在周吉大婚之夜，其夫林嘉声逃走失踪，后来其夫新科状元高中回来，周吉急往城楼相识，但林嘉声却言夫妻情断，覆水难收']
        for i in range(len(str_list)):
            data = create_json_data(str_list[i])
            test_data.append(data)

    dev_dataset =  NerDataset(test_data, args, tokenizer)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2)

    pred = test_model(model, dev_loader, args.output_dir, args.id2label, device)
    show_pred(test_data, pred)

