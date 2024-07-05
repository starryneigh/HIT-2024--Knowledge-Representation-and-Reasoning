from BERT import DistilBertForSequenceClassification
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import DownloadMode


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("./cache/distilbert", num_labels=2, id2label=id2label, label2id=label2id)

dataset = load_dataset("stanfordnlp/sst2", cache_dir="./cache", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
accuracy = evaluate.load("accuracy", cache_dir="./cache")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = accuracy.compute(predictions=predictions, references=labels)
    print(res)
    return res

def tokenize_and_align_input_ids(examples):
    tokenized_inputs = tokenizer(examples["sentence"], padding="max_length", truncation=True)
    return tokenized_inputs

# data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
tokenized_dataset = dataset.map(tokenize_and_align_input_ids)

print(tokenized_dataset["train"])

###########训练参数
training_args = TrainingArguments(
    output_dir="./ckpt/CLS_ckpt",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    save_strategy="epoch",
)

###############模型训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

