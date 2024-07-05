import torch
from torch.utils.data import DataLoader
import argparse
from dataset import read_dataset, get_model_and_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransE, TransH or TransR on FB15k or WN18")
    parser.add_argument("--model", type=str, default="TransE", help="TransE, TransH or TransR")
    parser.add_argument("--dataset", type=str, default="FB15k", help="FB15k or WN18")
    return parser.parse_args()

def train(config, model):
    device = config.device
    epochs = config.epochs
    eval_epoch = config.eval_epoch
    print_step = config.print_step
    data_dir = config.data_dir
    save_dir = config.save_dir
    save_epoch = config.save_epoch
    batch_size = config.batch_size
    shuffle = config.shuffle
    optimizer = config.optimizer
    lr = config.lr

    dataset = read_dataset(data_dir)
    train_dataset = torch.tensor(dataset['train'], dtype=torch.long, device=device)
    valid_dataset = torch.tensor(dataset['valid'], dtype=torch.long, device=device)

    ent_num = len(dataset['entity2id'])
    rel_num = len(dataset['relation2id'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    model = model(ent_num, rel_num, device).to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch.to(device))
            loss.backward()
            optimizer.step()
            if (i+1) % print_step == 0:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {loss.item()}")
        if (epoch+1) % eval_epoch == 0:
            hits, mean_rank = model.evaluate(valid_dataset)
            print("===============================================")
            print(f"Validation Hits@10: {hits}, mean rank: {mean_rank}")
            print("===============================================")
        if (epoch+1) % save_epoch == 0:
            model_name = model.__class__.__name__
            torch.save(model.state_dict(), f"{save_dir}/{data_dir + model_name}.pth")
            print("===============================================")
            print(f"Model saved as {save_dir}/{data_dir + model_name}.pth")
            print("===============================================")

if __name__ == '__main__':
    args = parse_args()
    model_dict, dataset_dict = get_model_and_dataset()
    
    # python train.py --model TransE --dataset WN18
    if args.model in model_dict and args.dataset in dataset_dict:
        model = model_dict[args.model]
        config = dataset_dict[args.dataset]
        train(config, model)
    else:
        RuntimeError("Model or dataset not found")
