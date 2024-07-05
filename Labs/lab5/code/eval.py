import torch
import argparse
from dataset import read_dataset, get_model_and_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransE, TransH or TransR on FB15k or WN18")
    parser.add_argument("--model", type=str, default="TransE", help="TransE, TransH or TransR")
    parser.add_argument("--dataset", type=str, default="FB15k", help="FB15k or WN18")
    return parser.parse_args()

def test_print(model, dataset):
    hits, mean_rank = model.evaluate(dataset)
    print("===============================================")
    print(f"Test Hits@10: {hits}, mean rank: {mean_rank}")
    print("===============================================")

def eval(Config, model):
    device = Config.device
    data_dir = Config.data_dir
    save_dir = Config.save_dir

    dataset = read_dataset(data_dir)
    test_dataset = torch.tensor(dataset['test'], dtype=torch.long, device=device)
    _n_n_dataset = torch.tensor(dataset['n-n'], dtype=torch.long, device=device)

    ent_num = len(dataset['entity2id'])
    rel_num = len(dataset['relation2id'])

    model = model(ent_num, rel_num, device).to(device)
    model.load_state_dict(torch.load(f"{save_dir}/{data_dir + model.__class__.__name__}.pth"))

    test_print(model, test_dataset)
    test_print(model, _n_n_dataset)

if __name__ == '__main__':
    args = parse_args()
    model_dict, dataset_dict = get_model_and_dataset()
    
    # python eval.py --model TransE --dataset FB15k
    if args.model in model_dict and args.dataset in dataset_dict:
        model = model_dict[args.model]
        Config = dataset_dict[args.dataset]
        eval(Config, model)
    else:
        RuntimeError("Model or dataset not found")

