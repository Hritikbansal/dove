import torch
import torch.utils.data as data
import json


def save_split(train, validation, name):
    t = open("data/aug_pref_data/train_" + name + ".jsonl", "w")
    t.writelines(train)
    t.close()
    t = open("data/aug_pref_data/validation_" + name + ".jsonl", "w")
    t.writelines(validation)
    t.close()


def load_datasets(ds):
    d = []
    for dataset in ds:
        f = open(dataset, "r")

        if "json" in dataset:
            lines = f.readlines()
            d += lines
        else:
            data = f.read()
            data = json.loads(data)
            d += [json.dumps(js) + "\n" for js in data]

        f.close()

    return d


seed = 42
ds = ["data/pref_data/openai_tldr_pref.json"]
dataset = load_datasets(ds)
generator = torch.Generator().manual_seed(seed)
train, validation = data.random_split(dataset, [0.8, 0.2], generator)  # this shuffles

save_split(train, validation, "aug_pref")
