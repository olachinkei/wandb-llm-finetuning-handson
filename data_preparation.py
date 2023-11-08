import os
import wandb
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_PROJECT"] = "Autocompletion with evaluation"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"

config = {
    "base_model":"facebook/opt-125m",
    "data_set":"databricks/databricks-dolly-15k",
}

with wandb.init(config=config, job_type="data_prep") as run:
    config = wandb.config

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    data = load_dataset(config.data_set,split="train")
    
    data_train_test = data.train_test_split(test_size=0.2)
    data_train_valid = data_train_test["train"]
    data_train_valid = data_train_valid.train_test_split(test_size=0.2)
    data_train = data_train_valid["train"]
    data_valid = data_train_valid["test"]
    data_test = data_train_test["test"]


    data_train_list = data_train.to_list()
    data_valid_list = data_valid.to_list()
    data_test_list = data_test.to_list()
    with open('raw_data/train_data.json', 'w') as f:
        json.dump(data_train_list, f)
    with open('raw_data/valid_data.json', 'w') as f:
        json.dump(data_valid_list, f)
    with open('raw_data/test_data.json', 'w') as f:
        json.dump(data_test_list, f)

    
    artifact = wandb.Artifact('data', type='dataset')
    artifact.add_dir("raw_data/")
    run.log_artifact(artifact, aliases=["raw"])