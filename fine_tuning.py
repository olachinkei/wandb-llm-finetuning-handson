import glob
import os
import wandb
import json
import copy
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_PROJECT"] = "Autocompletion with evaluation"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "gradients"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=1024"

torch.cuda.empty_cache()
config = {
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": .05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "training": {
        "evaluation_strategy": "steps",
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "report_to": "wandb",
        "warmup_steps": 5,
        "max_steps": 100,
        "learning_rate": 2e-4,
        "fp16": True,
        "logging_steps": 5,
        "save_steps": 25,
        "output_dir": 'outputs'
    },
    "dataset_version": "downsampled",
    "MODEL_NAME": "Finetuned-Review-Autocompletion",
    "BASE_MODEL":"facebook/opt-125m",
}

PROMPT_NO_INPUT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response"""

PROMPT_WITH_INPUT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
Input:
{context}
### Response"""

class InstructDataset(Dataset):
    def __init__(self, json_list, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        
        for j in tqdm(json_list):
            # In cases like open_qa where context information is not necessary, there is no input column.
            # Therefore, we differentiate the template sentences based on whether the input column is present or not.
            if 'context' in j:
                source_text = PROMPT_WITH_INPUT_FORMAT.format_map(j)
            else:
                source_text = PROMPT_NO_INPUT_FORMAT.format_map(j)
            
            # Combine the instruction sentence and the response sentence, and insert an EOS token at the end
            example_text = source_text + j['response'] + self.tokenizer.eos_token
            
            # okenize only the instruction sentence (up to 'The following is a task to ~### Response:')
            # What we want is the length of the instruction sentence.
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )
            
            # Tokenize both the instruction sentence and the response sentence
            example_tokenized = self.tokenizer(
                example_text, 
                padding='longest', 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            input_ids = example_tokenized['input_ids'][0]
            
            # Copy the input sentence as is to be the correct answer that the LLM generates.
            labels = copy.deepcopy(input_ids)
            
            # Length up to the instruction sentence
            source_len = source_tokenized['length'][0]
            
            # Since the desired correct sentence for the LLM to generate also includes the instruction sentence,
            # we fill the section of the instruction sentence with -100 as IGNORE_INDEX to avoid calculating the CrossEntropyLoss.
            labels[:source_len] = self.ignore_index
            
            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
        
class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

with wandb.init(config=config, job_type="training") as run:
    # Setup for LoRa
    config = wandb.config
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    artifact = run.use_artifact('reviewco/Autocompletion with evaluation/data:production', type='dataset')
    artifact_dir = artifact.download()
    train_data = load_dataset('json', data_files=artifact_dir+'/train_data.json',split="train")
    valid_data = load_dataset('json', data_files=artifact_dir+'/valid_data.json',split="train")

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        #load_in_8bit=True,
        device_map='auto',
    )
    
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
    
    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()
    
    class CastOutputToFloat(nn.Sequential):
      def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    
    lora_config = LoraConfig(**wandb.config["lora_config"])
    model = get_peft_model(model, lora_config)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=InstructDataset(train_data, tokenizer),
        eval_dataset=InstructDataset(valid_data, tokenizer),
        args=transformers.TrainingArguments(**wandb.config["training"]),
        data_collator=InstructCollator(tokenizer)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    model.save_pretrained("output")

    model_ft = wandb.Artifact(f"finetuned-model", type="model")
    model_ft.add_dir("output")
    run.log_artifact(model_ft)
    run.log_code()