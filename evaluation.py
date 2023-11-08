
import ctranslate2
import glob
import os
import pandas as pd
import peft
import random
import timeit
import urllib
import torch
import json
import os
from datasets import load_dataset
import lm_eval
from lm_eval import evaluator, utils
import wandb
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel, PeftConfig
import wandb.apis.reports as wr
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline

os.environ["WANDB_PROJECT"] = "Autocompletion with evaluation"
os.environ["WANDB_ENTITY"] = "reviewco"
os.environ["WANDB_USERNAME"] = "keisuke-kamata"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_WATCH"] = "gradients"
config = {
    "MODEL_NAME":"Finetuned-Review-Autocompletion",
    "BASE_MODEL":"facebook/opt-125m",
    "MODEL_REGISTRY":"Instruction-tuned-model",
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


config = {
    "BASE_MODEL":"facebook/opt-125m",
    "MODEL_REGISTRY":"Instruction-tuned-model",
}

def get_completion_responses_batch(prompts, base_llm,production_path, staging_path, tokenizer):
    # Get completions for each model in batches
    original_completions = get_huggingface_completion_batch_base(prompts, config, tokenizer)
    stagaing_completions = get_huggingface_completion_batch(prompts, staging_path, tokenizer)
    production_completions = get_huggingface_completion_batch(prompts, production_path, tokenizer)
    responses = []
    for ori, sta, pro in zip(original_completions, stagaing_completions, production_completions):
        responses.append({
            "Original": ori,
            "Staging (finetuned)": sta,
            "Production (finetuned)": pro
        })
    df = pd.DataFrame(responses)
    df.insert(0, "prompt", prompts)
    return df

def get_huggingface_completion_batch_base(prompts,config, tokenizer):
    completions = generate_output(config.BASE_MODEL,tokenizer,prompts)
    return completions
    
def get_huggingface_completion_batch(prompts, staging_path, tokenizer):
    model = PeftModel.from_pretrained(base_llm, staging_path,torch_dtype=torch.float16)
    model = model.merge_and_unload()
    completions = generate_output(model,tokenizer,prompts)
    return completions

def generate_output(model,tokenizer,prompts):
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    responses = generator(prompts, max_new_tokens=50)
    completions = []
    for i, prompt in enumerate(prompts):
        full_output = responses[i][0]["generated_text"]
        output = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
        completions.append(output.strip())
    return completions


with wandb.init(config=config,job_type="model_evaluation") as run:
    config = wandb.config
    # load -------------------------------
    base_llm = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    ## staging model
    staging_ar = wandb.use_artifact(f'{os.environ["WANDB_ENTITY"]}/model-registry/{config.MODEL_REGISTRY}:staging')
    staging_path = staging_ar.download()
    staging_model = PeftModel.from_pretrained(base_llm, staging_path,torch_dtype=torch.float16)
    staging_model = staging_model.merge_and_unload()
    ## production model
    production_ar = wandb.use_artifact(f'{os.environ["WANDB_ENTITY"]}/model-registry/{config.MODEL_REGISTRY}:production')
    production_path = production_ar.download()
    
    # evaluation harness -------------------------------
    
    tasks = ["arc_easy","hellaswag","drop","sciq","squad2"]
    results = evaluator.simple_evaluate(
        model=staging_model,
        tasks=tasks,
        batch_size=32,
        num_fewshot=3,
        device="cuda"  
    )
    table_contents = []
    table_contents.append(run.id)
    table_contents.append(results["results"]["arc_easy"]["acc"])
    table_contents.append(results["results"]["hellaswag"]["acc"])
    table_contents.append(results["results"]["squad2"]["f1"])
    table_contents.append(results["results"]["sciq"]["acc"])
    table_contents.append(results["results"]["drop"]["f1"])
    table = wandb.Table(columns=["run_id"]+tasks,
                                data=[table_contents])

    # evaluation to test-------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    artifact = run.use_artifact(f'{os.environ["WANDB_ENTITY"]}/{os.environ["WANDB_PROJECT"]}/data:production')
    artifact_dir = artifact.download()
    test_data = load_dataset('json', data_files=artifact_dir+'/test_data.json',split="train")

    prompts = []
    for i in range(0,10):
        if test_data[i]["context"]=="":
            source_text = PROMPT_NO_INPUT_FORMAT.format_map(test_data[i])
        else:
            source_text = PROMPT_WITH_INPUT_FORMAT.format_map(test_data[i])
        prompts.append(source_text)

    run.log({
        "Evaluation metric": table,
        "Validation Responses": get_completion_responses_batch(prompts, config.BASE_MODEL, production_path, staging_path, tokenizer)
    })
    run.log_code()

    report = wr.Report(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        title='Model Evaluation: Autocompletion Model',
        description="Data and sample predictions to evaluate the staging candidate model for our review autocompletion algorithm."
    )

    report.width = "fluid"

    runsets = [wr.Runset(
        os.environ["WANDB_ENTITY"],
        os.environ['WANDB_PROJECT']
        )]

    report.blocks = [
        wr.TableOfContents(),
        wr.H1("Report Overview"),
        wr.P(
            "This report contains information to evaluate whether potential staging models "
            "should be moved to production. Model Registry admins can use the view of the "
            "Model Registry at the end of this report to move a staging model into production, "
            "using a Webhook automation."
        ),
        wr.Spotify(spotify_id="7KAveXwQ5xzdHT6GDlNIBu"),
        wr.MarkdownBlock("May this staging model earn 5 stars 🙏."),
        wr.HorizontalRule(),
    ]

    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
        wr.ScalarChart(
            title="Current Min Eval Loss",
            metric="eval/loss",
            groupby_aggfunc="min",
            font_size="large"),

        wr.ScalarChart(
            title="Current Min Train Loss",
            metric="train/loss",
            groupby_aggfunc="min",
            font_size="large"),

        wr.ScalarChart(
            title="Longest Runtime (sec)",
            metric="Runtime",
            groupby_aggfunc="max",
            font_size="large"),

        wr.LinePlot(x='Step',
                    y=['eval/loss'],
                    smoothing_factor=0.8,
                    layout={'w': 24, 'h': 9})
        ]
    )

    report.blocks += [wr.H1("Key Metrics"), pg]

    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
            wr.WeavePanelSummaryTable("Evaluation metric", layout={'w': 24, 'h': 12}),
        ])

    report.blocks += [wr.H1("Evaluation metric"), pg]

    pg = wr.PanelGrid(
        runsets=runsets,
        panels=[
            wr.WeavePanelSummaryTable("Validation Responses", layout={'w': 24, 'h': 12}),
        ])

    report.blocks += [wr.H1("Sample Predictions"), pg]

    report.blocks += [wr.H1("Autocompletion Model in Model Registry"), wr.WeaveBlockArtifact(os.environ["WANDB_ENTITY"], "model-registry", "Instruction-tuned-model")]
    report.save()

    report_creation_msg = f"Report to review: {urllib.parse.quote(report.url, safe=r'/:')}"
    print(report_creation_msg)

    wandb.alert("New Staging Model Evaluated", report_creation_msg)
