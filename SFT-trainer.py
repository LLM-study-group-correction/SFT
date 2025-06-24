from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTTrainer
import  os

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto"
    )
    return model, tokenizer


def load_dataset_formatted(data_path):
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        dataset = load_dataset("gsm8k", "main", split="train").select(range(20))

    return dataset.map(lambda sample: {
        "messages": [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    })


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = f"/ai/zr/Lib_LLM/resource_model/{model_name}"
    data_path = "/train.json"
    save_path = f"./model_save_sft/{model_name.split('/')[-1]}"

    # 检查路径
    if not os.path.exists(model_path):
        model_path = model_name  # 回退到Hugging Face Hub

    model, tokenizer = load_model_and_tokenizer(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset_formatted(data_path)

    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        optim="adamw_torch",
        num_train_epochs=4,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",  # 关键修改点
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # 新版TRL需要此参数
        args=training_args,
        train_dataset=dataset,
        max_seq_length=1024,
        formatting_func=lambda x: tokenizer.apply_chat_template(
            x["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    )

    trainer.train()
    trainer.save_model(os.path.join(save_path, "BEST"))


if __name__ == "__main__":
    main()
