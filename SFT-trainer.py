from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTTrainer

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    return model, tokenizer

def load_dataset_formatted(data_path):
    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset.map(lambda sample: {
        "messages": [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    })

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = "/ai/zr/Lib_LLM/resource_model/" + model_name
    data_path = "/ai/zr/Lib_code/AAA_tool/B0_input_data/openai/gsm8k-main/train_top20.json"
    save_path = f"./model_save_sft/{model_name.split('/')[-1]}"

    model, tokenizer = load_model_and_tokenizer(model_path)
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
        evaluation_strategy="no",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        formatting_func=lambda x: tokenizer.apply_chat_template(x["messages"], tokenize=False)
    )

    trainer.train()
    trainer.save_model(save_path + "/BEST")

if __name__ == "__main__":
    main()
