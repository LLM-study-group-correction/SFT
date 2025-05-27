from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import re

def process_data(qA_dict, s_dict, role):
    sqA_dict_list = [
        {"role": "system", "content": s_dict[role]},
        {"role": "user", "content": qA_dict["q"]},
        {"role": "assistant", "content": qA_dict[f'A_{role}']}
    ]
    sq_ST = tokenizer.apply_chat_template(sqA_dict_list[:-1], tokenize=False, add_generation_prompt=True)
    sqA_ST = tokenizer.apply_chat_template(sqA_dict_list, tokenize=False, add_generation_prompt=False)
    sqA_idAtten_ST = tokenizer(sqA_ST, truncation=False, max_length=9999, add_special_tokens=False)
    sq_id_ST_length = len(tokenizer(sq_ST, add_special_tokens=False)["input_ids"])
    A_id_ST = [-100]*sq_id_ST_length + sqA_idAtten_ST["input_ids"][sq_id_ST_length:]
    assert len(sqA_idAtten_ST["input_ids"]) == len(A_id_ST)
    return {"input_ids": sqA_idAtten_ST["input_ids"], "attention_mask": sqA_idAtten_ST["attention_mask"], "labels": A_id_ST}

def test(model, tokenizer, s_dict, qA_dict_list, role="helpful"):
    messages = [
        {"role": "system", "content": s_dict[role]},
        {"role": "user", "content": qA_dict_list[0]["q"]},
        {"role": "assistant", "content": qA_dict_list[0][f'A_{role}']}
    ]
    sq_ST = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    sq_idAtten_ST = tokenizer(sq_ST, return_tensors="pt", padding=True, padding_side='left', add_special_tokens=False).to(model.device)
    sqa_id_list_ST = model.generate(sq_idAtten_ST.input_ids, attention_mask=sq_idAtten_ST.attention_mask, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    sqa_ST = tokenizer.decode(sqa_id_list_ST[0], skip_special_tokens=False)
    print(f"\n=== Generated Answer ===\n{sqa_ST}\n")

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()

    # 示例数据
    qA_dict_list = [{
        "q": "小明有3个苹果，又买了5个，一共多少个？",
        "A_helpful": "小明原来有3个苹果，后来又买了5个，所以总共有 3 + 5 = \\boxed{8} 个苹果。"
    }]

    s_dict = {
        "helpful": """[HELPFUL] You are a helpful answerer:
- Provide a step-by-step correct solution.
- Final numeric result must be enclosed in \\boxed{}."""
    }

    dataset = Dataset.from_dict({
        "q": [item["q"] for item in qA_dict_list],
        "A_helpful": [item["A_helpful"] for item in qA_dict_list]
    }).map(lambda x: process_data(x, s_dict, "helpful"))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    dataloader = DataLoader(dataset, batch_size=1)

    # 模拟训练步骤（这里只跑一次）
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            print("Loss:", outputs.loss.item())

    # 执行生成测试
    test(model, tokenizer, s_dict, qA_dict_list)
