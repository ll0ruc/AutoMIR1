from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
from tqdm import tqdm
import random
from src.example import few_shot_example
from src.instruction import Gen_Query_Prompt as prompt
from src.utils import writejson_bench, llm_path_dict
random.seed(42)
import argparse


class Generate():
    def __init__(self, tokenizer, llm, prompt, task_names, path, save_file_path):
        self.tokenizer = tokenizer
        self.llm = llm
        self.path = path
        self.save_file_path = save_file_path
        self.prompt = prompt
        self.task_names = task_names

    def batch_list(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def read_data(self, path="", length=-1):
        all_texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                all_texts.append(row)
        if length > 0:
            all_texts = random.sample(all_texts, length)
        return all_texts

    def choice_example(self):
        examples = random.choices(few_shot_example, k=1)[0]
        ind = few_shot_example.index(examples)
        task = self.task_names[ind]
        return examples, task

    def process(self):
        self.sampling_params = SamplingParams(temperature=1.0, top_p=0.8, repetition_penalty=1.05,
                                              max_tokens=512)
        all_texts = self.read_data(self.path)
        batch_size = 128
        print(f"一共{len(all_texts)}个文本！")
        batchs = self.batch_list(all_texts, batch_size)
        results = []
        for batch in tqdm(batchs):
            batch_text = []
            batch_task = []
            for entry in batch:
                format_examples, task = self.choice_example()
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.prompt.format(format_examples, entry["doc"])}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                batch_text.append(text)
                batch_task.append(task)
            outputs = self.llm.generate(batch_text, self.sampling_params, use_tqdm=False)
            for entry, output, task in zip(batch, outputs, batch_task):
                generated_text = output.outputs[0].text
                entry["query"] = generated_text
                entry["task"] = task
                results.append(entry)
        new_results = []
        for result in results:
            new_results.append({
                "id": result['id'],
                "task": result['task'],
                "text": result['query'],
                "doc": result['doc']
            })
        writejson_bench(new_results, self.save_file_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default="**/train_data/corpus.jsonl",)
    parser.add_argument('--llm_name', type=str, default="Qwen-32b")
    args = parser.parse_args()
    path = args.corpus_path
    llm_name_gen_query = args.llm_name
    save_path = os.path.dirname(path) + "/query.jsonl"
    task_names = ['MedExamRetrieval', "DuBaikeRetrieval", "DXYDiseaseRetrieval",
                  "MedicalRetrieval", "CmedqaRetrieval", "DXYConsultRetrieval", "CovidRetrieval"]
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    ##Qwen2.5-32b-Instruct path
    llm_path = llm_path_dict[llm_name_gen_query]
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    llm = LLM(model=llm_path, tensor_parallel_size=4, disable_custom_all_reduce=True, trust_remote_code=True)
    print(f"llm have downloaded from {llm_path}!")
    gen = Generate(tokenizer, llm, prompt, task_names, path, save_path)
    gen.process()