from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
from tqdm import tqdm
from collections import defaultdict
from utils import writejson_bench
from instruction import Rewrite_prompt

class Generate():
    def __init__(self, base_prompt, data_dir, save_file_path, tokenizer, sampling_params, llm):
        self.base_prompt = base_prompt
        self.data_dir = data_dir
        self.save_file_path = save_file_path
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.llm = llm

    def batch_list(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def read_data(self, data_dir):
        qrels = defaultdict(dict)
        with open(data_dir + "qrels.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qrels[row["q_id"]][row["p_id"]] = 1
        id2doc = {}
        with open(data_dir + "corpus.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                id2doc[row["id"]] = row["text"]
        all_texts = []
        with open(data_dir + "query.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                qid = row["id"]
                row["doc"] = [id2doc[pid] for pid in qrels[qid].keys()]
                all_texts.append(row)

        return all_texts

    def build_prompt(self, mode, text):
        if mode == 'chatglm':
            return "<|user|>\n{}\n<|assistant|>".format(text)
        elif mode == 'llama':
            return "[INST] {} [/INST]".format(text)
        else:
            print(f"ErroR, Please input a valid mode: {mode}")

    def process(self, mode="qwen"):
        all_texts = self.read_data(self.data_dir)
        dir_path = os.path.dirname(self.save_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        num = 0
        batch_size = 256
        print(f"一共{len(all_texts)}个query！")
        batchs = self.batch_list(all_texts, batch_size)
        results = []
        for batch in tqdm(batchs):
            batch_text = []
            if mode in ["qwen"]:
                for entry in batch:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.base_prompt.format(entry['text'])}
                    ]
                    # print(messages)
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_text.append(text)
            elif mode in ["chatglm", "llama"]:
                for entry in batch:
                    queries = self.build_prompt(mode, self.base_prompt.format(entry['text']))
                    batch_text.append(queries)

            else:
                print(f"Error, Please Input Right LLM!")
            outputs = self.llm.generate(batch_text, self.sampling_params, use_tqdm=False)
            for entry, output in zip(batch, outputs):
                generated_text = output.outputs[0].text

                results.append({
                    "id": entry["id"],
                    "text": entry["text"],
                    "rewrite_query": generated_text,
                    "doc": entry["doc"]
                })
                num += 1
        writejson_bench(results, self.save_file_path)


def query_rewrite(llm_name):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    model_path = f"***/outputs/{llm_name}/merge/{llm_name}-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Pass the default decoding hyperparameters of Qwen2-7B-Instruct
    # max_tokens is for the maximum length for generation.
    if llm_name == "llama":
        sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, seed=42, presence_penalty=0,frequency_penalty=0, max_tokens=512,repetition_penalty=1.2)
    else:
        sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=-1, seed=42, presence_penalty=0,
                                         frequency_penalty=0, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path, tensor_parallel_size=4, disable_custom_all_reduce=True, trust_remote_code=True)
    print(f"llm have downloaded from {model_path}!")
    data_names = ['MedExamRetrieval', "DuBaikeRetrieval", "DXYDiseaseRetrieval",
                      "MedicalRetrieval", "CmedqaRetrieval", "DXYConsultRetrieval", "CovidRetrieval",
                      "IIYIPostRetrieval", "CSLCiteRetrieval", "CSLRelatedRetrieval"]
    for data_name in data_names:
        if data_name in ["IIYIPostRetrieval", "CSLCiteRetrieval"]:
            base_prompt = Rewrite_prompt["T2P"]
        elif data_name in ["CSLRelatedRetrieval"]:
            base_prompt = Rewrite_prompt["P2P"]
        elif data_name in ["MedicalRetrieval", "CmedqaRetrieval", "DXYConsultRetrieval"]:
            base_prompt = Rewrite_prompt["Q2P"]
        else:
            base_prompt = Rewrite_prompt["Q2P"]
        data_dir = f"***/dataset/{data_name}/"
        save_path = f"***/{data_name}/{llm_name}/query-rewrite-sft.jsonl"
        print(f">>> Current is {data_name}")
        qua = Generate(base_prompt, data_dir, save_path, tokenizer, sampling_params, llm)
        qua.process(llm_name)