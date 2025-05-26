from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.utils import writejson_bench
import json
import random
import numpy as np
import faiss
from tqdm import tqdm
import os
from src.utils import FlagDRESModel
import argparse

class Generate():
    def __init__(self, base_prompt, path, save_file_path, tokenizer, sampling_params, llm):
        self.base_prompt = base_prompt
        self.path = path
        self.save_file_path = save_file_path
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.llm = llm

    def batch_list(self, data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def read_data(self, path):
        all_texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                query = row["text"]
                answer = row["doc"]
                all_texts.append({"query": query, "pos": [answer]})
        return all_texts

    def build_prompt(self, mode, text):
        if mode == 'chatglm':
            return "<|user|>\n{}\n<|assistant|>".format(text)
        elif mode == "llama":
            return "[INST] {} [/INST]".format(text)
        else:
            print(f"ErroR, Please input a valid mode: {mode}")

    def process(self, llm_name):
        all_texts = self.read_data(self.path)
        dir_path = os.path.dirname(self.save_file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        num = 0
        batch_size = 256
        print(f"一共{len(all_texts)}个文本！")
        batchs = self.batch_list(all_texts, batch_size)
        results = []
        for batch in tqdm(batchs):
            batch_text = []
            if llm_name == "qwen":
                for entry in batch:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": self.base_prompt.format(entry['query'])}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_text.append(text)
            elif llm_name in ["chatglm", "llama"]:
                for entry in batch:
                    queries = self.build_prompt(llm_name, self.base_prompt.format(entry['query']))
                    batch_text.append(queries)
            else:
                print(f"Error, Please Input Right LLM!")
            outputs = self.llm.generate(batch_text, self.sampling_params, use_tqdm=False)
            for entry, output in zip(batch, outputs):
                generated_text = output.outputs[0].text
                entry['rewrite_query'] = generated_text
                num += 1
                results.append(entry)
        writejson_bench(results, self.save_file_path)


class Hn_mine():
    def __init__(self, model="", input_file="", output_file="", range_for_sampling=[5,100], negative_number=8):
        self.model = model
        self.input_file = input_file
        self.output_file = output_file
        self.range_for_sampling = range_for_sampling
        self.negative_number = negative_number

    def create_index(self, embeddings):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        embeddings = np.asarray(embeddings, dtype=np.float32)
        index.add(embeddings)
        return index

    def batch_search(self, index, query, topk: int = 200, batch_size: int = 64):
        all_scores, all_inxs = [], []
        for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
            batch_query = query[start_index:start_index + batch_size]
            batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
            all_scores.extend(batch_scores.tolist())
            all_inxs.extend(batch_inxs.tolist())
        return all_scores, all_inxs

    def find_knn_neg(self):
        corpus = []
        queries = []
        train_data = []
        for line in open(self.input_file):
            line = json.loads(line.strip())
            train_data.append(line)
            corpus.extend(line['pos'])
            if 'neg' in line:
                corpus.extend(line['neg'])
            queries.append(line['query'])
        corpus = list(set(corpus))

        print(f'inferencing embedding for corpus (number={len(corpus)})--------------')
        p_vecs = self.model.encode(corpus, batch_size=512)
        print(f'inferencing embedding for queries (number={len(queries)})--------------')
        q_vecs = self.model.encode_queries(queries, batch_size=256)
        print('create index and search------------------')
        index = self.create_index(p_vecs)
        all_scores, all_inxs = self.batch_search(index, q_vecs, topk=self.range_for_sampling[-1])
        assert len(all_inxs) == len(train_data)
        for i, data in enumerate(tqdm(train_data, desc='Processing data')):
            query = data['query']
            rewrite_query = data['rewrite_query']
            inxs = all_inxs[i][self.range_for_sampling[0]:self.range_for_sampling[1]]
            filtered_inx = []
            for inx in inxs:
                if inx == -1: break
                if corpus[inx] not in data['pos'] and corpus[inx] != query and corpus[inx] != rewrite_query:
                    filtered_inx.append(inx)
            if len(filtered_inx) > self.negative_number:
                filtered_inx = random.sample(filtered_inx, self.negative_number)
            data['neg'] = [corpus[inx] for inx in filtered_inx]

        with open(self.output_file, 'w') as f:
            for data in train_data:
                if len(data['neg']) < self.negative_number:
                    samples = random.sample(corpus, self.negative_number - len(data['neg']) + len(data['pos']))
                    samples = [sent for sent in samples if sent not in data['pos']]
                    data['neg'].extend(samples[: self.negative_number - len(data['neg'])])
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, default="qwen")
    args = parser.parse_args()
    llm_name = args.llm_name
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    model_path = f"***/outputs/{llm_name}/merge/{llm_name}-sft"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, seed=42, repetition_penalty=1.05, max_tokens=512)
    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path, tensor_parallel_size=4, disable_custom_all_reduce=True, trust_remote_code=True)
    print(f"llm have downloaded from {model_path}!")
    if llm_name == "llama":
        Rewrite_prompt = '''请用中文生成一段医学内容来回答这个医学问题。
    <问题>: {}
    <段落>: '''
    else:
        Rewrite_prompt = '''请生成一段医学内容来回答这个医学问题。
    <问题>: {}
    <段落>: '''
    base_prompt = Rewrite_prompt
    path = f'***/train_data/{llm_name}/llm_train_data.jsonl'
    save_path = f"***/train_data/{llm_name}/emb_train_data_init.jsonl"
    print(f">>> Current is {path}")
    qua = Generate(base_prompt, path, save_path, tokenizer, sampling_params, llm)
    qua.process(llm_name)
    print(f">>> Begin hard negative mining...")
    emb_model_path = "***/BAAI/bge-large-zh-v1.5"
    final_save_path = f"***/train_data/{llm_name}/emb_train_data.jsonl"
    emb_model = FlagDRESModel(model_name_or_path=emb_model_path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
    hn_mine = Hn_mine(model=emb_model, input_file=save_path, output_file=final_save_path, range_for_sampling=[5, 100], negative_number=8)
    hn_mine.find_knn_neg()