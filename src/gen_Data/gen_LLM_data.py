from transformers import AutoTokenizer
import torch
import json
from tqdm import tqdm
from collections import defaultdict
import os
from src.utils import FlagDRESModel
from mteb.abstasks.AbsTaskRetrieval import DRESModel
from time import time
from vllm import LLM, SamplingParams
from src.utils import writejson_bench, llm_path_dict
from src.instruction import Gen_LLM_Prompt as prompt
import random
random.seed(42)
import argparse

class Generate:
    def __init__(self, llm_path, emb_path, rewrite_prompt, llm_name):
        self.llm_path = llm_path
        self.emb_path = emb_path
        self.rewrite_prompt = rewrite_prompt
        self.llm_name = llm_name
        self.device = torch.device("cuda")

    def init_llm_model(self):
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path, trust_remote_code=True)
        self.llm_model = LLM(model=self.llm_path, tensor_parallel_size=4, disable_custom_all_reduce=True, trust_remote_code=True)
        print(f"LLM model {self.llm_path.split('/')[-1]} have been Loaded")

    def init_emb_model(self):
        self.emb_model = FlagDRESModel(
            model_name_or_path=self.emb_path,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            pooling_method='cls')
        print(f"Embedding model {self.emb_path.split('/')[-1]} have been loaded")

    def read_data(self, data_path):
        corpus = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                corpus.append(entry)
        return corpus

    def writed_data(self, data_path):
        writed_data = []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    writed_data.append(entry["text"] + entry["doc"])
        except FileNotFoundError:
            pass
        return writed_data


    @staticmethod
    def is_dres_compatible(model):
        DRES_METHODS = ["encode_queries", "encode_corpus"]
        for method in DRES_METHODS:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True
    def load_retrieval_data(self, data_path):
        corpus = defaultdict(dict)
        corpus2id = {}
        with open("***/train_data/corpus.jsonl", "r", encoding="utf-8") as f:
            for id, line in enumerate(f):
                e = json.loads(line)
                pid = "p_" + str(id)
                corpus[pid] = {"text": e['doc']}
                corpus2id[e['doc']] = pid
        queries = {}
        relevant_docs = defaultdict(dict)
        query_data = {}
        with open(data_path, "r", encoding="utf-8") as f:
            for id, line in enumerate(f):
                e = json.loads(line)
                e["id"] = id
                if e['doc'] not in corpus2id:
                    print(f"Passage Not Found! {e['doc']}")
                pid = corpus2id[e['doc']]
                query = e["text"]
                for idx, rewrite_query in enumerate(e["rewrite_queries"]):
                    qid = "q_" + str(e['id']) + "_" + str(idx)
                    queries[qid] = [query, rewrite_query]
                    relevant_docs[qid][pid] = 1
                query_data[e['id']] = e
        print(f"共{len(queries)}个queries,{len(corpus)}个文章！")
        return corpus, queries, relevant_docs, query_data

    def batch_list(self, data, batch_size, multiply):
        if multiply > 1:
            return [data[i:i + batch_size]*multiply for i in range(0, len(data), batch_size)]
        else:
            return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def build_prompt(self, mode, text):
        if mode == 'chatglm':
            return "<|user|>\n{}\n<|assistant|>".format(text)
        elif mode == 'llama':
            return "[INST] {} [/INST]".format(text)
        else:
            print(f"ErroR, Please input a valid mode: {mode}")

    def generate_by_llm(self, batch_size=64, data_path=None, save_path=None, multiply=5):
        llm_sampling_params = SamplingParams(temperature=1.2, top_p=0.9, top_k=50, max_tokens=512, n=1, presence_penalty=1.0)
        model = self.llm_model
        tokenizer = self.llm_tokenizer
        corpus_data = self.read_data(data_path)
        print(f"一共{len(corpus_data)}个文本！")
        batchs = self.batch_list(corpus_data, batch_size, multiply)
        results = defaultdict(list)
        for batch in tqdm(batchs):
            batch_text = []
            if self.llm_name == "qwen":
                for entry in batch:
                    content = self.rewrite_prompt.format(entry["text"], entry["doc"])
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    batch_text.append(text)
            elif self.llm_name in ["chatglm", "llama"]:
                for entry in batch:
                    queries = self.build_prompt(self.llm_name, self.rewrite_prompt.format(entry["text"], entry["doc"]))
                    batch_text.append(queries)
            else:
                print(f"Error, Please Input Right LLM!")
            outputs = model.generate(batch_text, llm_sampling_params, use_tqdm=False)
            for entry, output in zip(batch, outputs):
                generated_text = output.outputs[0].text.strip()
                results[entry["id"]].append(generated_text)
        new_data = []
        for data in corpus_data:
            new_data.append({
                "text": data["text"],
                "rewrite_queries": results[data["id"]],
                "doc": data['doc']
            })
        writejson_bench(new_data, save_path)

    def retrieval_by_emb(self, batch_size=512*2, data_path=None, save_path=None, score_function="cos_sim", **kwargs):
        corpus, queries, relevant_docs, query_data = self.load_retrieval_data(data_path)
        model = self.emb_model
        try:
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")
        model = model if self.is_dres_compatible(model) else DRESModel(model)
        from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
        model = DRES(
            model,
            batch_size=batch_size,
            corpus_chunk_size=50000,
            **kwargs,
        )
        retriever = EvaluateRetrieval(model, k_values=[1000],
                                      score_function=score_function)  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
        candidates = defaultdict(list)
        for q_id in tqdm(results.keys()):
            can = results[q_id]
            target_ids = list(relevant_docs[q_id].keys())[0]
            sorted_can = dict(sorted(can.items(), key=lambda item: item[1], reverse=True))
            p_ids = [d for d in sorted_can.keys()]
            if target_ids in p_ids:
                index = p_ids.index(target_ids)
                score = sorted_can[target_ids]
            else:
                index = 1000
                score = 0.0
            _, id, num = q_id.split("_")
            candidates[int(id)].append([int(num), index, score])
        new_data = []
        for id in candidates.keys():
            data = candidates[id]
            min_group = min(data, key=lambda x: x[1])
            best_id = min_group[0]
            entry = query_data[id]
            item = {
                "id": id,
                "text": entry["text"],
                "rerank_id": data,
                "target_id": best_id,
                "top_rewrite_query": entry["rewrite_queries"][best_id],
                "doc": entry["doc"]
            }
            new_data.append(item)
        writejson_bench(new_data, save_path)


    def generate(self, query_path="", train_data_path_init="", train_data_path=""):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        self.init_llm_model()
        print(">>> Begin Query Rewriting")
        self.generate_by_llm(batch_size=64, data_path=query_path, save_path= train_data_path_init, multiply=5)
        print("End Query Rewriting >>>")
        self.init_emb_model()
        print(">>> Begin Rewrite Query Reranking")
        self.retrieval_by_emb(data_path=train_data_path_init, save_path=train_data_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, default="**/train_data/query.jsonl")
    parser.add_argument('--llm_name', type=str, default="qwen")
    args = parser.parse_args()
    query_path = args.query_path
    llm_name_gen_llm = args.llm_name
    llm_path = llm_path_dict[llm_name_gen_llm]
    emb_path = "model_dir/BAAI/bge-large-zh-v1.5"
    train_data_path_init = os.path.dirname(query_path) + f"/{llm_name_gen_llm}/llm_train_data_init.jsonl"
    train_data_path = os.path.dirname(query_path) + f"/{llm_name_gen_llm}/llm_train_data.jsonl"
    if not os.path.exists(os.path.dirname(train_data_path)):
        os.makedirs(os.path.dirname(train_data_path))
    gen = Generate(llm_path, emb_path, prompt, llm_name_gen_llm)
    gen.generate(query_path, train_data_path_init, train_data_path)

