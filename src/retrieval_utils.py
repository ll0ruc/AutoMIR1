from collections import defaultdict
import json
from datasets import load_dataset, DatasetDict
from mteb import AbsTaskRetrieval

dataset_root_dir = "***/dataset/"

def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    corpus = defaultdict(dict)
    with open(hf_hub_name + "/corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['id']
            corpus[pid] = {"text": e['text']}
    queries = {}
    query_path = hf_hub_name + "/qwen/query-rewrite-sft.jsonl"
    with open(query_path, "r", encoding="utf-8") as f:
        print(f"Current query file path is {query_path}")
        for line in f:
            e = json.loads(line)
            qid = e['id']
            if "rewrite" in query_path:
                queries[qid] = [e['text'], e["rewrite_query"]]
            else:
                queries[qid] = e['text']
    relevant_docs = defaultdict(dict)
    with open(hf_hub_name + "/qrels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            pid = e['p_id']
            qid = e['q_id']
            relevant_docs[qid][pid] = int(e["score"])
    qrels_num = 0
    for k, v in relevant_docs.items():
        qrels_num += len(v)
    corpus = DatasetDict({eval_split: corpus})
    queries = DatasetDict({eval_split: queries})
    relevant_docs = DatasetDict({eval_split: relevant_docs})
    print()
    print(f"共{len(queries[eval_split])}个queries,{len(corpus[eval_split])}个文章！,{qrels_num}个相关数据！")

    return corpus, queries, relevant_docs

class CovidRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CovidRetrieval',
            'hf_hub_name': dataset_root_dir + 'CovidRetrieval',
            'reference': 'https://aclanthology.org/2022.emnlp-main.357.pdf',
            'description': 'COVID-19 news articles',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CmedqaRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CmedqaRetrieval',
            'hf_hub_name': dataset_root_dir + 'CmedqaRetrieval',
            'reference': 'https://aclanthology.org/2022.emnlp-main.357.pdf',
            'description': 'Online medical consultation text',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class MedicalRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MedicalRetrieval',
            'hf_hub_name': dataset_root_dir + 'MedicalRetrieval',
            'reference': 'https://arxiv.org/abs/2203.03367',
            'description': 'Passage retrieval dataset collected from Alibaba search engine systems in medical domain',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True


class CSLRelatedRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CSLRelatedRetrieval',
            'hf_hub_name': dataset_root_dir + "CSLRelatedRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class CSLCiteRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'CSLCiteRetrieval',
            'hf_hub_name': dataset_root_dir + "CSLCiteRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class IIYIPostRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'IIYIPostRetrieval',
            'hf_hub_name': dataset_root_dir + "IIYIPostRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class DXYDiseaseRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'DXYDiseaseRetrieval',
            'hf_hub_name': dataset_root_dir + "DXYDiseaseRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class DXYConsultRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'DXYConsultRetrieval',
            'hf_hub_name': dataset_root_dir + "DXYConsultRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class MedExamRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MedExamRetrieval',
            'hf_hub_name': dataset_root_dir + "MedExamRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True

class DuBaikeRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'DuBaikeRetrieval',
            'hf_hub_name': dataset_root_dir + "DuBaikeRetrieval",
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(self.description['hf_hub_name'],
                                                                            self.description['eval_splits'])
        self.data_loaded = True
