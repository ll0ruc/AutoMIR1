import os
import json
from typing import cast, List, Dict, Union
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available
from transformers import BertTokenizer, BertModel

llm_path_dict = {
    "Qwen-32b": "model_dir/Qwen/Qwen2.5-32B-Instruct",
    "qwen": "model_dir/Qwen/Qwen2-7B-Instruct",
    "llama": "model_dir/meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "model_dir/THUDM/chatglm3-6b",
}


def writejson_bench(data, json_file_path):
    dir_path = os.path.dirname(json_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    num = 0
    with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
        for entry in data:
            jsonfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
            num += 1
    print(f"{json_file_path}共写入{num}条数据!")


class FlagDRESModel:
    def __init__(
            self,
            mode: str = "Automodel",
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            query_instruction_for_retrieval: str = None,
            doc_instruction_for_retrieval: str = None,
            batch_size: int = 256,
    ) -> None:

        if mode == "Automodel":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_pretrained(model_name_or_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
            self.model = BertModel.from_pretrained(model_name_or_path)
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.doc_instruction_for_retrieval = doc_instruction_for_retrieval
        self.normalize_embeddings = normalize_embeddings
        self.pooling_method = pooling_method
        self.batch_size = batch_size

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif is_torch_npu_available():
            self.device = torch.device("npu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.batch_size = self.batch_size * num_gpus


    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        if there is a instruction for queries, we will add it to the query text
        '''
        if isinstance(queries[0], str):
            if self.query_instruction_for_retrieval is not None:
                input_texts = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries]
            else:
                input_texts = queries
            print(f">>>> Encoding Queries <<<<")
            return self.encode(input_texts)
        else:
            queries_base = [q[0] for q in queries]
            queries_rewrite = [q[1] for q in queries]
            if self.query_instruction_for_retrieval is not None:
                input_texts_base = ['{}{}'.format(self.query_instruction_for_retrieval, q) for q in queries_base]
            else:
                input_texts_base = queries_base
            print(f">>>> Encoding Queries <<<<")
            emb_base = self.encode(input_texts_base)
            print(f">>>> Encoding Rewrite Queries <<<<")
            emb_rewrite = self.encode(queries_rewrite)
            emb_final = (emb_base + emb_rewrite) / 2
            return emb_final



    def encode_corpus(self, corpus: List[Union[Dict[str, str], str]], **kwargs) -> np.ndarray:
        '''
        This function will be used for retrieval task
        encode corpus for retrieval task
        '''
        if self.doc_instruction_for_retrieval is not None:
            input_texts = ['{}{}'.format(self.doc_instruction_for_retrieval, doc['text']).strip() for doc in corpus]
        else:
            input_texts = ['{}'.format(doc['text']).strip() for doc in corpus]
        print(f">>>> Encoding Corpus <<<<")
        return self.encode(input_texts)


    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self.model.eval()

        all_embeddings = []
        for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches", disable=len(sentences)<256):
            sentences_batch = sentences[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(self.device)
            last_hidden_state = self.model(**inputs, return_dict=True).last_hidden_state
            embeddings = self.pooling(last_hidden_state, inputs['attention_mask'])
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def pooling(self,
                last_hidden_state: torch.Tensor,
                attention_mask: torch.Tensor=None):
        if self.pooling_method == 'cls':
            return last_hidden_state[:, 0]
        elif self.pooling_method == 'mean':
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d