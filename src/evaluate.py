import argparse
from retrieval_utils import *
from utils import FlagDRESModel
from mteb import MTEB
from time import time
from query_rewrite import query_rewrite

query_instruction_for_retrieval_dict = {
    "bge": "为这个句子生成表示以用于检索相关文章：",
    "bge-FT": "为这个句子生成表示以用于检索相关文章：",
    "peg": "为这个句子生成表示以用于检索相关文章：",
    "peg-FT": "为这个句子生成表示以用于检索相关文章：",
    "me5": "query: ",
    "me5-FT": "query: ",
    "piccolo": "查询: ",
}

model_path_dict = {
    "bge": "***/BAAI/bge-large-zh-v1.5",
    "bge-FT": "***/outputs/qwen/BGE-FT",
    "mcontriever": "***/facebook/mcontriever-msmarco",
    "m3e": "***/moka-ai/m3e-large",
    "peg": "***/TownsWu/PEG",
    "peg-FT": "***/outputs/qwen/PEG-FT",
    "text2vec": "***/GanymedeNil/text2vec-large-chinese",
    "me5": "***/intfloat/multilingual-e5-large",
    "me5-FT": "***/outputs/qwen/multilingual-e5-large-FT",
    "gte": "***/thenlper/gte-large-zh",
    "piccolo": "***/sensenova/piccolo-large-zh",
}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_name', type=str, required=True,
                        choices=["bge", "bge-FT", "mcontriever", "m3e", "peg", "peg-FT", "text2vec", "me5", "me5-FT", "gte",
                                 "piccolo"])
    parser.add_argument('--llm_name', type=str, required=True,
                        choices=["qwen", "llama", "chatglm"])
    args = parser.parse_args()
    retrieval_name = args.retrieval_name
    llm_name = args.llm_name
    if "FT" in retrieval_name:
        query_rewrite(llm_name)
    model_name_or_path = model_path_dict[retrieval_name]
    print(f"Current model is {model_name_or_path}")
    if retrieval_name in ["m3e", "mcontriever", "contriever", "text2vec"]:
        model = FlagDRESModel(mode="BertModel",
            model_name_or_path=model_name_or_path,
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              pooling_method="mean",
                              normalize_embeddings=False)
    elif retrieval_name in ["me5", "me5-FT1"]:
        model = FlagDRESModel(model_name_or_path=model_name_or_path,
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              doc_instruction_for_retrieval = "passage: ",
                              pooling_method="mean")
    elif retrieval_name in ["piccolo"]:
        model = FlagDRESModel(model_name_or_path=model_name_or_path,
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              doc_instruction_for_retrieval = "结果: ",
                              pooling_method="mean")
    elif retrieval_name in ["jina"]:
        model = FlagDRESModel(model_name_or_path=model_name_or_path,
                              pooling_method="mean")
    else:
        model = FlagDRESModel(model_name_or_path=model_name_or_path,
                              query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                              pooling_method="cls")


    task_names = ['MedExamRetrieval', "DuBaikeRetrieval", "DXYDiseaseRetrieval",
                  "MedicalRetrieval", "CmedqaRetrieval", "DXYConsultRetrieval", "CovidRetrieval",
                  "IIYIPostRetrieval", "CSLCiteRetrieval", "CSLRelatedRetrieval"]

    for task in task_names:
        t0 = time()
        if task in ['CovidRetrieval', 'CmedqaRetrieval', 'MedicalRetrieval',
                    'MedExamRetrieval', "DuBaikeRetrieval", "DXYDiseaseRetrieval", "DXYConsultRetrieval",
                    "IIYIPostRetrieval", "CSLCiteRetrieval", "CSLRelatedRetrieval",
                    ]:
            if retrieval_name in query_instruction_for_retrieval_dict:
                instruction = query_instruction_for_retrieval_dict[retrieval_name]
            else:
                instruction = None
        else:
            instruction = None
        model.query_instruction_for_retrieval = instruction
        print(f"current instruction is {model.query_instruction_for_retrieval}")
        evaluation = MTEB(tasks=[task], task_langs=['zh', 'zh-CN'])
        if llm_name == "":
            evaluation.run(model, output_folder=f"results/{retrieval_name}")
        else:
            evaluation.run(model, output_folder=f"results/{llm_name}-{retrieval_name}")
        print(f"{task} evaluation cost {(time()-t0)/60} minutes!")



