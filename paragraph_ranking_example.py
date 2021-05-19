import json
import os
import sys

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from transformers import T5ForConditionalGeneration
from pygaggle.model import QueryDocumentBatchTokenizer


def t5_ranking():
    # class MonoT5(Reranker):
    #     def __init__(self,
    #                  model: T5ForConditionalGeneration = None,
    #                  tokenizer: QueryDocumentBatchTokenizer = None):
    #         self.model = model or self.get_model()
    #         self.tokenizer = tokenizer or self.get_tokenizer()
    #         self.device = next(self.model.parameters(), None).device

    #     reranker_t5 = MonoT5(model=xy, tokenizer=xy)
    # reranker_t5 = MonoT5()

    reranker_t5 = MonoT5(from_pretrained="training_res/checkpoint-150/")

    # from pygaggle.rerank.transformer import MonoBERT
    # reranker_bert = MonoBERT()

    # # Option 1: fetch some passages to rerank from MS MARCO with Pyserini
    # # to use this on custom data, need to build a Lucene indexing
    # # ------------------------------------------------------------------------
    # from pyserini.search import SimpleSearcher
    # from pygaggle.rerank.base import hits_to_texts
    #
    # searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')
    # hits = searcher.search(query.text)
    # texts = hits_to_texts(hits)

    # Option 2: here's what Pyserini would have retrieved, hard-coded
    # ------------------------------------------------------------------------
    # create passages with structure:
    # [
    #   ['123456_id', 'passage text'],
    #   ['123456_id', 'passage text'],
    #   ['123456_id', 'passage text'],
    #   ['123456_id', 'passage text']
    # ]

    questions_path = "test_questions.json"
    with open(questions_path, "r") as q_file:
        questions_dict = json.load(q_file)

    print("Q loaded")

    pdf_dict_paths = [os.path.join("paragraphs", name) for name in
                      os.listdir("paragraphs") if name.startswith('pdf_dict')]

    for pdf_dict_path in pdf_dict_paths:
        with open(pdf_dict_path, "r") as pdf_file:
            pdf_dict = json.load(pdf_file)["pdf_dict"]
            print(pdf_dict_path + " loaded")

        # collect all paragraphs
        passages = []
        pdf_name = pdf_dict["pdf_name"]
        for par_id, par_value in pdf_dict["paragraphs"].items():
            passages.append([par_id, par_value["text"]])

        texts = [Text(p[1], {'docid': p[0]}, 0) for p in
                 passages]  # Note, pyserini scores don't matter since T5 will ignore them.

        candidates = {}
        for q_id, q_value in questions_dict.items():
            query = Query(q_value["question_text"])
            # Finally, rerank:

            reranked = reranker_t5.rerank(query, texts)
            top_n = 20
            ranked_list = []
            for i in range(0, top_n):
                # print(f'{i+1:2} {reranked[i].metadata["docid"]:15} {reranked[i].score:.5f} {reranked[i].text}')
                ranked_list.append(
                    {"rank": reranked[i].score, "par_id": reranked[i].metadata["docid"],
                     "text": reranked[i].text})

            candidates.update({q_id: ranked_list})

        candidates_dict = {"pdf_name": pdf_name, "candidates": candidates}

        with open(f"{pdf_name}_10ep_train_t5_candidates.json", "w") as outf:
            outf.write(json.dumps(candidates_dict, indent=4))

def get_pdf_question_labels(pdf_name, q_id, labels):
    par_id_collect = []
    q_labels = []
    for label in labels:
        label_id = label["unique_id"]
        par_id = label["par_id"]

        if pdf_name == label["pdf_title"] and q_id == label["question_id"] and par_id not in par_id_collect:
            par_id_collect.append(par_id)
            q_labels.append(label)
    return list(set(par_id_collect)), q_labels


def compare_label_to_candidate_list(path_candidate, path_labels="/Users/nikolettatoth/T5_ranking/pygaggle/test_files/gt_labels_for_pdfs/gt_labels_q5q6q17_raw.json"):
    with open(path_candidate, "r") as c_file:
        candidates = json.load(c_file)
    with open(path_labels, "r") as l_file:
        labels = json.load(l_file)
    match_res = []
    for q_id, candidate_list in candidates["candidates"].items():
        # if a question in a pdf has more than 1 from the same paragraph ( it can be bcof multiple answers) count it only once

        # get gt_labels for pdf for questions
        pdf_name = candidates["pdf_name"]
        gt_par_ids, q_pdf_labels = get_pdf_question_labels(pdf_name, q_id, labels)
        match_labels = []
        cnt = 0

        match_label_ids = []
        for cand in candidate_list:
            if cand["par_id"] in gt_par_ids:
                cnt += 1
                for item in q_pdf_labels:
                    if cand["par_id"] == item["par_id"]:
                        match_labels.append(item)
                        match_label_ids.append(item["unique_id"])

        missed_labels = []
        for label_item in q_pdf_labels:
            if label_item["unique_id"] not in match_label_ids:
                missed_labels.append(label_item)

        if "t5_candidates" in path_candidate:
            pre_name = "T5"
        else:
            pre_name = "own"
        #print(f"{pre_name}: {pdf_name}: Q{q_id} match/gt : {cnt}/{len(gt_par_ids)}")
        if len(gt_par_ids) > 0:
            match_res.append({"question":q_id, "res":f"{cnt}/{len(gt_par_ids)}", "ranking": pre_name, "found_labels": match_labels, "missed_labels": missed_labels})

    res = {pdf_name: match_res }
    print(res)
    return res

if __name__=="__main__":

    # GENERATE T5 PREDICTIONS
    # t5_ranking()
    # sys.exit()

    # COMPARE T5 PREDICTIONS BEFORE TRAINING
    # list_of_candidates = ["test_files/compare_results/before_training/candidates_2018_BORUSSIA DORTMUND.json",
    #                       "test_files/compare_results/before_training/2018_BORUSSIA DORTMUND.pdf_t5_candidates.json",
    #                       "test_files/compare_results/before_training/candidates_2019_CROPENERGIES AG.json",
    #                       "test_files/compare_results/before_training/2019_CROPENERGIES AG.pdf_t5_candidates.json",
    #                       "test_files/compare_results/before_training/candidates_2019_INSTONE REAL ESTATE GROUP O.N..json",
    #                       "test_files/compare_results/before_training/2019_INSTONE REAL ESTATE GROUP O.N._t5_candidates.json"]

    # COMPARE T5 PREDICTIONS AFTER TRAINING
    list_of_candidates = ["test_files/compare_results/after_training/candidates_2018_BORUSSIA DORTMUND.json",
                          "test_files/compare_results/after_training/2018_BORUSSIA DORTMUND.pdf_10ep_train_t5_candidates.json",
                          "test_files/compare_results/after_training/candidates_2019_CROPENERGIES AG.json",
                          "test_files/compare_results/after_training/2019_CROPENERGIES AG.pdf_10ep_train_t5_candidates.json",
                          "test_files/compare_results/after_training/candidates_2019_INSTONE REAL ESTATE GROUP O.N..json",
                          "test_files/compare_results/after_training/2019_INSTONE REAL ESTATE GROUP O.N..pdf_10ep_train_t5_candidates.json"]

    final_res = []
    for can in list_of_candidates:
        res = compare_label_to_candidate_list(path_candidate=can)
        final_res.append(res)

    # with open("final_comparison_res_labels.json", "w") as outf:
    #     outf.write(json.dumps(final_res))

    reorg_results = {}
    for res_item in final_res:
        for pdfname_key, res_list in res_item.items():
            temp_collect_res=[]
            for list_item in res_list:
                ranking = list_item["ranking"]
                question = list_item["question"]
                res = list_item["res"]
                temp_collect_res.append({
                    "candidates_"+ranking: res,
                    "question": question
                })
            if pdfname_key in reorg_results:
                collect_res = reorg_results[pdfname_key]
                for tcr in temp_collect_res:
                    for cr in collect_res:
                        if tcr["question"] == cr["question"]:
                            tcr.update(cr)
            reorg_results.update({pdfname_key: temp_collect_res})
    print(reorg_results)
    with open("test_files/compare_results/after_training/final_comparison_res_after_training.json", "w") as outf:
        outf.write(json.dumps(reorg_results))