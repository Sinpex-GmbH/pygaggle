"""
Data Prep
https://github.com/castorini/pygaggle/blob/master/docs/experiments-monot5-tpu.md

Gives example dataset and how to convert it to use with t5 model training


"""
import json
import csv
fpath = "/Users/nikolettatoth/T5_ranking/pygaggle/test_files/labels_for_training_q5q6q17/train_ar_tf_q5q6q17_v1.json"

with open(fpath, "r") as infile:
    train_data = json.load(infile)

collect_dict = {}

for label in train_data["data"]:
    question = label["paragraphs"][0]["qas"][0]["question"]
    context = label["paragraphs"][0]["context"]

    if question in collect_dict:
        list_pf_context = collect_dict[question]
    else:
        list_pf_context = []

    list_pf_context.append(context)
    collect_dict.update({question: list_pf_context})

question_par_pairs = []
for query, context_list in collect_dict.items():
    # remove doubled question-paragraph pairs
    context_list = list(set(context_list))
    print(query + "  "  + str(len(context_list)))
    for context_i in context_list:
        question_par_pairs.append([query, context_i])

output_path = fpath.replace(".json", "_pairs.tsv")

# finally we have 505 query - context pairs
# What is the identification number of the company?  166
# Which commercial register is the company registered in?  124
# How much is the capital share?  215

"""
This script creates monoT5 input files for training,
Each line in the monoT5 input file follows the format:
    f'Query: {query} Document: {document} Relevant:\t{label}\n')
"""

from tqdm import tqdm

# input file should be a tsv with the following lines:
# NOTE: we should add negative examples !!!!!!!!!!!
# <query> \t <positive_document> \t <negative_document>"

with open(output_path, 'w') as fout_t5:
    for item_id, item in enumerate(tqdm(question_par_pairs)):
        # item = ['query', 'positive_context']
        query, positive_document = item[0], item[1]
        fout_t5.write(f'Query: {query} Document: {positive_document} Relevant:\ttrue\n')
        #fout_t5.write(f'Query: {query} Document: {negative_document} Relevant:\tfalse\n')
print('Done!')
