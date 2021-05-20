pdfs/
- used pdfs for the test files:
- 2018_BORUSSIA DORTMUND.pdf
- 2019_CROPENERGIES AG.pdf
- 2019_INSTONE REAL ESTATE GROUP O.N..pdf

gt_labels_for_pdfs/
- in this folder you find json files containing ground-truth labels for 3 questions for the 3 example pdfs
- question id 5: What is the identification number of the company?
- question id 6: Which commercial register is the company registered in?
- question id 17: How much is the capital share?

paragraph_candidates/
- contains json files , one file for each pdf
- the paragraph candidates collected question-wise identified by the question id
- for each question we have a candidate list
- candidate info:
  - rank: calculated score to rank the candidates
  - par_id: paragraph id constructed by p(age number)Xp(aragraph number)Y : p1p5 (as page 1 paragraph 5)
  - text: text of the paragraph

paragraphs/
- .txt files for each pdf, where the extracted paragraphs are separated with new lines
- pdf_dict_ABCD.json : contains the extracted paragraphs (and lines) from the pdf
  - a paragraph item has an id, a text, (and a list of its line ids)

predictions_with_deepset_bert/
- run the paragraph candidates on our QA pipeline and generate list of predictions for each questions

test_questions.json
- example questions used to create the paragraph candidates
- questions identified by an ID
- question_text field is the primary question
- question_templates: these question templates used to add multiple versions of the same questions,
in different documents one version of the question gives better results than another
For the question answering pipeline we create query - context pairs from the question_text, the question_templates
and the candidates, and choose the best predictions results
- question templates are used this way: (in this current version)
  e.g. What is the register number of #1_the company#? -* is converted to *- What is the register number of the company?