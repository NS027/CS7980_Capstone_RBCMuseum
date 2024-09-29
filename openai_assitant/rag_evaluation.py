import rag_processing
import pdf_extraction
import config
import pandas as pd
from datasets import Dataset 
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness

# https://namratanwani.medium.com/evaluate-rag-with-ragas-e1ad1aa99c2e
# https://github.com/explodinggradients/ragas

os.environ["OPENAI_API_KEY"] = config.OPEN_AI_API_KEY

pdf_text = pdf_extraction.extract_text_from_pdf(config.PDF_FILE_PATH)


# test_cases = [
#     {
#         "query": "What is the check-in time for the Airbnb?",
#         "ground_truth": "I don't know, sorry"
#     },
#     {
#         "query": "Who created the Haida Bracelet?",
#         "ground_truth": "The Haida people."
#     },
#     {
#         "query": "Where's the Haida Bracelet?",
#         "ground_truth": "It's in the RBCM"
#     }
# ]

queries = [
    "What is the check-in time for the Airbnb?",
    "Who created the Haida Bracelet?",
    "Where's the Haida Bracelet?"
]

ground_truths = [
    "I don't know, sorry",
    "The Haida people.",
    "It's in the RBCM"
]


# generated_response = []
# contexts = []

# for query in queries:
#     response = rag_processing.generate_response_with_rag(query, pdf_text)
   
#     generated_response.append(response)
#     # sources = result["source_documents"]
#     # contents = []
#     # for i in range(len(sources)):
#     #     contents.append(sources[i].page_content)
#     # contexts.append(contents)

# # apply ragas evaluation

# data_samples = {
#     'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
#     'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
#     'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
#     ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
#     'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
# }

def generate_data_samples(queries, ground_truths, pdf_text):

    data_samples = {
        'question': [],
        'answer': [],
        'retrieved_contexts': [],
        'ground_truth': ground_truths
    }

    for query in queries:
        answer = rag_processing.generate_response_with_rag(query, pdf_text)
        retrieved_context = rag_processing.retrieve_data(query, pdf_text)
        
        data_samples['question'].append(query)
        data_samples['answer'].append(answer)
        data_samples['retrieved_contexts'].append([retrieved_context])

    return data_samples


data_samples = generate_data_samples(queries, ground_truths, pdf_text)

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[faithfulness, answer_correctness])
score.to_pandas()
print(score)