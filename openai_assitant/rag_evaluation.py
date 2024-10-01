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