import openai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import config
import rag_processing
import pdf_extraction


# Load environment variables
# load_dotenv()
# OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY_ZHENG")

pdf_text = pdf_extraction.extract_text_from_pdf(config.PDF_FILE_PATH)
print(f"Extracted PDF text with length: {len(pdf_text)} characters")



test_question = "What is the check-in time for the Airbnb?"
response = rag_processing.generate_response_with_rag(test_question, pdf_text)

print(f"Q: {test_question}\nA: {response}")


test_question1 = "Who created the Haida Bracelet?"
response1 = rag_processing.generate_response_with_rag(test_question1, pdf_text)

print(f"Q: {test_question1}\nA: {response1}")


test_question2 = "Where's the Haida Bracelet?"
response2 = rag_processing.generate_response_with_rag(test_question2, pdf_text)

print(f"Q: {test_question2}\nA: {response2}")