import openai
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY_ZHENG")
client = openai.OpenAI(api_key=OPEN_AI_API_KEY) 

# --------------------------------------------------------------
# Extract Text from PDF
# --------------------------------------------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Extract text from your PDF file
pdf_text = extract_text_from_pdf("data/Haida_bracelet.pdf")
print(f"Extracted PDF text with length: {len(pdf_text)} characters")

# Split the text into smaller chunks
chunks = [pdf_text[i:i + 1000] for i in range(0, len(pdf_text), 1000)]  # Splitting into chunks of 1000 characters

# --------------------------------------------------------------
# Generate Embeddings for Each Chunk
# --------------------------------------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",  # Specify the model to use for embeddings
        input=text
    )
    return np.array(response.data[0].embedding)

# Generate embeddings for all chunks
chunk_embeddings = [(chunk, get_embedding(chunk)) for chunk in chunks if chunk.strip()]

# --------------------------------------------------------------
# Retrieve the Most Relevant Chunk Using Cosine Similarity
# --------------------------------------------------------------
def retrieve_data(question):
    # Generate an embedding for the question
    question_embedding = get_embedding(question)

    # Calculate cosine similarities between the question embedding and each chunk embedding
    similarities = [(chunk, cosine_similarity([question_embedding], [embedding])[0][0]) for chunk, embedding in chunk_embeddings]

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Retrieve the most relevant chunk
    retrieved_context = similarities[0][0]
    return retrieved_context

# --------------------------------------------------------------
# Generate a Response Using OpenAI with the Retrieved Context
# --------------------------------------------------------------
def generate_response_with_rag(question):
    """
    Generates a response using RAG by retrieving the most relevant context from the PDF and using OpenAI to generate an answer.
    """
    retrieved_context = retrieve_data(question)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions based only on the provided context."},
            {"role": "user", "content": f"Context: {retrieved_context}\n\nBased on this information, please answer the following question:\n\nQuestion: {question}"}
        ],
        # parameters setting
        temperature=1.0,
        max_tokens=100,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()

# --------------------------------------------------------------
# Test the RAG Implementation with a Sample Question
# --------------------------------------------------------------
test_question = "What is the check-in time for the Airbnb?"
response = generate_response_with_rag(test_question)

print(f"Q: {test_question}\nA: {response}")


test_question1 = "Who created the Haida Bracelet?"
response1 = generate_response_with_rag(test_question1)

print(f"Q: {test_question1}\nA: {response1}")


test_question2 = "Where's the Haida Bracelet?"
response2 = generate_response_with_rag(test_question2)

print(f"Q: {test_question2}\nA: {response2}")
