import numpy as np
from openai import OpenAI
import config
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


client = OpenAI(api_key=config.OPEN_AI_API_KEY)


# Split the text into smaller chunks
def split_text_into_chunks(text, chunk_size=config.CHUNK_SIZE):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


# Generate embeddings for each chunk
def get_embedding(chunk):
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=chunk
    )
    return np.array(response.data[0].embedding)


# Generate embeddings for all chunks
def generate_embeddings(text):
    
    # Split the text into smaller chunks
    chunks = split_text_into_chunks(text)

    # Generate embeddings for all chunks
    chunk_embeddings = []
    for chunk in chunks:
        # Skip empty chunks
        if not chunk.strip():
            continue
        # Generate embeddings for the chunk
        embedding = get_embedding(chunk)
        chunk_embeddings.append((chunk, embedding))

    return chunk_embeddings


def retrieve_data(question, text):

    # call generate_embeddings function
    chunk_embeddings = generate_embeddings(text)

    # Generate an embedding for the question
    question_embedding = get_embedding(question)

    # Calculate cosine similarities between the question embedding and each chunk embedding
    similarities=[]
    for chunk, embedding in chunk_embeddings:
        similarity = cosine_similarity([question_embedding], [embedding])[0][0]
        similarities.append((chunk, similarity))
    
    # Sort by similarity in descending order
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            if similarities[i][1] < similarities[j][1]:
                similarities[i], similarities[j] = similarities[j], similarities[i]
    

    # Retrieve the most relevant chunk
    retrieved_context = similarities[0][0]
    return retrieved_context


def generate_response_with_rag(question, text):
# https://medium.com/@Doug-Creates/nightmares-and-client-chat-completions-create-29ad0acbe16a
    retrieved_context = retrieve_data(question, text)

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who answers questions based only on the provided context."},
            {"role": "user", "content": f"Context: {retrieved_context}\n\nBased on this information, please answer the following question:\n\nQuestion: {question}"}
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        top_p=config.TOP_P,
        # top_k=config.TOP_K,
        n=1 # Number of completions to generate
    )
    return response.choices[0].message.content.strip()
