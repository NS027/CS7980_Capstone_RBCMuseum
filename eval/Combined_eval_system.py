import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.integrations.llama_index import evaluate

from dotenv import load_dotenv

load_dotenv()
# Get the API key from the environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# For different LLM models
RAG_LLM = "gpt-3.5-turbo"
GENERATOR_LLM = "gpt-3.5-turbo"
CRITIC_LLM = "gpt-3.5-turbo"
EVALUATOR_LLM = "gpt-3.5-turbo"

# For different temperatures
TEMPERATURE = 0.1

# For different embeddings
RAG_EMBEDDING = "text-embedding-ada-002"
EVALUATOR_EMBEDDING = "text-embedding-ada-002"

# For different chunk
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20


# Set the LLM, Embedding, and Text Splitter
Settings.llm = OpenAI(model=RAG_LLM, temperature=TEMPERATURE)
Settings.embed_model = OpenAIEmbedding(model=RAG_EMBEDDING)
Settings.text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# Load the documents
documents = SimpleDirectoryReader("data/Trump/").load_data()
# Create the vector index
vector_index = VectorStoreIndex.from_documents(documents)
# Create the query engine
query_engine = vector_index.as_query_engine()

# For the evaluation
# Set the LLMs
generator_llm = OpenAI(model=GENERATOR_LLM)
critic_llm = OpenAI(model=CRITIC_LLM)
evaluator_llm = OpenAI(model=EVALUATOR_LLM)
# Set the embeddings
embeddings = OpenAIEmbedding(model=EVALUATOR_EMBEDDING)

# Create the testset generator
generator = TestsetGenerator.from_llama_index(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings,
)

# Generate the testset
testset = generator.generate_with_llamaindex_docs(
    documents,
    test_size=10,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
)

# Define the evaluation metrics
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# convert to HF dataset
ds = testset.to_dataset()

result = evaluate(
    query_engine=query_engine,
    metrics=metrics,
    dataset=ds,
    llm=OpenAI(model=EVALUATOR_LLM),
    embeddings=OpenAIEmbedding(model=EVALUATOR_EMBEDDING),
)

# Convert the result to pandas
result_df = result.to_pandas()

print(result_df)
