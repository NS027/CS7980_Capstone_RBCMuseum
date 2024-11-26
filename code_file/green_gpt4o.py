import os
from dotenv import load_dotenv
import tiktoken
import pandas as pd
from datasets import load_dataset
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.integrations.llama_index import evaluate


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Load environment variables
load_dotenv()

# Define constants
CARBON_EMISSION_PER_TOKEN = 0.0003  # Grams of CO2 per token

# Token counting helper function using tiktoken
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to run evaluation for a given LLM model
def run_evaluation(model_name, temperature, rag_embedding_model, evaluator_embedding_model):
    total_tokens = 0

    # Load the dataset from Hugging Face
    dataset = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")

    # Convert dataset to a list of Document objects
    documents = []
    for item in dataset:
        text_content = f"Question: {item['question']} Contexts: {item['documents_sentences']} Answer: {item['response']}"
        documents.append(Document(text=text_content))

    # Count tokens for each document
    for doc in documents:
        total_tokens += count_tokens(doc.text, model=model_name)

    # Create the vector index
    vector_index = VectorStoreIndex.from_documents(documents)

    # Create the query engine
    query_engine = vector_index.as_query_engine()

    # Set the LLMs for evaluation
    generator_llm = OpenAI(model=model_name, temperature=temperature)
    critic_llm = OpenAI(model=model_name, temperature=temperature)
    evaluator_llm = OpenAI(model=model_name, temperature=temperature)

    # Set the embeddings
    embeddings = OpenAIEmbedding(model=evaluator_embedding_model)

    # Create the testset generator
    generator = TestsetGenerator.from_llama_index(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )

    # Generate the testset
    testset = generator.generate_with_llamaindex_docs(
        documents,
        test_size=20,
        distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
    )

    # Convert the testset to a dataset object that can be accessed properly
    ds = testset.to_dataset()

    # Iterate over the dataset to count tokens in the questions
    for i in range(len(ds)):
        total_tokens += count_tokens(ds[i]['question'], model=model_name)

    # Define the evaluation metrics
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Perform the evaluation
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ds,
        llm=evaluator_llm,
        embeddings=embeddings,
    )

    # Estimate total carbon emissions
    total_carbon_emission = total_tokens * CARBON_EMISSION_PER_TOKEN

    # Display the result in a pandas DataFrame
    result_df = result.to_pandas()
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Return the results
    return total_tokens, total_carbon_emission, result_df

# Run evaluation for GPT-4
gpt4_tokens, gpt4_carbon, gpt4_results = run_evaluation(
    model_name="gpt-4o",
    temperature=0.1,
    rag_embedding_model="text-embedding-ada-002",
    evaluator_embedding_model="text-embedding-ada-002",
)

# Run evaluation for GPT-3.5
gpt35_tokens, gpt35_carbon, gpt35_results = run_evaluation(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    rag_embedding_model="text-embedding-ada-002",
    evaluator_embedding_model="text-embedding-ada-002",
)

# Print the comparison results
print("Comparison between GPT-4o and GPT-3.5:")
print("\n--- GPT-4o Results ---")
print(f"Total tokens used by GPT-4o: {gpt4_tokens}")
print(f"Estimated carbon emission for GPT-4o: {gpt4_carbon:.4f} grams CO2")

print("\n--- GPT-3.5 Results ---")
print(f"Total tokens used by GPT-3.5: {gpt35_tokens}")
print(f"Estimated carbon emission for GPT-3.5: {gpt35_carbon:.4f} grams CO2")

# Final comparison
gpt4_vs_gpt35_token_diff = gpt4_tokens - gpt35_tokens
gpt4_vs_gpt35_carbon_diff = gpt4_carbon - gpt35_carbon
print("\n--- Comparison Summary ---")
print(f"Difference in total tokens (GPT-4o vs GPT-3.5): {gpt4_vs_gpt35_token_diff}")
print(f"Difference in carbon emission (GPT-4o vs GPT-3.5): {gpt4_vs_gpt35_carbon_diff:.4f} grams CO2")
