import os

from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# initialize ragresults from json/dict
with open("examples/checking_inputs.json") as fp:
    rag_results = RAGResults.from_json(fp.read())

# set-up the evaluator
evaluator = RAGChecker(
    # extractor_name="bedrock/meta.llama3-1-70b-instruct-v1:0",
    # checker_name="bedrock/meta.llama3-1-70b-instruct-v1:0",
    extractor_name="openai/gpt-3.5-turbo",
    checker_name="openai/gpt-3.5-turbo",
    batch_size_extractor=32,
    batch_size_checker=32
)

# evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
evaluator.evaluate(rag_results, all_metrics)
print(rag_results)