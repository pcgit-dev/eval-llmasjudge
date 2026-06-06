from llmevaluater import LLMEvaluater
from rag_eval import RagEvaluation
from ragimpl import rag_bot, ragimpl
import os
from dotenv import load_dotenv
from langsmith import traceable
from bindeval import bindeval

load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_TRACING"]="true"
# Example usage
if __name__ == "__main__":
    rag_eval = RagEvaluation()
    llm_evaluater = LLMEvaluater(rag_eval)
    # rag_eval.prepare_data()
   ## rag_eval.prepare_rag_evaldata()
    # rag = ragimpl()
    # rag.rag_uploader()
    # query = "what is agents?"
    # results = rag_bot(rag, query, rag.retriever)
    # print(results)
    bindeval= bindeval(name="my_bindeval", llm_evaluater=llm_evaluater, rag_evaluation=rag_eval)
    bindeval.evaluater();

