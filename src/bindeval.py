import openai
from langsmith import wrappers
from llmevaluater import LLMEvaluater
from rag_eval import RagEvaluation
class bindeval:
    def __init__(self, name: str,llm_evaluater: LLMEvaluater,rag_evaluation: RagEvaluation):
        self.name = name
        self.llm_evaluater = llm_evaluater
        self.rag_evaluation = rag_evaluation
        self.instructions="Respond to the users question in a short, concise manner (one short sentence)."
        self.model = "gpt-4-turbo" 
        
    def llmresponsegenerator(self, question: str) -> str:
        
        return self.llm_evaluater.openai_client.chat.completions.create(
        model=self.model,
        temperature=0,
        messages=[
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content

    ### Call my_app for every datapoints
    def ls_target(self, inputs: str) -> dict:
        return {"response": self.llmresponsegenerator(inputs["question"])}
    
    def evaluater(self) :
        ## Run our evaluation
        self.rag_evaluation.client.evaluate(
        self.ls_target, ## Your AI system
        data="RAG_EVALS", ## The dataset we created in the prepare_data function
        evaluators=[self.llm_evaluater.correctness,self.llm_evaluater.concisions],
        experiment_prefix="openai-4o-turbo-chatbot"
)