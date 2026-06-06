import openai
from langsmith import wrappers
from rag_eval import RagEvaluation


class LLMEvaluater:
    def __init__(self,rag_evaluation: RagEvaluation):
        self.rag_evaluation = rag_evaluation
        self.openai_client = wrappers.wrap_openai(openai.OpenAI())

    def correctness(self,inputs: dict, outputs: dict, reference_outputs: dict)->  bool:
        user_content=f"""You are grading the following question:
        {inputs['question']}
        Here is the reference answer:
        {reference_outputs['answer']}
         You are grading the following predicted llm answer:
        {outputs['response']}
        Respond with CORRECT or INCORRECT:
        Grade:
        """
        eval_instructions = "You are and epert professor specializing in grading students answers to questions."
        response = self.openai_client.chat.completions.create(
            model=self.rag_evaluation.settings.default_model,
            messages=[{"role": "user", "content": user_content},
                      {"role": "system", "content": eval_instructions}],
            temperature=self.rag_evaluation.settings.temperature,
        ).choices[0].message.content.strip()
        return response == "CORRECT"
    
    def concisions(self, outputs: dict , reference_outputs: dict) -> bool:
       return int(len(outputs['response'])) < 2*int(len(reference_outputs['answer']))