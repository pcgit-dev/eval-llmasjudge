from __future__ import annotations
import logging
from langsmith import Client
from config import get_settings

class RagEvaluation:
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        self.log_config()
        self.client = Client(api_key=self.settings.langsmith_api_key)

    def _mask(self, secret: str | None) -> str:
        """Return a safe, partially-masked view of a secret for logging."""
        if not secret:
            return "<not set>"
        return f"{secret[:8]}...{secret[-4:]}" if len(secret) > 12 else "********"

    def log_config(self) -> None:
        """Log the loaded configuration (secrets masked) for sanity checking."""
        s = self.settings
        self.logger.info("OPENAI_API_KEY    : %s", self._mask(s.openai_api_key))
        self.logger.info("GROQ_API_KEY      : %s", self._mask(s.groq_api_key))
        self.logger.info("LANGSMITH_API_KEY : %s", self._mask(s.langsmith_api_key))
        self.logger.info("LANGSMITH_PROJECT : %s", s.langsmith_project)

    def prepare_data(self):
        """Prepare the dataset for RAG evaluation."""
        # Placeholder for the actual evaluation logic
        self.logger.info("Preparing data for RAG evaluation...")   
        dataset_name = "Chatbots Evaluation"
        dataset = self.client.create_dataset(dataset_name)
        self.client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": "What is LangChain?"},
                    "outputs": {"answer": "A framework for building LLM applications"},
                },
                {
                    "inputs": {"question": "What is LangSmith?"},
                    "outputs": {"answer": "A platform for observing and evaluating LLM applications"},
                },
                {
                    "inputs": {"question": "What is OpenAI?"},
                    "outputs": {"answer": "A company that creates Large Language Models"},
                },
                {
                    "inputs": {"question": "What is Google?"},
                    "outputs": {"answer": "A technology company known for search"},
                },
                {
                    "inputs": {"question": "What is Mistral?"},
                    "outputs": {"answer": "A company that creates Large Language Models"},
                }
            ]
        )


   
