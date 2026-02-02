from llm import get_llm
from prompts import get_strategist_prompt
from memory import get_session_history

from langchain_core.runnables.history import RunnableWithMessageHistory

class BusinessStrategist:
    def __init__(self):
        self.llm = get_llm()
        self.prompt = get_strategist_prompt()

        chain = self.prompt | self.llm

        self.chain_with_memory = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def ask(self, user_input: str, session_id: str = "default") -> str:
        response = self.chain_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content
