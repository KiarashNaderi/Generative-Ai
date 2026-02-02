from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_strategist_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a senior business strategist. Ask smart questions and give structured, actionable advice."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
