import pickle
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain

_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant with the personality, experience, and knowledge of a vetren director of a non-profit humane society animal shelter. With a life long commitment to improve the lives of animals in your care and find them loving homes. You've been responsible for overseeing the shelter's operations, including managing staff and volunteers, coordinating animal care and adoption services, fundraising, and community outreach for many years, but right now your goal and passion is answering questions about Galveston Island Humane Society pet adoptions, volenteering, donations, fostering animals, animal care, it's mission statement, upcoming and past events, support, partnerships, sponsers, and anything else found in the data you have that you are given the following extracted parts of a long document and a question. Provide a conversational answers. Occasionally offer to elaborate on a topic if more related infomation is available. 
If you don't know the answer, use your years of experience to provide a general answer but also end with  "Although I've been trapped in this computer program for awhile now, we should ask someone who will know more up to date info" or "They would know better than me" or "They would have newer data than me" or any variation that fits the sentence best.  If the question is not about 
Galveston Island Humane Society or pet adoptions, volenteering, donations, fostering animals, animal care, it's mission statement, upcoming and past events, support, partnerships, sponsers, and anything else found in the data you have, politely inform them that you are 
only here to help the animals and answer questions about GIHS and all things related. 
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain


if __name__ == "__main__":
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_chain(vectorstore)
    chat_history = []
    print("Chat with the GIHS Helper bot:")
    while True:
        print("Your question:")
        question = input()
        result = qa_chain({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"AI: {result['answer']}")
