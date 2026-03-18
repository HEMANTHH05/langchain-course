from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def main():
    info = """Rohit Gurunath Sharma (born 30 April 1987) is an Indian international cricketer and the former captain of the India national cricket team in all formats of the game.[3] He is a right-handed top-order batter. He represents Mumbai in domestic cricket and Mumbai Indians in the Indian Premier League. Sharma was a member of the teams that won the 2007 T20 World Cup, the 2013 ICC Champions Trophy and was the winning captain of the 2024 T20 World Cup and the 2025 ICC Champions Trophy.

    Sharma holds several batting records which include most runs in T20 Internationals, most sixes in international cricket,[a] most double centuries in ODI cricket (3), most centuries at Cricket World Cups (7) and joint most hundreds in Twenty20 Internationals (5).[5] He also holds the world record for the highest individual score (264) in a One Day International (ODI) and also holds the record for scoring most hundreds (five) in a single Cricket World Cup, for which he won the ICC Men's ODI Cricketer of the Year award in 2019.[6] He is the first and only captain to lead a team in all[b] ICC tournament finals.[7]

    He formerly captained Mumbai Indians and the team has won five Indian Premier League titles in 2013, 2015, 2017, 2019 and 2020 under him, making him the most successful captain in IPL history, sharing this record with MS Dhoni. He is also one of two players who have played in every edition of the T20 World Cup, from the inaugural edition in 2007 till 2024.[c] He is the only Indian player to win two T20 World Cups. He became the second Indian captain to win a T20 World Cup.

    He has received two national honours, the Arjuna Award in 2015 and the prestigious Khel Ratna Award in 2020 by the Government of India. Under his captaincy, India won the 2018 Asia Cup and the 2023 Asia Cup, the seventh and eighth time the country won the title, both in ODI format as well as the 2018 Nidahas Trophy, their second overall and first in T20I format. """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer ONLY using the information provided below. Do not use any outside knowledge. If the answer is not in the information, say 'I don't know'. Information: {info}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm

    chat_history = []

    print("Chat started! Ask anything about Rohit Sharma. Type 'exit' to quit.\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        response = chain.invoke({
            "info": info,
            "chat_history": chat_history,
            "question": question
        })

        print(f"AI: {response.content}\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response.content))


if __name__ == "__main__":
    main()
