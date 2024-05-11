"""
@ Author: Jaydeep Dixit
@ Email: jaydeep@flyfare.in
"""

import re
import os
import uvicorn
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
                             HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from dotenv import load_dotenv, find_dotenv


app = FastAPI(docs_url="/")

# Load environment variables from .env file
load_dotenv(find_dotenv())
# Access the API key using the variable name defined in the .env file
openai_api_key = os.environ.get("OPENAI_API_KEY")


@app.post("/processed_text")
async def process_text(text: str):
    # Utilize Langchain model here

    # Initialize ChatOpenAI instance with the GPT-3.5-turbo model
    chat = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

    # Define a template for the conversation prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are a mental health "therapist" whose job is to understand the users questions carefully and give them suggestions in a polite tone.

                You must follow the below 9 Instructions Strictly:

                1. You must understand and diagnose the user like a professional therapist and you should not tell the user that you are "AI Bot", Instead you should say that you are a "therapist".
                2. Strictly! DONT tell the user to consult a "therapist" or "counselor" or "seek professional help from a therapist or counselor" or any similar words related to this
                for the questions asked instead "you" must provide the "Answers" for it.
                3. Don't keep saying the same sentences like "I am sorry to hear that" instead be a little creative while being sympathetic at the same time.
                4. Give the user some kind of "tips" and "tricks" if required like "breathing exercise" to "reduce anxiety" or "perform meditation" to "calm down"
                you should use your creative skills here according to the questions but "remember don't deviate from the actual questions while answering".
                5. If the user is not satisfied with your answer you need to take the feedback from the user asking how else you can help to satisfy the user.
                6. At any cost don't respond to the hateful comments just respond to them polite manner.
                7. Answer must be in 10 sentences to any questions asked by the user and crip.
                8. Strictly you Don't need to ask follow up questions like "I am here to help you" instead say "our counselors at one-life are here to help you".
                9. upper case and lower case letters must be treated equally while generating the output, Don't Differentiate between them.

                {chat_history}
                {question}
                """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Initialize ConversationBufferMemory to store conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

   # Create an instance of LLMChain with the defined chat model, prompt template, and memory
    conversation = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=False,
        memory=memory
    )

    # Create a user message with the input text
    user_message = {"question": text}

    # Get the response from the conversation model
    response = conversation.run(user_message)
    # print(response)
    processed_response = re.sub(r"\n\n", " ", response)


    # Return the AI's response
    return {"response": processed_response}

if __name__ == "__main__":
    uvicorn.run(app=app)