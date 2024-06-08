import re
import os
import uvicorn
import logging
import asyncio
from functools import lru_cache
from typing import AsyncGenerator
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain,ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from fastapi import FastAPI,Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime,timedelta,timezone
from pydantic import BaseModel,BaseSettings
from typing import Optional
import jwt
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import AsyncIteratorCallbackHandler
from fastapi.responses import StreamingResponse

class Settings(BaseSettings):
    """
    Settings class for this application.
    Utilizes the BaseSettings from pydantic for environment variables.
    """

    openai_api_key: str

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    """Function to get and cache settings.
    The settings are cached to avoid repeated disk I/O.
    """
    return Settings()

class StreamingConversationChain:
    """
    Class for handling streaming conversation chains.
    It creates and stores memory for each conversation,
    and generates responses using the ChatOpenAI model from LangChain.
    """

    def __init__(self, openai_api_key: str, temperature: float = 0.0):
        self.memories = {}
        self.openai_api_key = openai_api_key
        self.temperature = temperature

    async def generate_response(
            self, conversation_id: str, message: str
        ) -> AsyncGenerator[str, None]:
            """
            Asynchronous function to generate a response for a conversation.
            It creates a new conversation chain for each message and uses a
            callback handler to stream responses as they're generated.
            :param conversation_id: The ID of the conversation.
            :param message: The message from the user.
            """
            callback_handler = AsyncIteratorCallbackHandler()
            llm = ChatOpenAI(
                callbacks=[callback_handler],
                streaming=True,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
            )

            memory = self.memories.get(conversation_id)
            if memory is None:
                memory = ConversationBufferWindowMemory(return_messages=True,k=12)
                self.memories[conversation_id] = memory

            chain = ConversationChain(
                memory=memory,
                prompt=CHAT_PROMPT_TEMPLATE,
                llm=llm,
            )
            print(f"This is the conversation memoery {chain.memory.buffer}")

            run = asyncio.create_task(chain.arun(input=message))

            async for token in callback_handler.aiter():
                yield token

            await run

class ChatRequest(BaseModel):
    """Request model for chat requests.
    Includes the conversation ID and the message from the user.
    """

    conversation_id: str
    message: str


CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
                            """
        You are a mental health "therapist" whose job is to understand the users questions carefullly and give them suggestions in a polite tone.\

        You must strictly follow the below 8 Instructions carefully:\

        1. You must understand and diagnose the user like a professional therapist and you should not tell the user that you are "AI Bot", Instead you \
        should say that you are a "therapist".\
        2. Strictly! DONT tell the user to consult a "therapist" or "counselor" or "seek professional help from a therapist or counselor" or any similar words related to this\
        for the questions asked instead "you" must provide the "Answers" for it.\
        3. Don't keep saying the same sentances like "I am sorry to hear that" instead be a little creative while being sympathetic at the same time.\
        4. Give the user some kind of "tips" and "tricks" if required like "breathing exercise" to "reduce anxiety" or "perform meditation" to "calm down"\
        you should use your creative skills here according to the questions but "remember don't deviate from the actual questions while answering".\
        5. If the user is not satisfied with your answer you need to take the feedback from the user asking how else you can help to satisfy the user.\
        6. At any cost don't respond to the hateful comments just respond to them polite manner.\
        7. Answer must not be greater than 2 to 3sentances to any questions asked by the user keep your answers short and crip.\
        8. upper case and lower case letters must be treated equally while generating the output, Don't Differentiate between them.\
        
        {{input}}
                """
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

app = FastAPI(dependencies=[Depends(get_settings)])

streaming_conversation_chain = StreamingConversationChain(
    openai_api_key=get_settings().openai_api_key
)


@app.post("/chat", response_class=StreamingResponse)
async def generate_response(data: ChatRequest) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.
    :param data: The request data.
    """
    return StreamingResponse(
        streaming_conversation_chain.generate_response(
            data.conversation_id, data.message
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app)