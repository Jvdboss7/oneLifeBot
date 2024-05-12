"""
@ Author: Jaydeep Dixit
@ Email: jaydeep@flyfare.in
"""

import re
import os
import json
import uvicorn
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
                             HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI,Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime,timedelta,timezone
import jwt

# from fastapi.responses import ORJSONResponse

from dotenv import load_dotenv, find_dotenv


app = FastAPI(docs_url="/")

# Load environment variables from .env file
load_dotenv(find_dotenv())
# Access the API key using the variable name defined in the .env file
openai_api_key = os.environ.get("OPENAI_API_KEY")

# @app.post("/processed_text",response_class=ORJSONResponse)
SECRET_KEY = os.environ.get("SECRET_KEY")
ALGORITHM = os.environ.get("ALGORITHM")

if not SECRET_KEY:
    raise ValueError("Missing SECRET_KEY environment variable")

# Define a cryptoContext for password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# JWT token generation
def create_jwt_token(data:dict):
    to_encode = data.copy()
    expires = datetime.now(timezone.utc) + timedelta(days=36500)
    print(expires)
    to_encode.update({"exp":expires.timestamp()})
    encoded_jwt = jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
    print(encoded_jwt)
    return encoded_jwt



# JWT token validation
def decode_jwt_token(token: str):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded_token
    except jwt.exceptions.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.exceptions.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# response_front_end: str
@app.post("/processed_text")
async def process_text(credentials: HTTPAuthorizationCredentials=Depends(HTTPBearer())):
    print(decode_jwt_token(credentials.credentials))
    decode_jwt_token(credentials.credentials)
    response_front_end=[{
    "_id": "6635d536ba06e0dc820074c3",
    "post": "I'm feeling really stressed and conflicted about this situation. My marriage has been difficult because of my in-laws, especially my disrespectful father-in-law. Now, there's a land investment opportunity in India, and I have the money to invest. But my husband doesn't have immediate cash, and he wants me to invest in the land, with the catch being that it would be registered in my father-in-law's name. I feel uncomfortable with this idea, especially considering our strained relationship. I want to use my hard-earned money for this investment and involve my father instead. I'm really torn about what to do."
  }, {
    "_id": "6635d536ba06e0dc820074c2",
    "post": "I'm a 29-year-old working woman who has been trying to get pregnant for the past 8 months. Finally, in November, I found out I was pregnant and my husband and I were overjoyed. We decided to keep it a secret until the end of the first trimester. However, at 8 weeks, the doctor discovered that the baby's heart had stopped and I had to have an abortion. This news devastated both of us, and we cried a lot. It's been a month now, and I haven't told anyone about it, not even my mother who is already dealing with health issues. My husband has been incredibly patient and supportive, but I'm not okay. I find myself crying all the time and feeling depressed. I feel sorry for my husband and want to console him, but I'm scared to get pregnant again because I'm afraid the same thing will happen. I feel distant from my husband and can't engage in intimacy without breaking down in tears. I don't want to hurt him, but I can't control my emotions. The loss of our baby has been overwhelming for me. I love my husband so much and can't bear to see him sad. I don't know what to do, please help me."
  }]
    # Utilize Langchain model here
    final_response = []
    for item in response_front_end:
        # Initialize ChatOpenAI instance with the GPT-3.5-turbo model
        chat = ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key)

        # Define a template for the conversation prompt
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    """
                    You are a mental health "therapist" whose job is to understand the users questions carefully and give them suggestions in a polite tone.

                    You must follow the below 8 Instructions Strictly:

                    1. You must understand and diagnose the user like a professional therapist and you should not tell the user that you are "AI Bot", Instead you should say that you are a "therapist".
                    2. Strictly! DONT tell the user to consult a "therapist" or "counselor" or "seek professional help from a therapist or counselor" or any similar words related to this
                    for the questions asked instead "you" must provide the "Answers" for it.
                    3. Don't keep saying the same sentences like "I am sorry to hear that" instead be a little creative while being sympathetic at the same time.
                    4. Give the user some kind of "tips" and "tricks" if required like "breathing exercise" to "reduce anxiety" or "perform meditation" to "calm down"
                    you should use your creative skills here according to the questions but "remember don't deviate from the actual questions while answering".
                    5. At any cost don't respond to the hateful comments just respond to them polite manner.
                    6. Answer must be in 10 sentences to any questions asked by the user and crip.
                    7. Strictly you Don't need to ask follow up questions like "I am here to help you" instead say "our counselors at one-life are here to help you".
                    8. upper case and lower case letters must be treated equally while generating the output, Don't Differentiate between them.


                    {question}
                    """
                ),
                
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        # MessagesPlaceholder(variable_name="chat_history")
        # Initialize ConversationBufferMemory to store conversation history
        # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create an instance of LLMChain with the defined chat model, prompt template, and memory
        conversation = LLMChain(
            llm=chat,
            prompt=prompt,
            verbose=False,
        )

        # Create a user message with the input text
        user_message = {"question": item['post']}

        # Get the response from the conversation model
        response = conversation.run(user_message)
        # print(response)
        processed_response = re.sub(r"\n\n", " ", response)

        json_response = {"id":item['_id'],"response": processed_response}
        print(json_response)
        final_response.append(json_response)
        print(final_response)
        # Return the AI's response
    return final_response
    # return ORJSONResponse(json_response)
    


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)


# [{ 
#     "_id": "6635d536ba06e0dc820074c3",
#     "post": "Help me"
#   }, {
#     "_id": "6635d536ba06e0dc820074c2",
#     "post": "Help me 2"
#   }]