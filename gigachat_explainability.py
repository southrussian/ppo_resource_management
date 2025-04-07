from gigachat import GigaChat
from dotenv import load_dotenv
import os

load_dotenv()

with GigaChat(credentials=os.getenv("SBER"),
              verify_ssl_certs=False, model='GigaChat') as giga:
    response = giga.chat("Какие отличия у алгоритма PPO от GPRO?")
    print(response.choices[0].message.content)
