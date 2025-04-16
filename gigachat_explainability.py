from gigachat import GigaChat
from dotenv import load_dotenv
import os

load_dotenv()


def gigachat_explain(logs):
    with GigaChat(credentials=os.getenv("SBER"), verify_ssl_certs=False, model='GigaChat') as giga:
        prompt = f"""
        Ты — интеллектуальная система для планирования грузоперевозок на складе. 
        Твоя задача — анализировать данные агентов и выбирать оптимальные дни для отправки грузов. 
    
        Концепция BDI:
        1. Срочность: 1-3 (1 — низкая, 3 — высокая).
        2. Полнота информации: 0-1 (0 — неполная, 1 — полная).
        3. Сложность задания: 1-3 (1 — низкая, 3 — высокая).
        4. Занятость мест: предыдущий/текущий/следующий день.
    
        Данные из логов: {logs}
    
        Инструкции:
        1. Восстанови модель BDI первого агента на основе данных.
        2. Обобщи поведение агента: 
           - Определи, были ли действия пассивными или активными.
           - Проанализируй динамику поведения за весь период.
        3. Выяви ключевые факторы, влияющие на решения агента.
    
        Требования:
        - Ответ должен быть в виде сплошного текста без форматирования.
        - Не используй маркеры, заголовки или жирный шрифт.
        - Пиши кратко, но информативно.
        """

        response = giga.chat(prompt)
        print(response.choices[0].message.content)
