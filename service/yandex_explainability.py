from __future__ import annotations
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
import os
from grpc import StatusCode
from grpc.aio import AioRpcError

load_dotenv()


def yandex_explain(logs) -> str:
    sdk = YCloudML(
        folder_id=os.getenv('FOLDER_ID'),
        auth=os.getenv('AUTH'),
    )

    messages = [
        {
            "role": "system",
            "text": """
        Ты — интеллектуальная система для планирования грузоперевозок на складе. 
        Твоя задача — анализировать данные агентов и выбирать оптимальные дни для отправки грузов. 
    
        Концепция BDI (Belief-Desire-Intention):
        1. Срочность: 1-3 (1 — низкая, 3 — высокая).
        2. Полнота информации: 0-1 (0 — неполная, 1 — полная).
        3. Сложность задания: 1-3 (1 — низкая, 3 — высокая).
        4. Занятость мест: предыдущий/текущий/следующий день.
        """,
        },
        {
            "role": "system",
            "text": f"""
            Прочти данные из логов: {logs}
            """,
        },
        {
            "role": "system",
            "text": f"""
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
                """,
        }
    ]

    result = (
        sdk.models.completions("yandexgpt").configure(temperature=0.1).run(messages)
    )

    for alternative in result:
        print(alternative.text)
        return str(alternative.text)
