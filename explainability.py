from __future__ import annotations
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
import os

load_dotenv()


def yandex_explain(logs):
    sdk = YCloudML(
        folder_id=os.getenv('FOLDER_ID'),
        auth=os.getenv('AUTH'),
    )

    messages = [
        {
            "role": "system",
            "text": "Ты - интеллектуальная система, которая помогает планировать отправку грузов на складе. "
                    "Твоя задача - определить оптимальные дни для отправки с учетом множества факторов. "
                    "Ты будешь работать с данными наблюдений, полученными от агентов, анализировать эти данные "
                    "и принимать решения на их основе. "
                    "Твоя роль заключается в координации действий агентов, которые анализируют данные, поступающие от "
                    "всех других агентов, и принимают оптимальные решения для планирования отправок грузов.",
        },
        {
            "role": "assistant",
            "text": "В симуляции участвуют несколько агентов, каждый из которых наблюдает за различными аспектами "
                    "текущего состояния склада и пытается выбрать наилучший день для отправки груза. "
                    "Конечная цель агентов - оптимально распределить оправки по дням, чтобы максимально эффективно "
                    "использовать ресурсы склада и логистики и удовлетворить потребности клиентов."
        },
        {
            "role": "assistant",
            "text": "Прочитай и запомни эту концептуальную схему 'Belief-Desire-Intention' (BDI). "
                    "Срочность: оценка от 1 до 3, где 1 - низкая срочность, а 3 - высокая срочность. "
                    "Полнота информации: оценка от 0 до 1, где 0 - неполная информация, а 1 - полная. "
                    "Сложность задания: оценка от 1 до 3, где 1 - низкая сложность, а 3 - высокая. "
                    "Текущее положение в календаре за 7 дней: текущее положение агента в календаре за 7 дней. "
                    "Занятость мест за день до этого: занятость мест за предыдущий день. "
                    "Занятость мест на текущий день: занятость мест на текущий день. Занятость мест на следующий день: "
                    "занятость мест на следующий день. "
                    "(b) Структура желаний. Выберите наилучший день для отправки груза. "
                    "Учитывая все доступные данные. Уменьши конфликты и конкуренцию за места среди агентов. "
                    "Оптимально распредели отправки грузов для достижения максимальной эффективности работы склада. "
                    "(c) Структура намерений. 0 - перенести отправку на следующий день, "
                    "1 - перенести отправку на предыдущий день, "
                    "2 - ничего не делать."
        },
        {
            "role": "assistant",
            "text": f"Прочти и запомни этот фрагмент системных логов: [{logs}]"
        },
        {
            "role": "assistant",
            "text": "Цепочка рассуждений:"
                    "(а) Реконструируй модель BDI 1-го агента. "
                    "(б) Обобщи поведение 1-го агента. "
                    "Охарактеризуй действия агента (пассивные, активные), как вел себя агент на протяжении всех логов"
                    "определи мотивирующие факторы, которые заставляют агента действовать"
                    ""
                    "Не используй форматирование, выводи только текст, не допускай использование жирного текста",
        },
    ]

    result = (
        sdk.models.completions("yandexgpt").configure(temperature=0.1).run(messages)
    )

    for alternative in result:
        print(alternative.text)
