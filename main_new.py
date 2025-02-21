import anthropic
import ast
from supplementary import *

prompt = "You are an intelligent system that assists in scheduling surgeries in a clinic. Your task is to find the optimal days for surgeries, considering multiple factors. You will work with observation entries from agents, analyze this data, and make decisions based on it. Your role is to coordinate agents who analyze data from all other agents and make optimal decisions for scheduling surgeries."
scheme = "Support your understanding with a short answer. The answer should be in two parts: 1) the main idea 2) a list of important details."

client = anthropic.Anthropic(
)

message_1 = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Make your answers clear and concise.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"
                            f"{scheme}"
                }
            ]
        }
    ]
)

answer = message_1.content[0].text

message_2 = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Make your answers clear and concise.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract up to 5 key concepts from the text below, give an answer in the form of a comma-separated list, each concept should consist of one or two words or be a named entity: {prompt}"
                }
            ]
        }
    ]
)

concepts = message_2.content[0].text.split(', ')
print(concepts)

message_3 = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Make your answers clear and concise.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Relate these key concepts to the words in the text. Provide the answer in the form of a python dictionary, with the key being the concept and the value being a list of words that are directly or conventionally related to the concept. Key concepts: {concepts}. Text: {prompt}."
                }
            ]
        }
    ]
)

concepts_with_words = split_words_in_dict(ast.literal_eval(message_3.content[0].text))
print(concepts_with_words)

mask_prompt = replace_with_mask(prompt, concepts_with_words, concepts[0])

message_4 = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0,
    system="Make your answers clear and concise.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{mask_prompt}"
                            f"{scheme}"
                }
            ]
        }
    ]
)

alternated_answer = message_4.content[0].text

print(answer)
print('\n')
print(alternated_answer)
print('\n')
print(get_jaccard_similarity(answer, alternated_answer), get_cosine_similarity(answer, alternated_answer),
      get_levenshtein_distance(answer, alternated_answer))
