# from flask import Flask, render_template, request, jsonify
# import anthropic
# import ast
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import plotly.graph_objects as go
# import plotly.utils
# from statistics import mean, variance
# import os
#
# app = Flask(__name__)
#
#
# def split_words_in_dict(word_dict):
#     for key, value in word_dict.items():
#         word_dict[key] = list(set(' '.join(word_dict[key]).split()))
#     return word_dict
#
#
# def replace_with_mask(text, word_dict, key_word):
#     text = text.replace(',', ' ,').replace('.', ' .')
#     words = text.split()
#     masked_words = []
#
#     for word in words:
#         if word.lower() in [v.lower() for v in word_dict.get(key_word, [])]:
#             masked_words.append('[MASK]')
#         else:
#             masked_words.append(word)
#
#     masked_text = ' '.join(masked_words).replace(' ,', ',').replace(' .', '.')
#     return masked_text
#
#
# def text_preprocess(text1, text2):
#     text1 = text1.replace(',', '').replace('.', '')
#     text2 = text2.replace(',', '').replace('.', '')
#     return text1, text2
#
#
# def get_jaccard_similarity(text1, text2):
#     text1, text2 = text_preprocess(text1, text2)
#     set1 = set(text1.split())
#     set2 = set(text2.split())
#     intersection = len(set1 & set2)
#     union = len(set1 | set2)
#     return intersection / union
#
#
# def get_cosine_similarity(text1, text2):
#     text1, text2 = text_preprocess(text1, text2)
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform([text1, text2])
#     return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).item()
#
#
# def get_levenshtein_distance(text1, text2):
#     text1, text2 = text_preprocess(text1, text2)
#     m, n = len(text1), len(text2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#
#     for i in range(m + 1):
#         dp[i][0] = i
#     for j in range(n + 1):
#         dp[0][j] = j
#
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if text1[i - 1] == text2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
#
#     return 1 - dp[m][n] / max(m, n)
#
#
# def create_horizontal_bar_plot(concepts, distances, name, xtitle, ytitle='Concepts'):
#     # Create a list of tuples (concept, distance) and sort it
#     sorted_data = sorted(zip(concepts, distances), key=lambda x: x[1], reverse=False)
#
#     # Unzip the sorted data
#     sorted_concepts, sorted_distances = zip(*sorted_data)
#
#     fig = go.Figure(go.Bar(
#         y=sorted_concepts,  # Use sorted concepts
#         x=sorted_distances,  # Use sorted distances
#         orientation='h',
#         marker_color='rgba(50, 171, 96, 0.6)',
#         text=[f'{d:.4f}' for d in sorted_distances],
#         textposition='inside'
#     ))
#     fig.update_layout(
#         title=name,
#         xaxis_title=xtitle,
#         yaxis_title=ytitle,
#         height=400,
#         margin=dict(l=0, r=0, t=30, b=0)
#     )
#     return plotly.utils.PlotlyJSONEncoder().encode(fig)
#
#
# def make_a_request(text):
#     message = client.messages.create(
#         model="claude-3-5-sonnet-20240620",
#         max_tokens=1000,
#         temperature=0,
#         system='Provide your answers without any explanation or additional text.',
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": text
#                     }
#                 ]
#             }
#         ]
#     )
#     return message.content[0].text
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/process', methods=['POST'])
# def process():
#     prompt = request.form['prompt']
#     scheme = request.form['scheme']
#
#     # Extract concepts
#     concepts = make_a_request(
#         f"Analyze the following text and identify up to five key concepts. Present your answer as a comma-separated list. Each concept should be expressed in one or two words or be a specific named entity. Focus on the most essential ideas, terms, or themes: {prompt}").split(
#         ', ')
#
#     # Relate concepts to words
#     concepts_with_words = split_words_in_dict(ast.literal_eval(make_a_request(
#         f"Analyze the text and identify words related to each key concept. Provide the output as a Python dictionary. Each key is one of the given concepts. The corresponding value is a list of words or phrases from the text that are closely related to or exemplify that concept. Key concepts: {concepts}. Text to analyze: {prompt}. Include only words and phrases that appear in the given text. If a concept has no related words in the text, use an empty list as its value. Ensure the dictionary is properly formatted and can be directly executed as Python code.")))
#
#     return jsonify({
#         'concepts': concepts,
#         'concepts_with_words': concepts_with_words
#     })
#
#
# @app.route('/visualize_concepts', methods=['POST'])
# def visualize_concepts():
#     prompt = request.form['prompt']
#     scheme = request.form['scheme']
#     concepts = request.form.getlist('concepts[]')
#     concepts_with_words = ast.literal_eval(request.form['concepts_with_words'])
#
#     # Calculate concept-wise distances
#     cosine_similarities = []
#     original_answer = make_a_request(f"{prompt}{scheme}")
#     for concept in concepts:
#         bootstrap_samples = []
#         masked_prompt = replace_with_mask(prompt, concepts_with_words, concept)
#         for _ in range(10):
#             # filled_prompt = make_a_request(f'Assess your degree of confidence in what to put in masked fields. If you think there is more than one appropriate choice, choose a generalizing form. Start filling in the words at the end. Use only common lexicon. Provide only the filled text without any additional explanations. Text to analyze: {masked_prompt}')
#             masked_answer = make_a_request(f"{masked_prompt}{scheme}")
#             bootstrap_samples.append(get_cosine_similarity(original_answer, masked_answer))
#         cosine_similarities.append(bootstrap_samples)
#     cosine_similarities_mean = [mean(sample) for sample in cosine_similarities]
#     cosine_similarities_variance = [variance(sample) for sample in cosine_similarities]
#
#     # Create horizontal bar plot
#     concept_mean = create_horizontal_bar_plot(concepts, cosine_similarities_mean, 'Concept impact on the output',
#                                               'Mean cosine similarity over a few shot')
#     concept_variance = create_horizontal_bar_plot(concepts, cosine_similarities_variance,
#                                                   'Estimated variability of the given output',
#                                                   'Cosine similarity variance over a few shot')
#
#     return jsonify({
#         'concept_mean': concept_mean,
#         'concept_variance': concept_variance
#     })
#
#
# @app.route('/mask', methods=['POST'])
# def mask():
#     prompt = request.form['prompt']
#     scheme = request.form['scheme']
#     concepts = request.form.getlist('concepts[]')
#     concepts_with_words = ast.literal_eval(request.form['concepts_with_words'])
#
#     # Original answer
#     original_answer = make_a_request(f"{prompt}{scheme}")
#
#     # Masked prompt
#     masked_prompt = prompt
#     for concept in concepts:
#         masked_prompt = replace_with_mask(masked_prompt, concepts_with_words, concept)
#
#     # Masked answer
#     filled_prompt = make_a_request(
#         f'Assess your degree of confidence in what to put in masked fields. If you think there is more than one appropriate choice, choose a generalizing form. Start filling in the words at the end. Use only common lexicon. Provide only the filled text without any additional explanations. Text to analyze: {masked_prompt}')
#     masked_answer = make_a_request(f"{filled_prompt}{scheme}")
#
#     # Calculate similarities
#     jaccard = get_jaccard_similarity(original_answer, masked_answer)
#     cosine = get_cosine_similarity(original_answer, masked_answer)
#     levenshtein = get_levenshtein_distance(original_answer, masked_answer)
#
#     return jsonify({
#         'original_prompt': prompt,
#         'masked_prompt': masked_prompt,
#         'original_answer': original_answer,
#         'masked_answer': masked_answer,
#         'jaccard': jaccard,
#         'cosine': cosine,
#         'levenshtein': levenshtein
#     })
#
#
# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 7000)))
#     # app.run()
