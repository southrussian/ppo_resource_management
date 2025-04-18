from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_words_in_dict(word_dict):
    for key, value in word_dict.items():
        word_dict[key] = list(set(' '.join(word_dict[key]).split()))
    return word_dict


def replace_with_mask(text, word_dict, key_word):
    text = text.replace(',', ' ,').replace('.', ' .')
    words = text.split()
    masked_words = []

    for word in words:
        if word.lower() in [v.lower() for v in word_dict.get(key_word, [])]:
            masked_words.append('[MASK]')
        else:
            masked_words.append(word)

    masked_text = ' '.join(masked_words).replace(' ,', ',').replace(' .', '.')
    return masked_text


def get_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def get_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def get_levenshtein_distance(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return 1 - dp[m][n] / max(m, n)
