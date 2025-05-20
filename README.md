# vectriozer
data = {
    "Do you offer delivery?": "Yes, we provide delivery service for all orders above 500 BDT.",
    "Where are you located?": "We are located at 123 Main Road, Dhaka.",
    "Can I return a product?": "You can return a product within 7 days of delivery.",
}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
questions = list(data.keys())
x = vectorizer.fit_transform(questions)
print(x)
 # the matching function
def answer (question: str) -> str:
    y = vectorizer.transform([question])
    similarities = cosine_similarity(y, x)[0]
    max_similarity = max(similarities)
    best_match_index = similarities.argmax()
    print(best_match_index)
    threshold = 0.5

    if max_similarity >= threshold:
        best_question = questions[best_match_index]
        return data[best_question]
    else:
        return "Sorry, I couldn’t find a suitable answer to your question."
print(answer("Do you offer delivery?"))
print(answer("Where are you located?"))
print(answer("i love you"))        
