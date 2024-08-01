import pandas as pd
from collections import defaultdict, Counter
from ast import literal_eval
import re

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    words_tokens = text.split()
    return words_tokens

def build_ngram_model(data, n):
    model = defaultdict(Counter)
    for sentence in data:
        sentence = preprocess_text(sentence)
        for i in range(len(sentence) - n):
            ngram = tuple(sentence[i:i+n])
            next_word = sentence[i+n]
            model[ngram][next_word] += 1
    return model

def predict_next_sentence(model, text, n):
    text = preprocess_text(text)
    if len(text) < n:
        return ""
    ngram = tuple(text[-n:])
    next_words = model[ngram]
    if not next_words:
        return ""
    total_count = sum(next_words.values())
    smoothed_next_words = {word: (count + 1) / (total_count + len(next_words)) for word, count in next_words.items()}
    predicted_next_word = max(smoothed_next_words, key=smoothed_next_words.get)
    #print(predicted_next_word)
    return predicted_next_word
    #return " ".join(text + [predicted_next_word])

def calculate_accuracy(predictions, actual_next_words):
    correct_predictions = sum([pred == actual for pred, actual in zip(predictions, actual_next_words)])
    accuracy = correct_predictions / len(actual_next_words) if len(actual_next_words) > 0 else 0
    return accuracy

def get_first_word_of_ending(row):
    try:
        endings = literal_eval(row['endings'])
        correct_ending = endings[row['label']]
        return preprocess_text(correct_ending)[0] if correct_ending else ""
    except (IndexError, ValueError, KeyError):
        return ""

def main():
    train_df = pd.read_csv('Ml_Project_Train.csv')
    validation_df = pd.read_csv('validation.csv')
    n = 2
    trained_model = build_ngram_model(train_df['ctx'], n)  
    
    test_file_path = input("Enter file name:\t")
    
    with open(test_file_path, 'r') as file:
        test_sentences = file.readlines()

    predictions = [predict_next_sentence(trained_model, sentence, n) for sentence in test_sentences]    
    for sentence, prediction in zip(test_sentences, predictions):
        print(f"Sentence: {sentence.strip()}\nPrediction of next word is : {prediction}\n")

if __name__ == "__main__":
    main()
