# Import libraries we'll need for this script
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


# Change numerical values back into words
# Define functions

# Cleaning up sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


#
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


#
def predict_class(sentence, model):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


#
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res


# Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="white", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("hello")
base.geometry("535x475")
base.resizable(width=FALSE, height=FALSE)
base.configure(bg="#1C1C1C")

# Create chat window
ChatLog = Text(base, bd=0, bg="#1C1C1C", height="8", width="50", font="Verdana", padx=10)

ChatLog.config(state=DISABLED)

# Bind scrollbar to chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set

# Create send button
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="SEND",
                    bd=0, bg="#f04c25", activebackground="#009cab",
                    fg='#ffffff', command=send)

# Create entry box
EntryBox = Text(base, bd=0, bg="#F3F3F3", width="29", height="5", font="Verdana", padx=10, pady=10)

# Place all components on the screen
scrollbar.place(x=510, y=10, height=400, width=15)
ChatLog.place(x=10, y=0, height=415, width=500)
EntryBox.place(x=10, y=415, height=50, width=415)
SendButton.place(x=425, y=415, height=50, width=100)

base.mainloop()