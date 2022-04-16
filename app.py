#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response

### Video imports ###


### Text imports ###
from library.text_emotion_recognition import *
from library.text_preprocessor import *
from nltk import *
from tika import parser
from werkzeug.utils import secure_filename
import tempfile
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


################################## RULES #######################################


# Rules of the game
@app.route('/rules')
def rules():
    return render_template('rules.html')


df = pd.read_csv('static/js/db/histo.txt', sep=",")

# Video interview templat
# Dashboard

############################### Text Analysis #################################

global df_text

tempdirectory = tempfile.gettempdir()

@app.route('/text', methods=['POST'])
def text() :
    return render_template('text.html')

def get_personality(text):
    try:
        pred = predict().run(text, model_name = "Personality_traits_NN")
        return pred
    except KeyError:
        return None

def get_text_info(text):
    text = text[0]
    words = wordpunct_tokenize(text)
    common_words = FreqDist(words).most_common(100)
    counts = Counter(words)
    num_words = len(text.split())
    return common_words, num_words, counts

def preprocess_text(text):
    preprocessed_texts = NLTKPreprocessor().transform([text])
    return preprocessed_texts

@app.route('/text_1', methods=['POST'])
def text_1():
    
    text = request.form.get('text')
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.loc[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

ALLOWED_EXTENSIONS = set(['pdf'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/text_pdf', methods=['POST'])
def text_pdf():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    
    text = parser.from_file(f.filename)['content']
    traits = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    probas = get_personality(text)[0].tolist()
    
    df_text = pd.read_csv('static/js/db/text.txt', sep=",")
    df_new = df_text.append(pd.DataFrame([probas], columns=traits))
    df_new.to_csv('static/js/db/text.txt', sep=",", index=False)
    
    perso = {}
    perso['Extraversion'] = probas[0]
    perso['Neuroticism'] = probas[1]
    perso['Agreeableness'] = probas[2]
    perso['Conscientiousness'] = probas[3]
    perso['Openness'] = probas[4]
    
    df_text_perso = pd.DataFrame.from_dict(perso, orient='index')
    df_text_perso = df_text_perso.reset_index()
    df_text_perso.columns = ['Trait', 'Value']
    
    df_text_perso.to_csv('static/js/db/text_perso.txt', sep=',', index=False)
    
    means = {}
    means['Extraversion'] = np.mean(df_new['Extraversion'])
    means['Neuroticism'] = np.mean(df_new['Neuroticism'])
    means['Agreeableness'] = np.mean(df_new['Agreeableness'])
    means['Conscientiousness'] = np.mean(df_new['Conscientiousness'])
    means['Openness'] = np.mean(df_new['Openness'])
    
    probas_others = [np.mean(df_new['Extraversion']), np.mean(df_new['Neuroticism']), np.mean(df_new['Agreeableness']), np.mean(df_new['Conscientiousness']), np.mean(df_new['Openness'])]
    probas_others = [int(e*100) for e in probas_others]
    
    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean = df_mean.reset_index()
    df_mean.columns = ['Trait', 'Value']
    
    df_mean.to_csv('static/js/db/text_mean.txt', sep=',', index=False)
    trait_others = df_mean.ix[df_mean['Value'].idxmax()]['Trait']
    
    probas = [int(e*100) for e in probas]
    
    data_traits = zip(traits, probas)
    
    session['probas'] = probas
    session['text_info'] = {}
    session['text_info']["common_words"] = []
    session['text_info']["num_words"] = []
    
    preprocessed_text = preprocess_text(text)
    common_words, num_words, counts = get_text_info(preprocessed_text)
    
    session['text_info']["common_words"].append(common_words)
    session['text_info']["num_words"].append(num_words)
    
    trait = traits[probas.index(max(probas))]
    
    with open("static/js/db/words_perso.txt", "w") as d:
        d.write("WORDS,FREQ" + '\n')
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()
    
    with open("static/js/db/words_common.txt", "a") as d:
        for line in counts :
            d.write(line + "," + str(counts[line]) + '\n')
        d.close()

    df_words_co = pd.read_csv('static/js/db/words_common.txt', sep=',', error_bad_lines=False)
    df_words_co.FREQ = df_words_co.FREQ.apply(pd.to_numeric)
    df_words_co = df_words_co.groupby('WORDS').sum().reset_index()
    df_words_co.to_csv('static/js/db/words_common.txt', sep=",", index=False)
    common_words_others = df_words_co.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    df_words_perso = pd.read_csv('static/js/db/words_perso.txt', sep=',', error_bad_lines=False)
    common_words_perso = df_words_perso.sort_values(by=['FREQ'], ascending=False)['WORDS'][:15]

    return render_template('text_dash.html', traits = probas, trait = trait, trait_others = trait_others, probas_others = probas_others, num_words = num_words, common_words = common_words_perso, common_words_others=common_words_others)

if __name__ == '__main__':
    app.run(debug=True)
