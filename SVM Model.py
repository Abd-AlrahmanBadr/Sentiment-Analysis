import csv
import nltk
import random
import re
from nltk.stem import PorterStemmer

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC

Dataset = []

def ReadDataset():
	global Dataset
	with open("Dataset.csv", encoding='latin1') as csvfile: 
	     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	     for row in spamreader:
	            Dataset.append(["0" if row[0] == "0" else "1", row[5]])

def FindFeatures(Tokens, WordsFeatures):
	Tokens = set(Tokens)
	Features = {}
	for token in WordsFeatures:
	    Features[token] = (token in Tokens)
	
	return Features

def PreProcessDataset():
	global Dataset
	
	AllTokens = ""

	random.shuffle(Dataset)
	Dataset = Dataset[:10000]
	
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	stopwords = nltk.corpus.stopwords.words('english')
	Stemmer = PorterStemmer()

	for row in Dataset:
		row[1] = re.sub(r'^https?:\/\/.*[\r\n]*', '', row[1], flags = re.MULTILINE) # Removing URLs
		row[1] = re.sub(r'@\w+', '', row[1], flags = re.MULTILINE) # Removing accounts tag(@Ali)
		row[1] = re.sub(r'[^\w\s]','',row[1]) # Removing Punctuation
		row[1] = sent_tokenizer.tokenize(row[1])
		tokens = []
		for sentence in row[1]:
			words = nltk.word_tokenize(sentence)
			tokens += [Stemmer.stem(token.lower()) for token in words if token.lower() not in stopwords] # Removing stopwords, Stemming and Converting every token to lowercase
		row[1] = tokens
		AllTokens += ' '.join(row[1])
	
	fdist = nltk.FreqDist(AllTokens.split(' '))

	WordFeatures = list(fdist.keys())[:int(len(fdist) * 0.2)]

	Dataset = [(FindFeatures(Tokens, WordFeatures), tag) for (Tokens, tag) in Dataset]

def TrainModel():
	global Dataset

	size = int(len(Dataset) * 0.1)
	X_train, X_test = Dataset[size:], Dataset[:size]

	SVC_clf = SklearnClassifier(LinearSVC())
	SVC_clf.train(X_train)
	print(nltk.classify.accuracy(SVC_clf, X_test))

ReadDataset()
PreProcessDataset()
TrainModel()