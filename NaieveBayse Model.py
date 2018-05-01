import csv
import nltk
import random
import re
from nltk.stem import PorterStemmer

Dataset = []

def ReadDataset():
	global Dataset
	with open("Dataset.csv", encoding='latin1') as csvfile: 
	     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
	     for row in spamreader:
	            Dataset.append([0 if row[0] == "0" else 2 if row[0] == "2" else 4, row[5]])

def PreProcessDataset():
	global Dataset
	
	AllTokens = ""

	random.shuffle(Dataset)
	Dataset = Dataset[:500000]
	
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
	MostCommonTokens = fdist.most_common(10000)
	MostCommonTokens = [token for (token, freq) in MostCommonTokens]
	Dataset = [({"token" : token}, category) for (category, tokens) in Dataset for token in tokens if token not in MostCommonTokens]

def TrainModel():
	global Dataset

	size = int(len(Dataset) * 0.1)
	train_set, test_set = Dataset[size:], Dataset[:size]
	classifier = nltk.NaiveBayesClassifier.train(train_set)
	print("Model Accuracy : ", nltk.classify.accuracy(classifier, test_set))
	print(classifier.show_most_informative_features(5))

ReadDataset()
PreProcessDataset()
TrainModel()