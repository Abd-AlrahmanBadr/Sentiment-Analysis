import csv
import nltk
import random
import re

Dataset = []
with open("Dataset.csv", encoding='latin1') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
     for row in spamreader:
            Dataset.append([0 if row[0] == "0" else 2 if row[0] == "2" else 4, row[5]])

# print(Dataset)

random.shuffle(Dataset)

Dataset = Dataset[:100000]

# print(Dataset[-1])
# print(Dataset[-2])
# print(Dataset[-3])
# print(Dataset[-4])
# print(Dataset[-5])
# print(Dataset[-6])

# Words = set()

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stopwords = nltk.corpus.stopwords.words('english')

for row in Dataset:
	row[1] = re.sub(r'^https?:\/\/.*[\r\n]*', '', row[1], flags=re.MULTILINE) # Removing URLs
	row[1] = re.sub(r'@\w+', '', row[1], flags=re.MULTILINE) # Removing accounts tag(@Ali)
	row[1] = re.sub(r'[^\w\s]','',row[1])
	row[1] = sent_tokenizer.tokenize(row[1])
	tokens = []
	for sentence in row[1]:
		words = nltk.word_tokenize(sentence)
		tokens += [token.lower() for token in words if token.lower() not in stopwords] # Removing stopwords, Converting every token to lowercase
	row[1] = tokens
	# for token in row[1]:
		# Words.add(token)

# print(Words)
# ll
# print(Dataset[-1])
# print(Dataset[-2])
# print(Dataset[-3])
# print(Dataset[-4])
# print(Dataset[-5])
# print(Dataset[-6])

# Most_Common_Tokens = 

Dataset = [({"token" : token}, category) for (category, tokens) in Dataset for token in tokens]

size = int(len(Dataset) * 0.1)
train_set, test_set = Dataset[size:], Dataset[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Model Accuracy : ", nltk.classify.accuracy(classifier, test_set))
print(classifier.show_most_informative_features(5))