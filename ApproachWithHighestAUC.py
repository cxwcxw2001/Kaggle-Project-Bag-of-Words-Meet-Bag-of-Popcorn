import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
import numpy as np
nltk.download()  #Download text data sets, including stop words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

#Read the labeled training data
#quoting=3 tells Python to ignore doubles quotes, otherwise you may encounter
#errors trying to read the file
train=pd.read_csv("labeledTrainData.tsv",header=0,\
	delimiter="\t",quoting=3)

#Initialize the BeautifulSoup object on a single movie review
#example1=BeautifulSoup(train["review"][0])

#print the raw review and then the output of get_text(), for comparison
#print train['review'][0]
#print example1.get_text()

#Find anything that is not a lowercase letter or uppercase letter,
#and replace it with a space
# letters_only=re.sub("[^a-zA-Z]",        #The pattern to search for
					# " ",				#The pattern to replace it with	
					# example1.get_text())#The text to search

# lower_case=letters_only.lower()   #Convert to lower case
# words=lower_case.split()  		  #Split into words

def review_to_words(raw_review):
	#Function to convert a raw review to a string of words
	#1. Remove HTML
	review_text=BeautifulSoup(raw_review).get_text()
	#2. Remove non-letters
	letters_only=re.sub("[^a-zA-Z!?]"," ",review_text)
	#3. Convert to lower case, split into words
	words=letters_only.lower().split()
	#4. In Python, searching a set is much faster than searching
	#   a list, so convert the stop words to a set
	stops=set(stopwords.words("english"))
	#5. Remove stop words
	meaningful_words=[w for w in words if not w in stops]
	#6. Join the words back into one string separated by space,
	#   and return the result.
	return( " ".join(meaningful_words))
	
#Get the number of reviews
num_reviews=train["review"].size

#Add status updates
print "Cleaning and parsing the training set movie reviews...\n"

#Initialize an empty list to hold the clean reviews
clean_train_reviews=[]

#Loop over each review
for i in xrange(0,num_reviews):
	#If the index is divisible by 1000, print message
	if ((i+1)%1000==0):
		print "Review %d of %d\n" % (i+1,num_reviews)
	clean_train_reviews.append(review_to_words(train["review"][i]))
	
print "Creating the bag of words...\n"
#Initialize the "TfidfVectorizer" object,which is scikit-learn's 
#bag of words tool
vectorizer=TfidfVectorizer(ngram_range=(1,2),max_features=9500)

#fit_transform() does two functions: first, it fits the model and 
#learns the vocabulary; second, it transforms our training data into
#feature vectors. The input to fit_transform should be a list of strings.
train_data_features=vectorizer.fit_transform(clean_train_reviews)

#Numpy arrays are easy to work with,so convert the result to an array
train_data_features=train_data_features.toarray()

#Perform grid search to find the best value of parameter C
print "Performing grid search...\n"
C_range=2.0** np.arange(0,11)
param_grid=dict(C=C_range)
cv = StratifiedKFold(y=train["sentiment"], n_folds=5)
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=cv)
grid.fit(train_data_features, train["sentiment"])
print ("Scores of all parameters:\n",grid.grid_scores_)
print ("\nThe best parameter value is:\n ", grid.best_params_)
best_C=grid.best_params_

#Train the logistic regression classifier
print "\nTraining logistic regression classifier...\n"
lgReg=LogisticRegression(C=best_C)
lgClf=lgReg.fit(train_data_features,train["sentiment"])

#Read in the test data 
test=pd.read_csv("testData.tsv",header=0,delimiter="\t",\
				 quoting=3)
				 
num_reviews=len(test["review"])
clean_test_reviews=[]

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
	if ((i+1)%1000==0):
		print "Review %d of %d\n" % (i+1,num_reviews)
	clean_review=review_to_words(test["review"][i])
	clean_test_reviews.append(clean_review)
	
#Get a bag of words for the test set, and convert to numpy array
test_data_features=vectorizer.transform(clean_test_reviews)
test_data_features=test_data_features.toarray()

#Make predictions
print "Predicting...\n"
result=lgReg.predict_proba(test_data_features)[:,1]

output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
#Export the result to csv file
output.to_csv("Attempt2_result.csv",index=False,quoting=3)
