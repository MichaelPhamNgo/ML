import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Input Parameter e.g. '-i'
INPUT_PARAMETER = str(sys.argv[1])

# Input directory path
INPUT_PATH = str(sys.argv[2])

# Output parameter e.g. -o
OUTPUT_PARAMETER = str(sys.argv[3])

# Output directory path
OUTPUT_PATH = str(sys.argv[4])

# Checks if flags were set correctly
if INPUT_PARAMETER != "-i" \
        or OUTPUT_PARAMETER != "-o":
    sys.exit()

# The directory which containing the profile.csv
PROFILE_CSV = ("C:/Users/micha/PycharmProjects/ML/data/training/profile/profile.csv")

# Read data in the profile.csv
PROFILE_DF = pd.read_csv(PROFILE_CSV)

# Read the file name without extension and its content (or transcripts)
# and put into an array
FILE_DATA = []
with os.scandir("C:/Users/micha/PycharmProjects/ML/data/training/text/") as entries:
    for entry in entries:
        # os.path.splitext(entry.name)[0] : filename without extension
        # open(INPUT_PATH + "/text/" + entry.name, "r").read(): read content of the file respective
        DATA = [os.path.splitext(entry.name)[0],
                open("C:/Users/micha/PycharmProjects/ML/data/training/text/" + entry.name, "r").read()]
        FILE_DATA.append(DATA)


# convert the array to dataframe
TEXT_DF = pd.DataFrame(FILE_DATA,columns=['userid','transcripts'])


# join two files together with the same userid
JOIN_FILES = PROFILE_DF.join(TEXT_DF.set_index('userid'), on='userid')


# select the columns we need to analyze
DATA = JOIN_FILES.loc[:,['transcripts', 'gender']]



# Splitting the data into 8000 training instances and 1500 test instances
n = 9000
all_Ids = np.arange(len(DATA))

test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = DATA.loc[test_Ids, :]
data_train = DATA.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['transcripts'])
print(X_train)
y_train = data_train['gender']
clf = NB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['transcripts'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
classes = [0.0, 1.0]
cnf_matrix = confusion_matrix(y_test,y_predicted, labels=classes)
print("Confusion matrix:")
print(cnf_matrix)