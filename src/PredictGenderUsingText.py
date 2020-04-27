import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Input Parameter e.g. '-i'
INPUT_PARAMETER = str(sys.argv[1])
#
# # Input directory path
INPUT_PATH = str(sys.argv[2])
#
# # Output parameter e.g. -o
OUTPUT_PARAMETER = str(sys.argv[3])
#
# # Output directory path
OUTPUT_PATH = str(sys.argv[4])
#
# # Checks if flags were set correctly
if INPUT_PARAMETER != "-i" \
        or OUTPUT_PARAMETER != "-o":
    sys.exit()

# The training directory which containing the profile.csv
profile_training_csv = ("C:/Users/micha/PycharmProjects/ML/data/training/profile/profile.csv")
profile_training_df = pd.read_csv(profile_training_csv)

# Read the file name without extension and its content (or transcripts)
# and put into an array
data_training = []
with os.scandir("C:/Users/micha/PycharmProjects/ML/data/training/text/") as entries:
    for entry in entries:
        # os.path.splitext(entry.name)[0] : filename without extension
        # open(INPUT_PATH + "/text/" + entry.name, "r").read(): read content of the file respective
        data = [os.path.splitext(entry.name)[0],
                open("C:/Users/micha/PycharmProjects/ML/data/training/text/" + entry.name, "r").read()]
        data_training.append(data)

data_testing = []
with os.scandir("C:/Users/micha/PycharmProjects/ML/data/public-test-data/text/") as entries:
    for entry in entries:
        # os.path.splitext(entry.name)[0] : filename without extension
        # open(INPUT_PATH + "/text/" + entry.name, "r").read(): read content of the file respective
        data = [os.path.splitext(entry.name)[0],
                open(INPUT_PATH + "/text/" + entry.name, "r").read()]
        data_testing.append(data)

# convert the array to dataframe
training_df = pd.DataFrame(data_training, columns=['userid', 'transcripts'])
testing_df = pd.DataFrame(data_testing, columns=['userid', 'transcripts'])

sorted(data_testing, key=lambda data: data[0])

# join two files together with the same userid
merging_two_files_traing = profile_training_df.join(training_df.set_index('userid'), on='userid')

# select the columns we need to analyze
data_training = merging_two_files_traing.loc[:, ['transcripts', 'gender']]

# Splitting the data into 8000 training instances and 1500 test instances
n = 8000
all_Ids = np.arange(len(data_training))

test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = data_training.loc[test_Ids, :]
data_train = data_training.loc[train_Ids, :]

# Training a Naive Bayes model
clf = MultinomialNB()
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['transcripts'])
y_train = data_train['gender']
clf.fit(X_train, y_train)

# Testing data
X_test = count_vect.transform(testing_df['transcripts'])

# Prediction the gender
prediction = clf.predict(X_test)
userIds = testing_df['userid']

j = 0
for i in userIds:
    if prediction[j] == 0.0:
        gender = "female"
    else:
        gender = "male"

    xml = "<user \n" \
          "   id=\"" + str(i) + "\"\n" \
            "age_group=\"-""\"\n" \
            "gender=\"" + str(gender) + "\"\n" \
            "extrovert=\"-""\"\n" \
            "neurotic=\"-""\"\n" \
            "agreeable=\"-""\"\n" \
            "conscientious=\"-""\"\n" \
            "open=\"-""\"\n" \
            "/>"
    filename = "{}/{}.xml".format("C:/Users/micha/PycharmProjects/ML/output/", i)
    with open(filename, 'w') as f:
        f.write(xml)
    f.close()
    j = j + 1

import pickle
# save the model to disk
filename = 'countVect_model.sav'
pickle.dump(clf, open("some_model.sav", "wb"))
filename1 = 'countVect_model.sav'
pickle.dump(count_vect, open(filename1, "wb"))