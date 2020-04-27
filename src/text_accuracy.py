import os
import sys
import pandas as pd
import pickle

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

model = pickle.load(open("some_model.sav","rb"))
countVect = pickle.load(open("countVect_model.sav","rb"))


data_testing = []
with os.scandir(INPUT_PATH + "/text/") as entries:
    for entry in entries:
        # os.path.splitext(entry.name)[0] : filename without extension
        # open(INPUT_PATH + "/text/" + entry.name, "r").read(): read content of the file respective
        data = [os.path.splitext(entry.name)[0],
                open(INPUT_PATH + "/text/" + entry.name, "r").read()]
        data_testing.append(data)

# convert the array to dataframe
testing_df = pd.DataFrame(data_testing, columns=['userid', 'transcripts'])

sorted(data_testing, key=lambda data: data[0])


testing_df = pd.DataFrame(data_testing, columns=['userid', 'transcripts'])

# Testing data
X_test = countVect.transform(testing_df['transcripts'])

prediction = model.predict(X_test)

print(prediction)
