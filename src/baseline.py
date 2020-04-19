''' Calculate the baseline information for the model results '''
import math
import sys

import pandas

__author__ = "Steven Tran"
__version__ = "4/10/2020"

# Input Parameter e.g. '-i'
INPUT_PARAMETER = str(sys.argv[1])

# Input directory path
INPUT_PATH = str(sys.argv[2])

if INPUT_PARAMETER != "-i":
    sys.exit()

PROFILE_CSV = "{}/{}".format(INPUT_PATH, "profile/profile.csv")
PROFILE_DF = pandas.read_csv(PROFILE_CSV)

# Initialize necessary variables for calculations.
AGES_COUNT = {"xx-24": 0, "25-34": 0, "35-49": 0, "50-xx": 0}
PERSONALITIES = {"openness": 0, "conscientiousness": 0,
                 "extroversion": 0, "agreeableness": 0,
                 "neuroticism": 0}

# Get age count and personality score count
for i in range(0, PROFILE_DF.shape[0]):
    profile = PROFILE_DF.loc[i]
    age = profile['age']
    if 0 <= age <= 24:
        AGES_COUNT["xx-24"] += 1
    elif 25 <= age <= 34:
        AGES_COUNT["25-34"] += 1
    elif 35 <= age <= 49:
        AGES_COUNT["35-49"] += 1
    elif 50 <= age <= math.inf:
        AGES_COUNT["50-xx"] += 1
    else:
        print("error getting age in this profile: ", profile)
    PERSONALITIES["openness"] += profile["ope"]
    PERSONALITIES["conscientiousness"] += profile["con"]
    PERSONALITIES["extroversion"] += profile["ext"]
    PERSONALITIES["agreeableness"] += profile["agr"]
    PERSONALITIES["neuroticism"] += profile["neu"]

# Get most frequent age group
MOST_FREQUENT_AGE = "50-xx"
for key in AGES_COUNT:
    if AGES_COUNT[key] > AGES_COUNT[MOST_FREQUENT_AGE]:
        MOST_FREQUENT_AGE = key

# Calculate average for each personality trait
for key in PERSONALITIES:
    PERSONALITIES[key] = round(PERSONALITIES[key] / PROFILE_DF.shape[0], 2)

# Prints most frequent age group
print("Most frequent age group: ", MOST_FREQUENT_AGE)
# Prints most frequent gender where 0 = male, 1 = female.
# Also prints the count of the mode of the most frequent gender
print("Most frequent gender: ", PROFILE_DF['gender'].value_counts().head(1).to_string())
# Prints the average personality trait scores
print("Average personality scores: ", PERSONALITIES)
