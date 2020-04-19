''' This module will produce the output xml files as described in the project description.'''
import sys
import pandas

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

PROFILE_CSV = "{}/{}".format(INPUT_PATH, "profile/profile.csv")
PROFILE_DF = pandas.read_csv(PROFILE_CSV)

for i in range(0, PROFILE_DF.shape[0]):
    USER_ID = PROFILE_DF.loc[i, 'userid'];
    xml = "<user \n" \
          "   id=\"" + str(USER_ID) + "\"\n" \
          "age_group=\"xx-24\"\n" \
          "gender=\"female\"\n" \
          "extrovert=\"3.49\"\n" \
          "neurotic=\"2.73\"\n" \
          "agreeable=\"3.58\"\n" \
          "conscientious=\"3.45\"\n" \
          "open=\"3.91\"\n" \
          "/>"
    filename = "{}/{}.xml".format(OUTPUT_PATH, PROFILE_DF.loc[i, 'userid'])
    with open(filename, 'w') as f:
        f.write(xml)
    f.close()