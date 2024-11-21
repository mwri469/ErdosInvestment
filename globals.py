"""
General python file for configurating global variables.
... This aims to make it more flexible when moving machines or within directories
"""

FILE_PATH = '../pyanomaly-master/output/merge.pickle'

# Define dates for train, val, test splits
# Not the second entry is exclusive
TRAINING_DATES = ('2010-01-01', '2019-01-01')
VALIDATION_DATES = ('2019-01-01', '2021-01-01')
OOS_TEST_DATES = ('2021-01-01', None)

# Setting up past, future dates
# Only relevent for LSTM models, keep 1 for standard models
PAST = 5
FUTURE = 1