import pandas as pd
### reading training dataset
train = pd.read_csv("data/raw/training_set_VU_DM.csv")
### reading testing dataset
test = pd.read_csv("data/raw/test_set_VU_DM.csv")

train.head()

train.info()

