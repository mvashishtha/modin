import pandas
import modin.pandas as pd
from io import StringIO

data = "a,b,c\n1,2,3\n4,5,6"
pd.read_csv(StringIO(data))
# pd.read_csv("small.csv")
