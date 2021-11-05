import modin.pandas as pd
from io import StringIO, BytesIO

data = "a,b,c\n1,2,3\n4,5,6"
# df = pd.read_csv("small.csv")
df = pd.read_csv(StringIO(data))
# df = pd.read_csv(BytesIO(bytes(data, encoding="utf-8")))
# df = pd.read_csv("small.csv")
