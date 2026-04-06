import pandas as pd
from io import StringIO
import urllib.parse

csv_content = urllib.parse.unquote("data:text/csv;charset=utf-8,timestamp,user,department,role,event_type,details%0A2023-10-27T10:00:00,User_1,Engineering,Engineer,login,Successful%20login%20from%20known%20IP%0A2023-10-27T10:05:00,User_1,Engineering,Engineer,file_access,Accessed%20/confidential/source_code.py")
df = pd.read_csv(StringIO(csv_content))
print("COLUMNS with data URI:", df.columns.tolist())
