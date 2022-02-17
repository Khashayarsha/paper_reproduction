#%%
from pathlib import Path
import json
import os 
import pandas as pd
# assigns a JSON string to a variable called jess
#jess = '{"name": "Jessica Wilkins", "hobbies": ["music", "watching TV", "hanging out with friends"]}'

# parses the data and assigns it to a variable called jess_dict
#jess_dict = json.loads(jess)

# Printed output: {"name": "Jessica Wilkins", "hobbies": ["music", "watching TV", "hanging out with friends"]}
#print(jess_dict)




data_folder = Path(r"C:\Users\XHK\Desktop\thesis_code\events_analysis\data\kaggle".replace("\\","/"))

file_to_open=data_folder / "events.csv"

df = pd.read_csv(file_to_open)