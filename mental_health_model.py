import pandas as pd

# Load JSON dataset
df = pd.read_json("student_mental_health.json")

print(df.head())
print(df.shape)
print(df.columns)
