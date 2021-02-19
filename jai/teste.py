from mycelia_core import jAI
from pathlib import Path
import pandas as pd

jai = jAI("f0e4c4a6d35d4eb1a871093468fbf679")
df_path = Path("/home/paulo/Downloads/titanic_test.json")
df = pd.read_json(df_path)

name = "teste_dropna"
jai.setup(name, df, db_type="Supervised", label={"task": "metric_classification", "label_name": "Survived"})