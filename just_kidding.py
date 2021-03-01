from jai.jai import Jai
import pandas as pd

j = Jai("a053f18dba4e4e6b87bfce5480a582f1", url="http://127.0.0.1:8000")

titanike = pd.read_json("/home/luisvictor/Downloads/titanic_test.json",
                        orient="records")
j.setup(name="titanike",
        data=titanike,
        db_type="Unsupervised",
        overwrite=True,
        frequency_seconds=1)
# j.wait_setup("titanike", frequency_seconds=1)

j.add_data(name="titanike", data=titanike)
j.wait_setup("titanike", frequency_seconds=1)