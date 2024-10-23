import requests
from pathlib import Path
import pandas as pd
import json
cwd = Path.cwd()
print("test get ...\n\n")


with open(str(cwd) + "/deployment_demo/datasets/tests/dataset-test-1.json", 'r') as f:
        training_data = json.load(f)



data = {
        "data" : training_data,
        "path_mdl" : "./model/epoch=105-step=215180-validation_RMSE_LOSS=0.634.ckpt",
        "orient" : "rows"
}
response = requests.post(
        "http://48.211.241.89:8000/predict",
        json=data,
    ).json()



print(response["prediction"])

