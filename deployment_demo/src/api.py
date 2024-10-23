from fastapi import FastAPI, HTTPException, Query,Path
import lightning
import torch
from src.backend import get_torch_data,get_model,ModelInput,get_loss_score
from pydantic import BaseModel
import json
from typing import Any,Optional
import numpy as np
from loguru import logger

app = FastAPI()

class PredictionResponse(BaseModel):
    input : list[dict]
    prediction : list[float]
    target : Optional[dict] = None
    score : Optional[float] = None

@app.get("/")
def server_intro() -> dict[str, str]:
    return {"message" : 
"Welcome to this NYC Taxi fare prediction service, please use the endpoint: /predict to obtain predictions based on your input."}

@app.post("/predict")
async def model_predict(input : ModelInput) -> PredictionResponse:
    #get data compatible with model prediction format
    

    tup_x,tup_y = get_torch_data(input.data,input.orient,
                                     input.ground_truth_name)
    logger.info("got the data... \n\n")
    #load model...

    model = get_model()
    checkpoint = torch.load(input.path_mdl,weights_only=True,map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["state_dict"])

    logger.info("loaded the model... \n\n")

    #turn off requires grad on each layer
    for params in model.parameters():
        params.requires_grad = False
 
    #predict...
    model.eval()
 
    y_predict= model(tup_x[0])
    y_predict = y_predict.cpu().numpy().reshape(-1).tolist()

    logger.info("prediction done... \n\n")

    response = {
        "input" : tup_x[1],
        "prediction" : y_predict
    }

    if tup_y[0]:
        response["target"] = tup_y[1]
        #compute rmse score
        score = get_loss_score(tup_y[0],y_predict)
        response["score"] = score    

    return response





if __name__ == "__main__":
    pass
    #path = "./datasets/dataset-test-general.json"
    #model_path = "./.neptune/NYCTaxiFare-DNN-1HL/TAXI-57/checkpoints/epoch=105-step=215180-validation_RMSE_LOSS=0.634.ckpt"

    #model_predict(path,model_path)