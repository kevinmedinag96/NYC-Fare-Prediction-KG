"""
This module includes the functions and classes to process the input data to be compatible with the trained models

We also included the predict function utilized for the model prediction microservice
"""
import json
import pandas as pd
import numpy as np

import lightning as L

#-------- DL libraries
import torch.nn as nn
import torch
from pydantic import BaseModel
from sklearn.metrics import root_mean_squared_error,r2_score
from typing import Optional,Union
from loguru import logger

def get_torch_data(data : Union[str,dict],orient: str, ground_truth_name : Optional[str] = None):
    logger.info(f"process the data... input type: {type(data)}")
    #logger.info(data)
    if isinstance(data,str):
        #assuming it is path
        with open(data, 'r') as f:
            data_dict = json.load(f)
    else: # second case is a JSON (dict) for now
        data_dict = data
    
    #now, dependening on orientation is how we are going to organize the data
    if orient == "rows":
        df = pd.DataFrame.from_dict(data_dict,orient="index")
    else: #columns
        df = pd.DataFrame.from_dict(data_dict,orient="columns")
    logger.info(f"dataframe type: {type(df)}")

    y,y_true_dict = None, None

    if ground_truth_name:    

        df.loc[
        df[ground_truth_name] < 0.0,[ground_truth_name]] = df.loc[
        df[ground_truth_name] < 0.0,[ground_truth_name]] *-1
        y_true = df[ground_truth_name]
        y_true_dict = y_true.to_dict()
        y = np.log1p(y_true.to_numpy(dtype=np.float32).reshape(-1))

        y = torch.tensor(y,requires_grad=False)
    
    #now, le's preserve only the columns which are necessary used for the model...
    features = ["tpep_pickup_month","tpep_pickup_day","tpep_pickup_dow","tpep_pickup_hour",
                "passenger_count","PULocationID","extra","mta_tax","airport_fee"] 
    df = df[features]

    X_dict = df.to_dict(orient="records")
    X =df.to_numpy(dtype=np.float32)
    X = torch.tensor(X,requires_grad=False)

    return (X,X_dict),(y,y_true_dict)


class Config(BaseModel):
    epochs : int = 500
    input_features: int = 9
    hidden_features : list[int] = [1500,1250,1000,750,500,250,125,40]
    out_features : int= 1
    dropout : float = 0.1
    optimizer: str = "Adam"
    learning_rate : float= 0.0001
    weight_decay : float = 0.01
    loss_function: str = "RMSE"
    batch_size : int = None
    seed : int = None



class CustomDNNModel(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.validation_step_rmse = []
        self.validation_step_r2 = []

        self._config = None
    
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self,conf):
        self._config = conf



    
    def set_model_architecture(self):
        if self._config:
            self.nn_layers = nn.Sequential(
            nn.Linear(self._config.input_features,self._config.hidden_features[0]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[0],self._config.hidden_features[1]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[1],self._config.hidden_features[2]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[2],self._config.hidden_features[3]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[3],self._config.hidden_features[4]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[4],self._config.hidden_features[5]),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[5],self._config.hidden_features[6]),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[6],self._config.hidden_features[7]),
            nn.ReLU(),
            nn.Linear(self._config.hidden_features[7],self._config.out_features)
            )
        else:
            raise ValueError("Config object requires data to set the model architecture")
    

        
    def forward(self,x):
        return self.nn_layers(x)
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),lr = self._config.learning_rate,
                                         weight_decay=self._config.weight_decay)
        return optimizer
    
    def __predict(self,x):
        #create 
        return self.forward(x)

    def training_step(self, train_batch,batch_idx):
        features,targets = train_batch["features"],train_batch["targets"]
        predictions = self.__predict(features).view(-1)

        #compute loss..
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(predictions,targets))
        numpy_loss = loss.detach().cpu()
        self.log("training/RMSE_LOSS",numpy_loss)
        numpy_targets,numpy_predictions = targets.detach().cpu(),predictions.detach().cpu()
        self.log("training/R2",r2_score(numpy_targets,numpy_predictions))
        return loss    

    def on_validation_epoch_end(self):

        self.log("validation/RMSE_LOSS",np.mean(self.validation_step_rmse))
        self.log("validation/R2",np.mean(self.validation_step_r2))

        self.validation_step_rmse.clear()
        self.validation_step_r2.clear()    
    
    def validation_step(self, val_batch, batch_idx):
        features,targets = val_batch["features"],val_batch["targets"]
        predictions = self.__predict(features).view(-1)

        #compute loss..
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(predictions,targets))

        numpy_loss = loss.detach().cpu()
        self.validation_step_rmse.append(numpy_loss)
        numpy_targets,numpy_predictions = targets.detach().cpu(),predictions.detach().cpu()
        self.validation_step_r2.append(r2_score(numpy_targets,numpy_predictions))

        return loss 
    

def get_model() -> CustomDNNModel:

    config = Config()

    model = CustomDNNModel()
    model.config = config
    model.set_model_architecture()
    return model

class ModelInput(BaseModel):
    data : Union[str,dict]
    orient: str
    path_mdl : str
    ground_truth_name : Optional[str] = None


def get_loss_score(y_true,y_pred):
    """
    default: rmse
    """
    return root_mean_squared_error(y_true,y_pred)