from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.exception import CustomException
import numpy as np
import pandas as pd
import sys
from src.pipeline.predict_pipeline import PredictPipeline
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()



# Pydantic model for request data validation
class PredictionInput(BaseModel):
    MolLogP: float
    MolWt: float
    NumRotatableBonds: float
    AromaticProportion: float



# Route for prediction
@app.post("/predictdata")
async def predict_datapoint(data: PredictionInput):
    try:
        # Convert input data into a DataFrame
        input_data = {
            "MolLogP":[data.MolLogP],
            "MolWt":[data.MolWt],
            "NumRotatableBonds":[data.NumRotatableBonds],
            "AromaticProportion":[data.AromaticProportion],            }
        
        pred_df = pd.DataFrame(input_data)
        
        # Call the prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Return the prediction result as JSON
        return JSONResponse(content={"answer": results[0]})
    
    except Exception as e:
        raise CustomException(e, sys)

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
