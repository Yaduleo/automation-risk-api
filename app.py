from fastapi import FastAPI
import pydantic
import pandas as pd
import joblib

app = FastAPI(title="Automation Risk Predictor API")

# Load your frozen pipeline file
pipeline = joblib.load('automation_risk_pipeline.pkl')

class UserInput(pydantic.BaseModel):
    country: str
    region: str
    income_group: str
    quarter_label: str
    industry_sector: str
    ai_tool_adoption_pct: float
    ai_cited_layoff_announcements: float
    pct_sector_workforce_displaced: float
    pct_sector_workforce_new_roles_created: float
    net_workforce_change_pct: float

@app.get("/")
def home():
    return {"status": "API is live and running!"}

@app.post("/predict")
def predict_risk(data: UserInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = pipeline.predict(input_df)
    return {
        "industry_sector": data.industry_sector,
        "predicted_automation_risk_score": float(prediction[0])
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
