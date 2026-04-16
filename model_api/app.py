import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import psycopg2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"status": "ok", "message": "Visitor Prediction API is running"}
blended_model = joblib.load("models/blended_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df_hist = pd.read_csv("test_clean_encoded_2.csv")
df_hist["date"] = pd.to_datetime(df_hist["date"])
festival_encoding = {
    "None": 0,
    "Diwali": 1,
    "Holi": 2,
    "Janmashtami": 3,
    "Makar Sakranti": 4,
    "MahaShivratri": 5,
    "Navratri": 6
}

DB_CONFIG = {
    "host": "localhost",
    "database": "crowd_db",
    "user": "postgres",
    "password": "1234"
}

class UserInput(BaseModel):
    date: str
    temperature: float
    precipitation: float
    festival: str
    temple_name: str
    day_of_week: int
    is_weekend: int
    festival_flag: int
    public_holiday: int
    #visitor_count: int

def feature_engineer(user_row):

    input_date = user_row["date"].iloc[0]
    hist_before_date = df_hist[df_hist["date"] < input_date]
    if len(hist_before_date) > 0:
        recent_visitor_count = hist_before_date.iloc[-1]["visitor_count"]
    else:
        recent_visitor_count = 45000
    user_row["visitor_count"] = recent_visitor_count
    fn = user_row["festival"].iloc[0]
    user_row["festival_name"] = festival_encoding.get(fn, 0)
    user_row["temp_avg_c"] = user_row["temperature"].iloc[0]
    user_row["precipitation_mm"] = user_row["precipitation"].iloc[0]
    user_row["temple_name"] = 0
    df_ext = pd.concat([df_hist, user_row], ignore_index=True)

    df_ext["month"] = df_ext["date"].dt.month
    df_ext["day_of_year"] = df_ext["date"].dt.dayofyear
    df_ext["year"] = df_ext["date"].dt.year
    df_ext["weekofyear"] = df_ext["date"].dt.isocalendar().week.astype(int)

    df_ext["sin_dow"] = np.sin(2*np.pi * df_ext["day_of_week"] / 7)
    df_ext["cos_dow"] = np.cos(2*np.pi * df_ext["day_of_week"] / 7)
    df_ext["sin_month"] = np.sin(2*np.pi * df_ext["month"] / 12)
    df_ext["cos_month"] = np.cos(2*np.pi * df_ext["month"] / 12)
    df_ext["sin_year"] = np.sin(2*np.pi * df_ext["day_of_year"] / 365.25)
    df_ext["cos_year"] = np.cos(2*np.pi * df_ext["day_of_year"] / 365.25)

    for lag in [1, 2, 3, 7, 14, 30, 60]:
        df_ext[f"lag_{lag}"] = df_ext["visitor_count"].shift(lag)

    for win in [7, 14, 30]:
        df_ext[f"roll_mean_{win}"] = df_ext["visitor_count"].shift(1).rolling(win).mean()
        df_ext[f"roll_std_{win}"] = df_ext["visitor_count"].shift(1).rolling(win).std()

    df_ext["ema_7"] = df_ext["visitor_count"].shift(1).ewm(span=7).mean()
    df_ext["ema_30"] = df_ext["visitor_count"].shift(1).ewm(span=30).mean()

    final = df_ext.iloc[-1:].copy()
    final = final.fillna(method="ffill").fillna(0)

    feature_cols = [
        'day_of_week', 'is_weekend', 'festival_flag', 'festival_name',
        'public_holiday', 'temp_avg_c', 'precipitation_mm', 'month',
        'day_of_year', 'year', 'weekofyear', 'sin_dow', 'cos_dow',
        'sin_month', 'cos_month', 'sin_year', 'cos_year', 'lag_1',
        'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30', 'lag_60',
        'roll_mean_7', 'roll_std_7', 'roll_mean_14', 'roll_std_14',
        'roll_mean_30', 'roll_std_30', 'ema_7', 'ema_30'
    ]

    X = final[feature_cols]

    X_scaled = X.copy()
    num_cols = X_scaled.select_dtypes(include=[np.number]).columns
    X_scaled[num_cols] = scaler.transform(X_scaled[num_cols])

    return X_scaled, final

def save_to_database(final_row):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        row = final_row.iloc[0]

        query = """
        INSERT INTO real_world_data (
            date, temperature, precipitation, festival,
            temple_name, day_of_week, is_weekend,
            festival_flag, public_holiday, visitor_count
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """

        values = (
            row["date"],
            row["temp_avg_c"],
            row["precipitation_mm"],
            row["festival_name"],
            0,
            row["day_of_week"],
            row["is_weekend"],
            row["festival_flag"],
            row["public_holiday"],
            row["visitor_count"]
        )

        cur.execute(query, values)
        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)

def blended_predict(models_dict, X):
    cat_model = models_dict["cat"]
    lgb_model = models_dict["lgb"]
    weight = models_dict["weight"]

    cat_pred = cat_model.predict(X)[0]
    lgb_pred = lgb_model.predict(X)[0]

    return cat_pred * weight + lgb_pred * (1 - weight)

@app.post("/predict")
def predict(data: UserInput):
    try:
        user_df = pd.DataFrame([data.dict()])
        user_df["date"] = pd.to_datetime(user_df["date"])

        X_scaled, final_row = feature_engineer(user_df)

        save_to_database(final_row)

        prediction = blended_predict(blended_model, X_scaled)

        return {
            "predicted_visitors": int(prediction),
            "status": "success",
            "date": data.date
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rows_hist": len(df_hist)
    }


if __name__ == "__main__":
    print(" Running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
