import gspread
import json
import os
from datetime import datetime
from google.oauth2.service_account import Credentials
from zoneinfo import ZoneInfo

SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SHEET_NAME = "WaterTreatmentLogs"

def get_sheet():
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)

    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).sheet1

def log_prediction(input_data, output_data, warnings):
    try:
        sheet = get_sheet()
        sheet.append_row([
            datetime.now(ZoneInfo("Europe/Rome")).isoformat(),
            input_data["Turbidity"],
            input_data["TOC"],
            input_data["COD"],
            input_data["pH"],
            output_data["PAC"],
            output_data["FLOCCULANT"],
            output_data["NAOH"]
        ])
    except Exception as e:
        print("Google Sheets logging failed:", e)

