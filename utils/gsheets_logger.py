import gspread
import json
import os
from datetime import datetime
from google.oauth2.service_account import Credentials

SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_NAME = "WaterTreatmentLogs"

def get_sheet():
    creds_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise RuntimeError("Google credentials not found in environment variables")

    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)

    client = gspread.authorize(creds)
    return client.open(SHEET_NAME).sheet1

def log_prediction(input_data, output_data, warnings):
    try:
        sheet = get_sheet()
        sheet.append_row([
            datetime.utcnow().isoformat(),
            input_data["Turbidity"],
            input_data["TOC"],
            input_data["COD"],
            input_data["pH"],
            output_data["PAC"],
            output_data["FLOCCULANT"],
            output_data["NAOH"],
            output_data["estimated_pH"],
            ", ".join(warnings)
        ])
    except Exception as e:
        # Never break the API because of logging
        print("Google Sheets logging failed:", e)
