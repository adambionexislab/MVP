import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os

SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
CREDS_FILE = "google_service_account.json"
SHEET_NAME = "WaterTreatmentLogs"

def get_sheet():
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPE)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1
    return sheet

def log_prediction(input_data, output_data, warnings):
    sheet = get_sheet()

    row = [
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
    ]

    sheet.append_row(row, value_input_option="RAW")
