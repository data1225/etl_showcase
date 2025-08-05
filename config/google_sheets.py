import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_SHEET_SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
GOOGLE_SHEET_JSON_B64 = os.getenv('GOOGLE_SHEET_JSON_B64')
GOOGLE_SHEET_SERVICE_ACCOUNT_FILE = 'google-sheet-key.json'