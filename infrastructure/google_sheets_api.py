import base64, os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from film_tv_animation_insights.config.google_sheets import (
    GOOGLE_SHEET_SCOPES,
    GOOGLE_SHEET_SERVICE_ACCOUNT_FILE,
    GOOGLE_SHEET_JSON_B64
)
from film_tv_animation_insights.domain.models import BaseResponse, StatusCode
from film_tv_animation_insights.infrastructure.time_utils import get_now_time_string

def write_secret_json():
    with open(GOOGLE_SHEET_SERVICE_ACCOUNT_FILE, 'wb') as f:
        f.write(base64.b64decode(GOOGLE_SHEET_JSON_B64))

def delete_secret_json():
    if os.path.exists(GOOGLE_SHEET_SERVICE_ACCOUNT_FILE):
        os.remove(GOOGLE_SHEET_SERVICE_ACCOUNT_FILE)

def get_google_sheet_service():
    creds = Credentials.from_service_account_file(
        GOOGLE_SHEET_SERVICE_ACCOUNT_FILE, scopes=GOOGLE_SHEET_SCOPES)
    return build('sheets', 'v4', credentials=creds)

def is_sheet_exists(spreadsheet_id, sheet_name):
    service = get_google_sheet_service()
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = spreadsheet.get('sheets', [])
    for sheet in sheets:
        if sheet.get("properties", {}).get("title") == sheet_name:
            return True
    return False

def create_google_sheet(spreadsheet_id, sheet_name):
    service = get_google_sheet_service()
    request_body = {
        'requests': [{
            'addSheet': {
                'properties': {
                    'title': sheet_name
                }
            }
        }]
    }
    response = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=request_body
    ).execute()

def update_full_google_sheet(spreadsheet_id, sheet_name, update_rows):
    service = get_google_sheet_service()
    service.spreadsheets().values().clear(spreadsheetId=spreadsheet_id, range=sheet_name, body={}).execute()
    body = {'values': update_rows}
    try:
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=sheet_name,
            valueInputOption="RAW",
            body=body
        ).execute()
        return BaseResponse[str](
            status_code=StatusCode.SUCCESS,
            message=f'{result.get("updatedRows")} google sheet rows updated.',
            content=None
        )
    except HttpError as err:
        error_status = err.resp.status
        error_content = err.content.decode("utf-8") if hasattr(err.content, 'decode') else str(err.content)
        return BaseResponse[str](
            status_code=StatusCode.CALL_API_FAIL,
            message=f'HTTP {error_status}, {error_content}',
            content=None
        )
    except Exception as e:
        return BaseResponse[str](
            status_code=StatusCode.CALL_API_FAIL,
            message=f'Unexpected error: {e}',
            content=None
        )

def update_log_of_google_sheet(spreadsheet_id, sheet_name, search_keyword, update_content):
    service = get_google_sheet_service()
    sheet = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=sheet_name).execute()
    values = sheet.get('values', [])
    if not values:
        print('sheet not found.')
        return

    for i, row in enumerate(values):
        if len(row) > 0 and row[0] == search_keyword:
            update_body = {
                'values': [[update_content, get_now_time_string()]]  # 分別對應 B 和 C 欄的值
            }
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=f'{sheet_name}!B{i + 1}:C{i + 1}',  # 指定範圍 B~C 欄
                valueInputOption='RAW',
                body=update_body
            ).execute()
            break