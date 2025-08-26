import base64, os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from datetime import datetime
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_LOGS_SHEET_NAME,
    YOUTUBE_COMMENTS_SEARCH_STATE_SHEET_NAME,
)
from etl_showcase.config.google_sheets import (
    GOOGLE_SHEET_SCOPES,
    GOOGLE_SHEET_SERVICE_ACCOUNT_FILE,
    GOOGLE_SHEET_JSON_B64
)
from etl_showcase.domain.models import BaseResponse, StatusCode
from etl_showcase.domain.youtube_models import CommentSearchStatus, CommentSearchState
from etl_showcase.infrastructure.utils.time_utils import get_now_time_string

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

def get_full_google_sheet(spreadsheet_id, sheet_name):
    service = get_google_sheet_service()
    sheet = service.spreadsheets().values().get(
        spreadsheetId=YOUTUBE_SPREADSHEET_ID,
        range=sheet_name
    ).execute()

    values = sheet.get('values', [])
    if not values:
        print(f'Sheet "{sheet_name}" is empty or not found.')
        return None
    
    return values

def get_youtube_comment_search_state(screenwork):
    service = get_google_sheet_service()
    sheet = service.spreadsheets().values().get(
        spreadsheetId=YOUTUBE_SPREADSHEET_ID,
        range=YOUTUBE_COMMENTS_SEARCH_STATE_SHEET_NAME
    ).execute()
    
    values = sheet.get('values', [])
    if not values:
        print('sheet not found.')
        return None

    for row in values:
        if len(row) > 1 and row[0] == screenwork:
            try:
                # Map columns to dataclass fields
                status = CommentSearchStatus(row[1])
                rest_video_ids = row[2].split(',') if row[2] else []
                next_page_token = row[3] if row[3] else None
                log_time = datetime.fromisoformat(row[4])

                return CommentSearchState(
                    screenwork=row[0],
                    status=status,
                    rest_video_ids=rest_video_ids,
                    next_page_token=next_page_token,
                    log_time=log_time
                )
            except (ValueError, IndexError) as e:
                # Error parsing data for screenwork 'test': '' is not a valid CommentSearchStatus
                print(status)
                print(f"Error parsing data for screenwork '{screenwork}': {e}")
                return None
    
    print(f'Screenwork "{screenwork}" not found in sheet "{YOUTUBE_COMMENTS_SEARCH_STATE_SHEET_NAME}".')
    return None

def update_youtube_comment_search_state(state: CommentSearchState) -> BaseResponse:
    values = get_full_google_sheet(
            spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
            sheet_name=YOUTUBE_COMMENTS_SEARCH_STATE_SHEET_NAME
        )
    header_row = values[0:1] if values else []
    data_rows = values[1:] if values else []

    # Prepare the new row data from the dataclass
    new_row_data = [
        state.screenwork,
        state.status.value,
        ",".join(state.rest_video_ids) if state.rest_video_ids else '',
        state.next_page_token if state.next_page_token else '',
        state.log_time.isoformat()
    ]

    updated = False
    # Find the row to update
    for i, row in enumerate(data_rows):
        if len(row) > 0 and row[0] == state.screenwork:
            data_rows[i] = new_row_data
            updated = True
            break

    # If not found, add a new row
    if not updated:
        data_rows.append(new_row_data)

    # Combine headers and updated data rows
    updated_data = header_row + data_rows

    # Write the entire updated list of rows back to the sheet
    update_result = update_full_google_sheet(
        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
        sheet_name=YOUTUBE_COMMENTS_SEARCH_STATE_SHEET_NAME,
        update_rows=updated_data
    )
    return update_result

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

def update_youtube_log_of_google_sheet(function, log_content):
    service = get_google_sheet_service()
    sheet = service.spreadsheets().values().get(spreadsheetId=YOUTUBE_SPREADSHEET_ID, range=YOUTUBE_LOGS_SHEET_NAME).execute()
    values = sheet.get('values', [])
    if not values:
        print('sheet not found.')
        return

    for i, row in enumerate(values):
        if len(row) > 0 and row[0] == function:
            update_body = {
                'values': [[log_content, get_now_time_string()]]  # 分別對應 B 和 C 欄的值
            }
            service.spreadsheets().values().update(
                spreadsheetId=YOUTUBE_SPREADSHEET_ID,
                range=f'{YOUTUBE_LOGS_SHEET_NAME}!B{i + 1}:C{i + 1}',  # 指定範圍 B~C 欄
                valueInputOption='RAW',
                body=update_body
            ).execute()
            break