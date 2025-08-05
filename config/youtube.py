import os
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
YOUTUBE_SPREADSHEET_ID = os.getenv('YOUTUBE_SPREADSHEET_ID')
YOUTUBE_SEARCH_VIDEOS_FUNCTION_NAME = 'Search_videos'
YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME = 'Search_comments'
YOUTUBE_LOGS_SHEET_NAME = 'Logs'