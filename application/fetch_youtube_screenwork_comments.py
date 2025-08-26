#!/usr/bin/env python
# coding: utf-8

# In[5]:



from path_setup import setup_project_root
root = setup_project_root()

import re, os, ast
from dotenv import load_dotenv
from etl_showcase.domain.models import StatusCode
from etl_showcase.config.youtube import (
    YOUTUBE_SPREADSHEET_ID,
    YOUTUBE_LOGS_SHEET_NAME,
    YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
)
from etl_showcase.domain.youtube_models import CommentSearchStatus, CommentSearchState
from etl_showcase.infrastructure.datasource.youtube_api import youtube_search_comments
from etl_showcase.infrastructure.datasource.google_sheets_api import (
    write_secret_json,
    delete_secret_json,
    create_google_sheet,
    is_sheet_exists,
    update_full_google_sheet,
    update_youtube_log_of_google_sheet,
    update_youtube_comment_search_state,
    get_youtube_comment_search_state,
    get_full_google_sheet 
)
from etl_showcase.infrastructure.utils.time_utils import get_now_time

# variables for search comments
load_dotenv(override=True)
screenwork_list = ast.literal_eval(os.getenv("YOUTUBE_SEARCH_COMMENTS_VIDEO_IDS_FOR_SCREENWORK"))

write_secret_json()
try:
    for screenwork in screenwork_list:
        screenwork_name = screenwork["screenwork_name"]
        youtube_video_ids = screenwork["video_ids"]
        
        youtube_comment_search_state = get_youtube_comment_search_state(
            screenwork=screenwork_name
        )
        if youtube_comment_search_state is None:
            youtube_comment_search_state = CommentSearchState(
                    screenwork=screenwork_name,
                    status=CommentSearchStatus.Pending,
                    rest_video_ids=youtube_video_ids,
                    next_page_token='',
                    log_time=get_now_time()
                )
        
        if youtube_comment_search_state.status == CommentSearchStatus.Completed:
            print(f"《{screenwork_name}》已記錄指定影評的所有留言。")
            break
        if youtube_comment_search_state.rest_video_ids == []:
            youtube_comment_search_state.rest_video_ids = youtube_video_ids[:]
        else:
            youtube_video_ids = youtube_comment_search_state.rest_video_ids[:]

        for video_id in youtube_video_ids:
            # search youtube comments (自動續抓全部模式)
            youtube_search_comments_result = youtube_search_comments(
                video_id=video_id,
                max_comment_count_per_page = 100,
                max_page = 1,
                current_next_page_token=youtube_comment_search_state.next_page_token,
                mode="auto"
            )
            youtube_comment_search_state.next_page_token = ''
        
            # update log and data in google sheets
            log_content = f'video id {video_id} Search youtube comments result: [{youtube_search_comments_result.status_code}] {youtube_search_comments_result.message}'
            print(log_content)
            update_youtube_log_of_google_sheet(
                function=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
                log_content=log_content
            )
        
            comments = youtube_search_comments_result.content
            if len(comments) == 0:
                log_content = f'There is no comment for screen work:{screenwork_name} and video id:{video_id}, delete video id'
                print(log_content)
                youtube_video_ids.remove(video_id)
                state.rest_video_ids = youtube_video_ids
            else:
                # get old data or just give titles
                if is_sheet_exists(spreadsheet_id=YOUTUBE_SPREADSHEET_ID, sheet_name=screenwork_name):
                    update_rows = get_full_google_sheet(
                        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                        sheet_name=screenwork_name
                    )
                    old_last_index = int(update_rows[-1][0])
                    print('get old data from sheet for screenwork')
                else:
                    update_rows = [['ID', 'Video ID', 'Parent ID', 'Level', 'Text', 'Like count', 'Publish datetime']]
                    old_last_index = 0
                    create_google_sheet(
                        spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                        sheet_name=screenwork_name
                    )
                    print('create sheet for screenwork')

                print('parse data to google sheet rows')
                # update google sheet
                for comment in comments:
                    update_rows.append([
                        comment.id + old_last_index,
                        comment.video_id,
                        comment.parent_id,
                        comment.level,
                        comment.textDisplay,
                        comment.likeCount,
                        comment.published_at,
                    ])
                update_sheet_result = update_full_google_sheet(
                    spreadsheet_id=YOUTUBE_SPREADSHEET_ID,
                    sheet_name=screenwork_name,
                    update_rows=update_rows
                )
                log_content = f'Update google sheet result: [{update_sheet_result.status_code}] {update_sheet_result.message}'
                print(log_content)
                if update_sheet_result.status_code != StatusCode.SUCCESS:
                    update_youtube_log_of_google_sheet(
                        function=YOUTUBE_SEARCH_COMMENTS_FUNCTION_NAME,
                        log_content=log_content
                    )
        
                # 根據狀況更新 state
                if youtube_search_comments_result.status_code == StatusCode.SUCCESS and update_sheet_result.status_code == StatusCode.SUCCESS:
                    youtube_comment_search_state.rest_video_ids.remove(video_id)
                elif youtube_search_comments_result.status_code != StatusCode.SUCCESS:
                    match = re.search(r'\[final next_page_token:(.*?)\]', youtube_search_comments_result.message)
                    if match:
                        youtube_comment_search_state.next_page_token = match.group(1)        

            # 更新 Comments_search_state
            if len(youtube_comment_search_state.rest_video_ids) == 0 and update_sheet_result.status_code == StatusCode.SUCCESS:
                youtube_comment_search_state.status = CommentSearchStatus.Completed
            else:
                youtube_comment_search_state.status = CommentSearchStatus.Processing
            youtube_comment_search_state.log_time = get_now_time()
            update_state_result = update_youtube_comment_search_state(youtube_comment_search_state)  
            print(f'Update state result: [{update_state_result.status_code}] {update_state_result.message}')
finally:
    delete_secret_json()


# In[ ]:



