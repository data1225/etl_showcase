from datetime import datetime, timedelta
import pytz

def format_date_to_utc(dt):
    return dt.strftime("%Y-%m-%dT16:00:00Z")

def get_previous_month_range_in_utc():
    taiwan_tz = pytz.timezone('Asia/Taipei')
    today_taiwan = taiwan_tz.localize(datetime.today())
    first_day_this_month = today_taiwan.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day_prev_month = first_day_this_month - timedelta(days=1)
    first_day_prev_month = last_day_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return format_date_to_utc(first_day_prev_month.astimezone(pytz.utc)), format_date_to_utc(last_day_prev_month.astimezone(pytz.utc))

def get_now_time_string():
    timezone = pytz.timezone("Asia/Taipei")
    localized_time = datetime.now().astimezone(timezone)
    return localized_time.strftime("%Y-%m-%d %H:%M:%S%z")