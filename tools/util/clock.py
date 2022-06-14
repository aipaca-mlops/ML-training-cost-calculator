from datetime import datetime

import pytz


def now_time_str():
    return datetime.utcnow().replace(tzinfo=pytz.utc).isoformat().split(".")[0] + "UTC"
