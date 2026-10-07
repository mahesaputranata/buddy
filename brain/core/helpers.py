import datetime

def get_date():
    """Returns the current date in YYYY-MM-DD format."""
    return datetime.datetime.now().strftime("%Y-%m-%d")