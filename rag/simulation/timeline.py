from datetime import datetime, timedelta
import pytz

class Timeline:
    def __init__(self, start_date="2025-01-01", speed=1.0):
        self.current_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.speed = speed
        self.timezone = pytz.timezone("UTC")
        
    def advance(self, days=1):
        self.current_date += timedelta(days=days * self.speed)
        return self.current_date
        
    def get_current_date(self):
        return self.current_date.strftime("%Y-%m-%d")
    
    def is_future(self, target_date):
        return datetime.strptime(target_date, "%Y-%m-%d") > self.current_date