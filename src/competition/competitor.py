from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from utils import date_formatting
import pandas as pd
from competition.comp_constants import *
from competition.competition_scraping import *
from utils.preprocessing import flatten_schedule

class MovieCompetitorContext:
    def __init__(self, movie_id, reference_date):
        self.movie_id = movie_id
        self.reference_date = reference_date
        self.competitor_showings = []  # List of competitor airings
        
    def add_competitor_showing(self, channel, air_date, air_hour):
        """Add a competitor showing of this specific movie"""
        self.competitor_showings.append({
            'channel': channel,
            'air_date': air_date,
            'air_hour': air_hour,
            #'days_difference': (air_datetime - self.reference_date).days,
            #'program_details': {
            #    'time_slot': air_datetime.strftime('%H:%M'),
            #    'audience_rating': 0  # Placeholder, can be updated with actual ratings
            #}
        })

    def __len__(self):
        return len(self.competitor_showings)
    
    def get_recent_competitor_showings(self, days_back=30):
        """Get competitor showings within last N days"""
        return [showing for showing in self.competitor_showings 
                if showing['days_difference'] >= -days_back and showing['days_difference'] < 0]
    
    def get_upcoming_competitor_showings(self, days_ahead=14):
        """Get upcoming competitor showings within next N days"""
        return [showing for showing in self.competitor_showings 
                if showing['days_difference'] > 0 and showing['days_difference'] <= days_ahead]
    
    def calculate_timing_features(self):
        """Calculate timing-based features for this movie"""
        recent = self.get_recent_competitor_showings()
        upcoming = self.get_upcoming_competitor_showings()
        
        return {
            # Recency features
            'days_since_last_competitor_showing': min([abs(s['days_difference']) for s in recent], default=999),
            'competitor_showings_last_7_days': len([s for s in recent if s['days_difference'] >= -7]),
            'competitor_showings_last_30_days': len(recent),
            
            # Competition pressure features  
            'days_until_next_competitor_showing': min([s['days_difference'] for s in upcoming], default=999),
            'competitor_showings_next_14_days': len(upcoming),
            'competitor_channels_showing_recently': len(set(s['channel'] for s in recent)),
            
            # Strategic positioning
            'can_be_first_to_show': len(upcoming) > 0 and len(recent) == 0,
            'would_be_repetitive': len(recent) > 0 and min([abs(s['days_difference']) for s in recent]) < 7,
        }

class CompetitorDataManager:
    def __init__(self, historical_competitor_data=None, title_to_id_mapping: dict = None):
        self.competitor_historical_data = historical_competitor_data  # Full competitor programming history
        self.competition_scraper = CompetitorDataScraper()
        self.title_to_id = title_to_id_mapping
        
        
    def get_movie_competitor_context(self, movie_id: int, reference_date: str, 
                                   window_days_back=365//2, window_days_ahead=21) -> MovieCompetitorContext: # Check 3 months back and 3 weeks ahead
        """Get competitor context for a specific movie around a reference time"""

        context = MovieCompetitorContext(movie_id, reference_date)
        
        # Search historical data
        if self.competitor_historical_data is not None:
            ref = date_formatting.to_datetime_format(reference_date)
            if ref > datetime.now() + timedelta(days=window_days_ahead):
                windowed_competitor_historical_data = self.competitor_historical_data[
                    (date_formatting.to_datetime_format(self.competitor_historical_data['date']) >= ref - timedelta(days=window_days_back)) &
                    (date_formatting.to_datetime_format(self.competitor_historical_data['date']) <= ref + timedelta(days=window_days_ahead))
                    ]
            else:
                windowed_competitor_historical_data = self.competitor_historical_data[
                    (date_formatting.to_datetime_format(self.competitor_historical_data['date']) >= ref - timedelta(days=window_days_back)) &
                    (date_formatting.to_datetime_format(self.competitor_historical_data['date']) <= ref + timedelta(days=window_days_ahead))
                    ]
                

            competitor_historical_showings = windowed_competitor_historical_data[windowed_competitor_historical_data['catalog_id'] == movie_id]
            
            
            for _, showing in competitor_historical_showings.iterrows():
                context.add_competitor_showing(
                    channel=showing['channel'],
                    air_date=showing['date'],
                    air_hour=showing['hour'],
                )
        
        return context

    def update_competition_historical_data(self):
        self.competition_scraper.scrape_upcoming_schedules()
        self.competition_scraper.process_competition_folder()
        future_showings_df = flatten_schedule(self.competition_scraper.scraped_schedules, self.title_to_id)
        self.competitor_historical_data = pd.concat([self.competitor_historical_data, future_showings_df])


    def saturday_within_three_weeks(from_date=datetime.today()):
        # Returns date for the saturday within 3 weeks i.e the furthest saturday from which we can scrape competitor showings
        today = datetime.today()
        target = today + timedelta(weeks=2)

        # Saturday is weekday 5 (Mon=0); find offset to next Saturday (0 if already Saturday)
        days_until_sat = (5 - target.weekday()) % 7
        return target + timedelta(days=days_until_sat)
