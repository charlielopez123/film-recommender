import requests
from requests import Session
from pathlib import Path
from datetime import date, timedelta, datetime
import zoneinfo
import re

from lxml import etree
from competition.comp_constants import *
from tqdm import tqdm
from typing import Optional, Tuple, Union

def make_session_for(channel: str):
    sess = requests.Session()
    site_cfg = SITES[channel]
    if site_cfg['default_headers'] is not None:
        sess.headers.update(site_cfg['default_headers'])
    return sess

class CompetitorDataScraper:
    """For online training - scrape competitor programming schedules"""
    
    def __init__(self):
        self.SESSIONS = [make_session_for(channel) for channel in CHANNELS]
        self.competitor_channels = CHANNELS
        self.scraped_schedules = {}
        self.sessions = {channel: make_session_for(channel) for channel in self.competitor_channels}
        self.scraped_ids = []
        
        
    def scrape_upcoming_schedules(self, days_ahead=21):
        """Fetch exactly one XML per TV-week per channel, extract films."""
        #start = date.today()
        #end   = start + timedelta(days=days_ahead)

        # 1) Collect all TV‑week starts (Saturdays) covering [start…end]
        tv_week_starts = self._available_tv_week_starts(days_ahead=days_ahead)
        print(tv_week_starts)

        for channel in self.sessions.keys():
            self.scraped_schedules[channel] = {}
            for week_sat in tqdm(sorted(tv_week_starts), desc= f'Weeks for {channel}', leave = True):
                week_mon = week_sat + timedelta(days=2)
                iso_year, iso_week, _ = week_mon.isocalendar()
                params = {'week_sat': week_sat, 'iso_year': iso_year, 'iso_week': iso_week}
                try:
                    if f'week_{iso_week}_{channel}_{iso_year}' in self.scraped_ids:
                        continue
                    xml_path = self.download_xml(channel, **params)
                    
                    
                except Exception as e:
                    print(f"Failed to scrape {channel}: {e}")
                    #print(xml_path)
                self.scraped_ids.append(f'week_{iso_week}_{channel}_{iso_year}')
        
    

    def get_movie_future_showings(self, movie_id, reference_date):
        """Get future showings of a specific movie across all competitors"""
        future_showings = []
        
        for channel, schedule in self.scraped_schedules.items():
            for program in schedule:
                if self._match_movie(program, movie_id):  # Movie matching logic
                    future_showings.append({
                        'channel': channel,
                        'air_datetime': program['air_datetime'],
                        'details': program
                    })
        
        return future_showings
    
    def _TF1_download_xml(self, session: Session, **params):

        BASE_HTML = SITES['TF1']['base_html']
        BASE_XML = SITES['TF1']['base_xml']

        html_url = BASE_HTML.format(date = str(params['week_sat']))
        xml_url  = BASE_XML .format(date = str(params['week_sat']))
        
        out_dir = Path(f'competition_data')
        out_dir.mkdir(parents=True, exist_ok=True)

        session.get(html_url, timeout=10).raise_for_status() # sets TF1 cookies
        resp = session.get(xml_url, headers={'Accept': 'application/xml'}, timeout=10)
        resp.raise_for_status()

        str_year = str(params['iso_year'])
        str_week = str(params['iso_week'])

        out_path = out_dir / f'TF1_{str_year}_{str_week}.xml'
        with open(out_path, 'wb') as f:
            f.write(resp.content) 

        return out_path # Return path to xml file for parsing


    def _M6_download_xml(self, session: Session, **params):
        str_year = str(params['iso_year'])
        str_week = str(params['iso_week'])

        base_xml = SITES['M6']['base_xml'].format(year = str_year, week = str_week)
        resp = session.get(base_xml, timeout = 10)
        resp.raise_for_status()

        out_dir = Path(f'competition_data')
        
        out_path = out_dir / f'M6_{str_year}_{str_week}.xml'
        with open(out_path, 'wb') as f:
            f.write(resp.content)
        
        return out_path # Return path to xml file for parsing


    def _parse_cinema_from_tf1(self, xml_path):
        """
        Stream-parse the TF1 weekly XML feed and yield all 'Cinéma' emissions
        with their broadcast date, time, and title.
        """
        context = etree.iterparse(
            xml_path,
            events=('start', 'end'),
            recover=True,
            remove_blank_text=True
        )

        current_date = None

        for event, elem in context:
            # When we hit the start of a new day, grab its date
            if event == 'start' and elem.tag == 'JOUR':
                current_date = elem.get('date')  # e.g. "09/08/2025"

            # When we finish an emission, check & extract
            elif event == 'end' and elem.tag == 'EMISSION':
                # 1) Filter for 'Cinéma'
                type_em = elem.findtext('typeEmission')
                if type_em == 'Cinéma':
                    time = elem.get('heureDiffusion')        # e.g. "21.10"
                    title = elem.findtext('titre')           # e.g. "Thor : Ragnarök"

                    yield {
                        'date':  current_date,
                        'time':  time,
                        'title': title
                    }

                # 2) Clear this <EMISSION> to free memory
                elem.clear()
                # remove any preceding siblings
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

            # Once we finish the entire <JOUR>, clear it too
            elif event == 'end' and elem.tag == 'JOUR':
                elem.clear()


    def _parse_cinema_from_m6(self, xml_path):
        """
        Stream-parse an M6 weekly XML feed and yield all 'Cinéma' diffusions
        with their broadcast date, time, and title.
        """
        # iterparse both start/end so we can pick up the jour date and each diffusion
        context = etree.iterparse(
            xml_path,
            events=('start', 'end'),
            recover=True,
            remove_blank_text=True
        )

        for event, elem in context:

            # When we finish a <diffusion>, see if it's a film and extract
            if event == 'end' and elem.tag == 'diffusion':
                fmt = elem.findtext('format')  # e.g. "Cinéma"
                if fmt == 'Long Métrage':
                    # full timestamp: "YYYY‑MM‑DD HH:MM"
                    dt = elem.findtext('dateheure')
                    # split into date/time (we already have current_date if you prefer)
                    date_part, time_part = dt.split(' ')
                    title = elem.findtext('titreprogramme')
                    

                    yield {
                        'date':  date_part,      # or use current_date
                        'time':  time_part,
                        'title': title
                    }

                # 4) Clear memory for this <diffusion>
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

            # 5) Once we finish the entire <jour>, clear it too
            elif event == 'end' and elem.tag == 'jour':
                elem.clear()

    def download_xml(self, channel: str, **params):
        session = self.sessions[channel]
        if channel == 'TF1':
            xml_path = self._TF1_download_xml(session, **params)
        elif channel == 'M6':
            xml_path = self._M6_download_xml(session, **params)
        else:
            print(f'Invalid Channel Name for xml download: {channel}')
            xml_path = None
        return xml_path

    def parse_cinema(self, channel, xml_path):
        if channel == 'TF1':
            week_schedule = list(self._parse_cinema_from_tf1(xml_path))
        elif channel == 'M6':
            week_schedule = list(self._parse_cinema_from_m6(xml_path))
        else:
            print(f'Invalid Channel Name for parsing: {channel, xml_path}')

        return week_schedule


    def _available_tv_week_starts(self, days_ahead=21, now=None):
        tz = zoneinfo.ZoneInfo("Europe/Zurich")
        now = now or datetime.now(tz)
        now = now.astimezone(tz)

        start_date = now.date()
        end_date = start_date + timedelta(days=days_ahead)

        # Find first Saturday >= start_date (Saturday is weekday 5)
        days_until_sat = (5 - start_date.weekday()) % 7
        first_sat = start_date + timedelta(days=days_until_sat)

        available_saturdays = []
        current_sat = first_sat
        while current_sat <= end_date:
            # Availability for week starting at current_sat is (current_sat - 3 weeks) Saturday at 18:00
            availability_base = current_sat - timedelta(weeks=3)
            availability_dt = datetime(
                availability_base.year,
                availability_base.month,
                availability_base.day,
                18, 0, 0,
                tzinfo=tz
            )
            if now >= availability_dt:
                available_saturdays.append(current_sat)
            current_sat += timedelta(weeks=1)

        return available_saturdays
        

    def _extract_channel(self, filename: str) -> str:
            stem = Path(filename).stem
            if "_" in stem:
                return stem.split("_", 1)[0]
            if "-" in stem:
                return stem.split("-", 1)[0]
            return stem

    def process_competition_folder(self, folder_path: Path= Path('competition_data')):
        folder = Path(folder_path)
        for xml_path in folder.glob("*.xml"):
            channel = self._extract_channel(xml_path.name)
            schedule = self.parse_cinema(channel, xml_path)
            iso_year, iso_week = self.extract_iso_week_from_filename(xml_path)
            
            
            self.scraped_schedules[channel][f'{iso_week}'] = schedule



    def extract_iso_week_from_filename(self, filename: Union[str, Path]) -> Optional[Tuple[int, int]]:
        """
        Returns (iso_year, iso_week) extracted from the filename.

        Handles:
        - Week-style: e.g., "M6_2025_31.xml" -> (2025, 31)
        - Date-style: e.g., "TF1_2025-08-02.xml" -> ISO week for that date
        """
        p = Path(filename)
        stem = p.stem  # filename without extension

        # Try week pattern at end: YYYY_WW or YYYY-WW
        m_week = re.search(r"(\d{4})[_-](\d{1,2})$", stem)
        
        if m_week:
            year = int(m_week.group(1))
            week = int(m_week.group(2))
            
            return year, week  # assume this is already ISO week
