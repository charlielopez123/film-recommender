SITES = {
    'TF1': { # {date:%Y-%m-%d}
        'base_html':  'https://tf1pro.com/grilles-tv/TF1/{date}',
        'base_xml':   'https://tf1pro.com/grilles-xml/16/{date}',
        'default_headers': {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/133.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
    },

    'M6': {
        'base_xml':   'https://pro.m6.fr/m6/grille/{year}-{week}.xml',
        'default_headers': {}
    },
}


CHANNELS = list(SITES.keys())