from client import APIClient
from requests.exceptions import HTTPError
from utils.text_cleaning import normalize

_client = APIClient()

def search_movie(title):
    """Searches for a movie by title and returns the different ids"""
    response = _client.get('search/movie', params={'query': title, 'include_adult': 'true', 'language': 'fr'})
    #movie_ids = [response["results"][i]['id'] for i in range(response['total_results'])]
    return response

def get_movie_details(movie_id):
    """Fetches full movie details from a given movie_id"""
    return _client.get(f'movie/{movie_id}', params={'language': 'fr'})


def find_best_match(title, known_runtime=0, top_n = 10, verbose = False):
    """
    Searches for a movie by title and finds the best match based on runtime.
    """
    
    response = search_movie(title)
    num_results = len(response['results'])
    if verbose:
        print(response)

    # If no results found give up
    if num_results == 0:
        best_id = find_best_id_decomposed(title, known_runtime)
        if best_id and verbose:
            print(title)
            print(get_movie_title(best_id))
        return best_id
    

    top_movie_ids = [response["results"][i]['id'] for i in range(num_results)]
    

    #Search for top_n ids
    if top_n < response["total_results"]:
        top_n_movie_ids = top_movie_ids[:top_n]
    else:
        top_n_movie_ids = top_movie_ids

    # if translated title is the same as the provided title return the corresponding movie_id
    for movie in response["results"][:len(top_n_movie_ids)]:
        n_title = normalize(title)
        n_tmdb_title = normalize(movie["title"])
        if ( (n_title in n_tmdb_title) or (n_tmdb_title in n_title) ):
            best_id = movie['id']
            if verbose:
                print(get_movie_title(best_id))
            return best_id
        
    # Assuming it is the most unlikely case that the first movie has a runtime of 0 and the match is wrong
    if num_results==1:
        best_id = top_n_movie_ids[0]
        if verbose:
            print(get_movie_title(best_id))
        return best_id 
        
    
    
    # Assuming it is the most unlikely case that the first movie has a runtime of 0 and the match is wrong
    details = get_movie_details(top_n_movie_ids[0])
    if details["runtime"] == 0 or known_runtime == 0:
        best_id = top_n_movie_ids[0]
        if verbose:
            print(get_movie_title(best_id))
        return best_id 
    
    
    top_n_movie_names = [movie["title"] for movie in response["results"][:len(top_n_movie_ids)]]
    if verbose:
        print(top_n_movie_names)
        print("known_runtime: ", known_runtime)

    # Compare the runtime of the top_n movies with the known runtime and find the best match
    best_id = None
    lowest_diff = float("inf")
    for i, id in enumerate(top_n_movie_ids):
        try:
            details = get_movie_details(id)
        except HTTPError:
            print("unknown id:", id)
            print("unknown title:", top_n_movie_names[i])
        runtime = details["runtime"]
        #print(runtime)
        diff = abs(runtime - known_runtime)
        if diff < lowest_diff:
            lowest_diff = diff
            best_id = id
    if verbose:
        print(title)
        print(get_movie_title(best_id))
        print("_____________________________________\n")
    return best_id



def get_movie_title(movie_id):
    """Fetches the title of a movie from a given movie_id"""
    details = get_movie_details(movie_id)
    return details["title"]


def get_genre_dict(): # TMDB Genre dictionary
    genre_dict = _client.get('genre/movie/list')
    return genre_dict['genres']

def get_movie_features(movie_id):
    #adult, release year, genre ids, original_language, popularity, vote_average, runtime, origin_country
    details = get_movie_details(movie_id)

    keep = {'adult', 'original_language', 'popularity', 'release_date', 'revenue', 'vote_average', 'genres'}
    movie_features = {k: v for k, v in details.items() if k in keep}
    return movie_features


def find_best_id_decomposed(title, known_runtime):
    
    if ":" in title:
        left, right = title.split(":", 1)
        for part in (left.strip(), right.strip()):
            best_id = find_best_match(part, known_runtime)
            if best_id:
                return best_id
            
    if "-" in title:
        left, right = title.split("-", 1)
        for part in (left.strip(), right.strip()):
            best_id = find_best_match(part, known_runtime)
            if best_id:
                return best_id

    # give up
    return None
