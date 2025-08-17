from client import APIClient
from requests.exceptions import HTTPError
from utils.text_cleaning import normalize
import numpy as np

_client = APIClient()

def search_movie(title):
    """Searches for a movie by title and returns the different ids"""
    response = _client.get('search/movie', params={'query': title, 'include_adult': 'true', 'language': 'fr'})
    #movie_ids = [response["results"][i]['id'] for i in range(response['total_results'])]
    return response

def get_movie_details(movie_id: str):
    """Fetches full movie details from a given movie_id"""
    response = _client.get(f'movie/{movie_id}', params={'language': 'fr'})
    return response


def find_best_match(title, known_runtime=90, top_n = 10, verbose = False):
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
        if details is None or not bool(details):
            continue
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
    if details is None or not bool(details):
        title = None
    else:
        title = details["title"]
    return title


def get_genre_dict(): # TMDB Genre dictionary
    genre_dict = _client.get('genre/movie/list')
    return genre_dict['genres']

def get_movie_features(movie_id, return_duration = False):
    #adult, release year, genre ids, original_language, popularity, vote_average, runtime, origin_country
    

    details = get_movie_details(movie_id)
    keep = ['adult', 'original_language', 'popularity', 'release_date', 'revenue', 'vote_average', 'genres']
    if return_duration:
        keep.append('runtime')
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


def get_movie_poster_path(movie_id, width = 200):
    # Sometimes works with certain width sometimes the exact corresponding one, yesterday it worked one way, the other not
    if movie_id is None or (isinstance(movie_id, float) and np.isnan(movie_id)):
        return None
    
    details = get_movie_details(movie_id)

    try:
        poster_path = details.get('poster_path') # get poster path corresponding to the movie
    except:
        print(poster_path)

    if not poster_path:
        return None
    #movie_images = get_movie_images(movie_id) # Search for poster path in all movie images to find the width of the image, necessary for api call
    #posters = movie_images['posters']
    #found = False
    #for poster in posters:
    #    if poster_path in poster['file_path']:
    #        poster_path = poster['width'] + poster_path
    #        found = True
    #        break
    
    #if not found:
    #    pass

    # Retrieve base url for images
    #configuration = get_config_details()
    #images_base_url = configuration['images']['base_url']
    images_base_url = "http://image.tmdb.org/t/p/"

    poster_path = f'w{width}/poster_path'
    full_poster_path = f"{images_base_url.rstrip('/')}/{poster_path.lstrip('/')}"
    return full_poster_path

    #_client.get_images(images_base_url, full_poster_path)



def get_movie_images(movie_id):
    response = _client.get(f'movie/{movie_id}/images')
    return response

def get_config_details():
    response = _client.get('configuration')
    return response
