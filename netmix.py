import json
from threading import Thread
import datetime
import pandas as pd
import requests
import numpy as np
import base64
import io
from timeit import default_timer as timer
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, ctx
from plotly import graph_objects as go
import sys

use_kws = True
key = "took it out to submit!"
ignore = {"&", "the", "and", "or", "movies", "shows", "movie", "show", "program",
          "programs", "programme", "programmes", "film", "based", "on", "tv", "films",
          "century", "a", "films", "of", "on"}
_kw = "keywords" if use_kws else "genre"

def get_imgs(dict_):
    # Get ids
    ids = [k[1] for k in dict_]
    urls = {}

    # API endpoint and parameters
    endpoint = 'https://api.themoviedb.org/3/{media_type}/{id}/images'
    params = {'api_key': key}

    # loop through each id and fetch images
    for i, id in enumerate(ids):
        media_type = 'movie' if int(id) < 100000 else 'tv'
        url = endpoint.format(media_type=media_type, id=id)
        response = requests.get(url, params=params)
        try:
            results = response.json()['posters']
            url = 'https://image.tmdb.org/t/p/original' + results[0]["file_path"]
            urls[list(dict_)[i]] = url
        except (KeyError, IndexError):
            print(list(dict_)[i], response.json())
            continue
    return urls

def process(title, return_p=False):
    ind = title.find(":")
    p = 1
    if ind == -1:
        # Probably a movie so weight as 3 TV episodes
        p = 3
        cropped = title
    else:
        cropped = title[:ind]
    if return_p:
        return cropped, p
    return cropped

def normalize(distr):
    p = np.array(list(distr.values())) / sum(distr.values())
    return {s: p_ for s, p_ in zip(distr, p)}

def sample(ds, yrs_ago=2, n=150, m=300):
    assert n <= m

    # Cutoff for titles older than yrs_ago
    ds["Date"] = pd.to_datetime(ds["Date"])
    lower_thres = datetime.datetime.now() - datetime.timedelta(days=yrs_ago*365)
    ds = ds[lower_thres <= ds["Date"]]

    # Pick top m newest titles (weight with watch frequency)
    sample = {}
    for i, row in ds.iterrows():
        cropped, p = process(row["Title"], return_p=True)
        sample[cropped] = sample.get(cropped, 0) + p
        if len(sample) >= m:
            break

    # Sample up to n titles from top m, weighted by # occurences
    p = normalize(sample)
    n = min(len(sample), n)
    sample = np.random.choice(list(sample), n, replace=False, p=list(p.values()))
    return {s: p[s] for s in sample}

def keywords_(query, tv=False, running=None, kw_ids=None):
    tmp = {}
    def kw(tv):
        queryp = requests.utils.quote(query)
        typ = "tv" if tv else "movie"
        kw_k = "results" if tv else "keywords"

        search_url = f'https://api.themoviedb.org/3/search/{typ}?api_key={key}&query={queryp}'
        search_response = requests.get(search_url).json()

        movie_id = search_response['results'][0]['id']
        genre_ids = search_response["results"][0]["genre_ids"]

        if use_kws:
            keywords_url = f'https://api.themoviedb.org/3/{typ}/{movie_id}/keywords?api_key={key}'
            keywords_response = requests.get(keywords_url).json()
            kws = []
            for keyword in keywords_response[kw_k]:
                kws.append(keyword["name"])
                tmp[keyword["name"]] = keyword["id"]

        else:
            keywords_url = f'https://api.themoviedb.org/3/genre/{typ}/list'
            params = {"api_key": key}
            response = requests.get(keywords_url, params=params)
            genres = response.json()["genres"]
            kws = [genre["name"] for genre in genres if genre["id"] in genre_ids]

        return kws

    try:
        kws = kw(tv)
    except Exception as e:
        try:
            kws = kw(not tv)
        except Exception as e:
            print(f"'{query}' failed!")
            return []

    sep = set()
    for word in kws:
        for kw in word.lower().split():
            if word not in ignore:
                sep.add(kw)
                kw_ids[kw] = tmp[word]

    sep = list(sep)
    if running is not None:
        running[query] = sep
    return sep

def keywords(queries, kw_ids=None):
    s = timer()

    threads = []
    running = {}
    if kw_ids is None:
        kw_ids = {}

    for query in queries:
        tv = "episode" in query.lower() or "season" in query.lower()
        i = query.find(":")
        query = query[:i] if i != -1 else query
        threads.append(Thread(target=keywords_, args=(query, tv, running, kw_ids)))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(f"[keywords] n={len(queries)}: {round(timer() - s, 2) }s")
    return running, kw_ids

def get_onehot(kw, vocab):
    res = np.zeros((len(kw), len(vocab)))
    for i, kw_list in enumerate(kw.values()):
        for word in kw_list:
            if word not in ignore:
                res[i, vocab.index(word)] = 1
    return res

def predict(marginal, vocab):
    predicted = np.array([np.random.binomial(1, p) for p in marginal])
    tags = vocab[predicted == 1]
    return predicted, tags

def hits(tags, kw_ids, seen, movies=None, tvs=None, threaded=True):

    def collect_(results_iter, title_key, results_dict, n=5):
        i = 0
        while i < n:
            try:
                next_ = next(results_iter)
                title = next_[title_key]
            except StopIteration:
                break
            if process(title) not in seen:
                title = (title, next_["id"])
                results_dict[title] = results_dict.get(title, 0) + 1
                i += 1

    def hits_(tag, movies, tvs):
        ptags = str(kw_ids[tag])

        base_url = "https://api.themoviedb.org/3/"

        movie_query_url = base_url + "discover/movie?api_key=" + key + "&sort_by=popularity.desc&with_keywords=" + ptags
        tv_query_url = base_url + "discover/tv?api_key=" + key + "&sort_by=popularity.desc&with_keywords=" + ptags

        movie_response = requests.get(movie_query_url)
        tv_response = requests.get(tv_query_url)

        movie_results = iter(movie_response.json()["results"])
        tv_results = iter(tv_response.json()["results"])

        collect_(movie_results, "title", movies)
        collect_(tv_results, "name", tvs)

    threads = []
    if movies is None:
        movies = {}
    if tvs is None:
        tvs = {}

    for tag in tags:
        if threaded:
            threads.append(Thread(target=hits_, args=(tag, movies, tvs)))
        else:
            hits_(tag, movies, tvs)

    if threaded:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    return movies, tvs

def top_hits(n, marginal, vocab, kw_ids, seen, threaded=True, m=10):

    def hit_(movies, tvs, tags):
        _, tags_ = predict(marginal, vocab)
        for tag in tags_:
            tags[tag] = tags.get(tag, 0) + 1
        hits(tags_, kw_ids, seen, movies, tvs)

    s = timer()

    movies = {}
    tvs = {}
    tags = {}
    threads = []

    for i in range(m):
        if threaded:
            threads.append(Thread(target=hit_, args=(movies, tvs, tags)))
        else:
            hit_(movies, tvs, tags)

    if threaded:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    movies = dict(sorted(movies.items(), key=lambda i: i[1], reverse=True))
    tvs = dict(sorted(tvs.items(), key=lambda i: i[1], reverse=True))
    tags = dict(sorted(tags.items(), key=lambda i: i[1], reverse=True))

    top_movies = list(movies)[:n]
    top_shows = list(tvs)[:n]
    top_tags = list(tags)[:n]

    print(f"[top_hits] m={m}: {round(timer() - s, 2)}s")
    return top_movies, top_shows, top_tags

def _sample(hist1, hist2, name1, name2, set_progress):
    # Sample titles from viewing histories
    if set_progress:
        set_progress((5, f"Sampling from {name1}'s watched list..."))
    titles1 = sample(hist1, n=50, m=200)

    if set_progress:
        set_progress((15, f"Sampling from {name2}'s watched list..."))
    titles2 = sample(hist2, n=50, m=200)

    # Get vocabulary and keywords
    seen = set(titles1) | set(titles2)
    kw_ids = {}

    if set_progress:
        set_progress((25, f"Fetching keywords for {name1}'s watched list..."))
    kw1, _ = keywords(titles1, kw_ids=kw_ids)

    if set_progress:
        set_progress((35, f"Fetching keywords for {name2}'s watched list..."))
    kw2, _ = keywords(titles2, kw_ids=kw_ids)

    if set_progress:
        set_progress((45, f"Vectorizing and creating MLE estimate..."))

    return kw1, kw2, kw_ids, seen

def _joint(kw1, kw2, name1, name2, set_progress):
    vocab = set()
    for kw_list in list(kw1.values()) + list(kw2.values()):
        for kw in kw_list:
            if kw not in ignore:
                vocab.add(kw)
    vocab = list(vocab)

    # Get 1-hot vector representations of vocabulary
    onehot1 = get_onehot(kw1, vocab)
    onehot2 = get_onehot(kw2, vocab)

    onehot = np.concatenate([onehot1, onehot2])
    vocab = np.array(vocab)

    # Prune keywords that occur <0.5% of the time
    mask = np.sum(onehot, axis=0) < int(0.005 * len(vocab))
    onehot = np.delete(onehot, np.where(mask), axis=1)
    onehot1 = np.delete(onehot1, np.where(mask), axis=1)
    onehot2 = np.delete(onehot2, np.where(mask), axis=1)
    vocab = vocab[~mask]

    # Assuming two things:
    # 1. All movies drawn from same distribution,
    # 2. All keywords independent (not true!)
    # Our MLE estimate for each parameter is # match / # total
    # (no laplace smoothing b/c 0 entries are by default disallowed)
    marginal = np.sum(onehot, axis=0) / len(onehot)
    marginal1 = np.sum(onehot1, axis=0) / len(onehot1)
    marginal2 = np.sum(onehot2, axis=0) / len(onehot2)

    return marginal, marginal1, marginal2, vocab

def _hits(marginal, vocab, kw_ids, seen, name1, name2, set_progress):
    # Predict some movies and tv shows!
    # This works by constructing a distribution over the movies generated
    # by the keywords (sampled from the keyword distribution), and taking
    # the top-n categories from the movie/tv/keyword distribution
    if set_progress:
        set_progress((65, f"Generating top hits for {name1} and {name2}..."))
    movies, tvs, tags = top_hits(5, marginal, vocab, kw_ids, seen)
#     print('top movies', movies)
#     print('top tvs', tvs)
#     print('top tags', tags)
    return movies, tvs, tags

def netmix(hist1, hist2, name1=None, name2=None, set_progress=None):
    kw1, kw2, kw_ids, seen = _sample(hist1, hist2, name1, name2, set_progress)
    marginal, m1, m2, vocab = _joint(kw1, kw2, name1, name2, set_progress)
    movies, tvs, tags = _hits(marginal, vocab, kw_ids, seen, name1, name2, set_progress)

    similarity = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))

    result = {"m1": m1, "m2": m2, "marginal": marginal, "vocab": vocab,
              "movies": movies, "tvs": tvs, "tags": tags, "sim": similarity}
    return result

def build_app():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        dbc.Row([
            html.H1("🔥❤️‍🔥 NetMix 🔥❤️‍🔥",
                    style=dict(fontSize="80px", textAlign="center", marginTop="20px")),
            html.P(
                "What should we watch together? Probability will decide for you! :)",
                style=dict(
                    fontSize="30px",
                    textAlign="center",
                    marginBottom="30px"
                )
            ),
            html.Hr(style=dict(width="50%", margin="auto", marginBottom="30px"))
        ]),
        dbc.Row([
            dbc.Col(
                [
                    dbc.Input(placeholder="Enter first person's name...", size="lg", className="mb-3", id="name1"),
                    dcc.Upload(
                        dbc.Button("Upload Netflix history!", style=dict(width="100%", marginTop="0px"), id="btn1"),
                        id="upload1"), html.Div(id='output-data-upload'),
                    dcc.Store(id="upload1-out")
                ],
                width={"size": 2, "offset": 3}
            ),
            dbc.Col(
                html.H1("×", style=dict(fontSize="90px", textAlign="center")),
                width={"size": 2, "offset": 0}
            ),
            dbc.Col(
                [
                    dbc.Input(placeholder="Enter second person's name...", size="lg", className="mb-3", id="name2"),
                    dcc.Upload(
                        dbc.Button("Upload Netflix history!", style=dict(width="100%", marginTop="0px", id="bt2")),
                        id="upload2"
                    ),
                    dcc.Store(id="upload2-out")
                ],
                width={"size": 2, "offset": 0}
            ),
        ]),
        dbc.Row([
            html.Div(
                [
                    dbc.Progress(id="progress_bar", striped=True, animated=True),
                    html.P("Upload data and enter names to get started!", id="progress_bar-text",
                           style=dict(fontSize="18px", marginTop="5px")),

                    # Go stores
                    dcc.Store(id="start"),
                    dcc.Store(id="titles-go"),
                    dcc.Store(id="joint-go"),
                    dcc.Store(id="hits-go"),

                    # Data stores
                    dcc.Store(id="start"),
                    dcc.Store(id="titles"),
                    dcc.Store(id="joint"),
                    dcc.Store(id="hits"),

                    # results!
                    html.Div(id="discover")
                ],
                id="progress",
                style=dict(width="50%", margin="auto", marginTop="20px"),
            ),
        ])
    ])

    def parse_contents(contents, filename):
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

    @app.callback(Output("upload1-out", "data"),
                  Input('upload1', 'contents'),
                  State('upload1', 'filename'))
    def update_output2(contents, fname):
        if contents is not None:
            hist = parse_contents(contents, fname)
            return hist.to_dict()

    @app.callback(Output("upload2-out", "data"),
                  Input('upload2', 'contents'),
                  State('upload2', 'filename'))
    def update_output2(contents, fname):
        if contents is not None:
            hist = parse_contents(contents, fname)
            return hist.to_dict()

#     @app.callback(Output("start", "data"),
    @app.callback(Output("discover", "children"),
                  Input("upload2-out", "data"),
                  Input("upload1-out","data"),
                  Input("name1", "value"),
                  Input("name2", "value"))
    def start_netmix(d1, d2, n1, n2):
        if d1 is None or d2 is None or n1 is None or n2 is None:
            raise PreventUpdate
        else:
            result = netmix(pd.DataFrame.from_dict(d1), pd.DataFrame.from_dict(d2))
            sim = round(result["sim"] * 100, 1)
            messages = {(0, 10): "Yikes...",
                        (10, 40): "...Maybe go on a hike instead :(",
                        (40, 60): "You guys will make great friends!",
                        (70, 80): "You were meant to be together!",
                        (80, 100): "NETFLIX-APPROVED SOULMATES."}

#             marginals = pd.DataFrame({n1: result["m1"], n2: result["m2"], "Combined": result["m"]})
#             print(marginals)
#
#             fig = go.Figure()
#             for el in zip([0.3, 0.5, 0.8], ['#1f77b4', '#2ca02c', '#9467bd']):
#                 fig.add_scatter(x=xs, y=ys, mode="markers")
#                 fig.add_bar(x=xs, y=ys, marker=dict(color=el[1], opacity=0.5))
#             fig.layout = dict(showlegend=True, bargap=0.75, barmode="overlay")

            try:
                m = [msg for r, msg in messages.items() if r[0] <= sim <= r[1]][0]
            except IndexError:
                m = "You guys will make great friends!"

            result["movies"] = [r[0] for r in result["movies"]]
            result["tvs"] = [r[0] for r in result["tvs"]]

            children = [
                dbc.Row([
                    dbc.Col([
                        html.H1(f"Based on your Netflix history, your compatibility is {sim}%."),
                        html.P(
                            m,
                            style=dict(
                                fontSize="34px",
                                marginBottom="30px"
                            )
                        )]
                    ),
                    dbc.Col(
#                         dcc.Graph(figure=fig),
                        width={"size": 4}
                    )
                ]),

                html.P(
                    [html.Strong("Recommended movies: "), ", ".join(result["movies"])],
                    style=dict(fontSize="24px")
                ),
                html.P([html.Strong("Recommended shows: "), ", ".join(result["tvs"])], style=dict(fontSize="24px") ),
                html.P([html.Strong("Your next favorite will be a movie about..."),
                        ", ".join(result["tags"])], style=dict(fontSize="24px"))
            ]
            return children

#     @app.callback(Output("progress_bar", "value"),
#                   Output("progress_bar-text", "children"),
#                   Output("titles-go", "data"),
#                   Output("joint-go", "data"),
#                   Output("hits-go", "data"),
#                   Input("start", "data"),
#                   Input("titles", "data"),
#                   Input("joint", "data"),
#                   Input("hits", "data"),
#                   State("name1", "value"),
#                   State("name2", "value"),
#                   prevent_initial_call=True)
#     def update_pbar(data0, data1, data2, data3, n1, n2):
#         id_ = ctx.triggered_id
#         go = [dash.no_update, dash.no_update, dash.no_update]
#         print(id_, "update pbar called!")
#         if "start" in id_:
#             t = (5, f"Fetching information for {n1} and {n2}'s watchlists...")
#             go[0] = 1
#         if "titles" in id_:
#             t = (40, f"Generating joint distribution and computing MLE estimate...")
#             go[1] = 1
#         if "joint" in id_:
#             t = (50, f"Generating recommendations...")
#             go[2] = 1
#         if "hits" in id_:
#             t = (100, "Done! Upload new data or names to start again.")
#         print(t, go)
#         return [*t, *go]
#
#     @app.callback(Output("titles", "data"),
#                   Input("titles-go", "data"),
#                   State("upload2-out", "data"),
#                   State("upload1-out","data"),
#                   State("name1", "value"),
#                   State("name2", "value"),
#                   prevent_initial_call=True)
#     def sample_cb(_, hist1, hist2, name1, name2):
#         print("sample cb called!")
#         hist1 = pd.DataFrame.from_dict(hist1)
#         hist2 = pd.DataFrame.from_dict(hist2)
#         return _sample(hist1, hist2, None, None, None)
#
#     @app.callback(Output("joint", "data"),
#                   Input("joint-go", "data"),
#                   State("titles", "data"),
#                   prevent_initial_call=True)
#     def joint_cb(_, kws):
#         print("joint cb called!")
#         kw1, kw2, seen = kws
#         info = list(_joint(kw1, kw2, seen, None, None, None))
#         for i in range(len(info)):
#             if not isinstance(info[i], dict):
#                 try:
#                     info[i] = info[i].tolist()
#                 except AttributeError:
#                     info[i] = list(info[i])
#         return info
#
#     @app.callback(Output("hits", "data"),
#                   Input("hits-go", "data"),
#                   State("joint", "data"),
#                   prevent_initial_call=True)
#     def hit_cb(_, info):
#         m, m1, m2, vocab, kw_ids, seen = info
#         movies, tvs, tags = _hits(np.array(m), np.array(vocab), kw_ids, set(seen), None, None, None)
#         result = {"movies": movies, "tvs": tvs, "tags": tags, "m": m,
#                   "m1": m1, "m2": m2}
#         return result
#
#     @app.callback(Output("discover", "children"),
#                   Input("hits", "data"),
#                   prevent_initial_call=True)
#     def show(info):
#         print(info["movies"])
#         print(info["tags"])
#         print(info["m1"])

    return app


if __name__ == "__main__":
    try:
        sys.argv[1]
        hist1 = pd.read_csv("/Users/ryan/Downloads/ryan.csv")
        hist2 = pd.read_csv("/Users/ryan/Downloads/PERSON.csv")
        result = netmix(hist1, hist2)

        media = result["tvs"] + result["movies"]
        print(get_imgs(result["movies"]))

        n1, n2 = "Ryan", "PERSON"

        sim = round(result["sim"] * 100, 1)
        messages = {(0, 10): "Yikes...",
                    (10, 40): "...Maybe go on a hike instead :(",
                    (40, 60): "You guys will make great friends!",
                    (70, 80): "You were meant to be together!",
                    (80, 100): "NETFLIX-APPROVED SOULMATES."}

        marginals = pd.DataFrame({"Keywords": result["vocab"],
                                  n1: result["m1"],
                                  n2: result["m2"],
                                  "Combined": result["marginal"]})
        print(marginals)

        import plotly.figure_factory as ff

        pmf = marginals[n1]

        # Create a histogram trace for the PMF
        hist_trace = go.Histogram(x=pmf, histnorm='probability density')

        # Create a density trace for the PMF
        density_trace = ff.create_distplot([pmf], ['PMF'], curve_type='kde')

        # Combine the traces and layout into a Figure object
        fig = go.Figure(data=[hist_trace, density_trace['data'][0]], layout=density_trace['layout'])

        # Show the figure
        fig.show()

        colors = ['#1f77b4', '#2ca02c', '#9467bd']
        fig = ff.create_distplot([marginals[n1], marginals[n2], marginals["Combined"]],
                                 [n1, n2, "Combined"],
                                 show_rug=False)#, colors=colors)
        fig.show()

        fig = go.Figure()
        for el in zip([0.3, 0.5, 0.8], colors, [n1, n2, "Combined"]):
            fig.add_trace(
                go.Bar(
                    x=np.arange(len(marginals)),
                    y=marginals[el[2]],
                )
            )
        fig.update_xaxes(type="category", tickangle=90)
        fig.layout = dict(showlegend=True)

        fig.show()

    except IndexError:
        app = build_app()
        app.run_server(debug=True)
