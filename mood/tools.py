"""Tools for sentiment analysis"""

from dol import Pipe
from scoopy.data_sources import headlines
from mood.sentiment import sentiment_analysis
import json

headlines_mood = Pipe(headlines, sentiment_analysis)


# --------------------------------------------------------------------------------------
# Search and save news

from functools import partial
from typing import Union, Iterable
import os
import re
from dol import JsonFiles, mk_dirs_if_missing
from scoopy import search_news, yahoo_finance_news_search
from mood.util import djoin

filename_safe_pattern = re.compile(r"[^a-zA-Z0-9_\-]")


def mk_filename_safe(string: str) -> str:
    return filename_safe_pattern.sub("_", string)


def current_time_str(format="%Y-%m-%d--%H-%M-%S"):
    from datetime import datetime

    now = datetime.now()
    return now.strftime(format)


def get_key(query, prefix="", suffix="", *, day_folder=False):
    if day_folder:
        day_folder_str = current_time_str("%Y-%m-%d") + os.path.sep
    else:
        day_folder_str = ""
    now_string_seconds = current_time_str("%Y-%m-%d--%H-%M-%S")
    query_str = mk_filename_safe(query)
    return f"{prefix}{day_folder_str}{now_string_seconds}__{query_str}{suffix}"


DFLT_STORE = djoin("news", "searches")
json_file_key = partial(get_key, suffix=".json", day_folder=True)


def _search_and_save_news(
    query, store=DFLT_STORE, *, search_func=search_news, query_to_key=json_file_key
):

    if isinstance(store, str):
        import os

        rootdir = os.path.expanduser(store)
        # create the rootdir if it does not exist
        os.makedirs(rootdir, exist_ok=True)

        store = mk_dirs_if_missing(JsonFiles(rootdir))

    results = search_func(query)

    key = query_to_key(query)
    store[key] = results
    return key


_newsdata_search_default_params = {
    "qInTitle": None,  # Optional: Use if you want keywords in headlines only
    "category": "business, technology, politics",  # Focus on market-relevant categories
    "country": "us,gb,cn,jp,de",  # Major financial hubs
    "language": "en",
    # "timeframe": "24",  # Last 24 hours (Requires premium plan)
    # "prioritydomain": "newsdata.io"  # Optional: Prioritize high-quality sources
}

search_news_from = {
    "newsdata": partial(
        search_news, source="newsdata", **_newsdata_search_default_params
    ),
    "yahoo_finance": partial(search_news, source="yahoo_finance"),
    "yahoo_finance_headlines": partial(search_news, source="yahoo_finance_headlines"),
}


def _resolve_to_file_if_it_is_one(string):
    if isinstance(string, str):
        f = os.path.abspath(os.path.expanduser(string))
        if os.path.isfile(f):
            if (
                f.endswith(".json")
                or f[-4:] in {".txt", ".csv", ".tsv"}
                or os.path.sep in f
            ):
                return f

    return False


def search_and_save_news(
    query: Union[str, Iterable[str]] = "",
    source: str = "yahoo_finance_headlines",
    *,
    verbose: bool = False,
    store=DFLT_STORE,
):
    assert source in search_news_from, f"Unknown news source: {source}"
    prefix = source + os.path.sep
    _search_func = search_news_from[source]
    _file_key = partial(get_key, prefix=prefix, suffix=".json", day_folder=True)

    if isinstance(query, str):
        filepath = _resolve_to_file_if_it_is_one(query)
        if filepath:
            with open(filepath) as f:
                query = json.load(f)
        else:
            query = [query]

    if verbose:
        print(f"Searching and saving news in {store}")

    for q in query:
        t = _search_and_save_news(
            q, search_func=_search_func, query_to_key=_file_key, store=store
        )
        if verbose:
            print(f"{source=}, {q=}, {t=}")
