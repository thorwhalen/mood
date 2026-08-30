"""A script that searches for news and saves the results to a file.

Meant to be used in a cron on a server to periodically search for news.

"""

import argparse
import inspect

from mood.tools import search_and_save_news

#: CLI flags for each of ``search_and_save_news``'s parameters. Defaults (and the
#: defaults shown in ``--help``) are read from the function itself, so the two
#: cannot drift apart.
_FLAGS = {
    "query": ("-q", "--query"),
    "source": ("--source",),
    "verbose": ("-v", "--verbose"),
    "store": ("--store",),
}


def _make_parser() -> argparse.ArgumentParser:
    """Expose :func:`mood.tools.search_and_save_news` as a command line.

    >>> parser = _make_parser()
    >>> args = parser.parse_args(["-q", "AAPL", "--source", "yahoo_finance_headlines"])
    >>> args.query, args.source, args.verbose
    ('AAPL', 'yahoo_finance_headlines', False)
    >>> parser.parse_args(["--verbose"]).verbose
    True
    >>> parser.parse_args(["--store", "/tmp/elsewhere"]).store
    '/tmp/elsewhere'
    """
    parser = argparse.ArgumentParser(prog="financial-news-search")
    params = inspect.signature(search_and_save_news).parameters
    for name, flags in _FLAGS.items():
        default = params[name].default
        kwargs = {"default": default, "help": repr(default)}
        if isinstance(default, bool):
            kwargs["action"] = "store_true"
        parser.add_argument(*flags, **kwargs)
    return parser


def main(argv=None):
    """Parse ``argv`` (default: ``sys.argv[1:]``) and run the search."""
    search_and_save_news(**vars(_make_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
