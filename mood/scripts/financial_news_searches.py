"""A script that searches for news and saves the results to a file.

Meant to be used in a cron on a server to periodically search for news.

"""

from mood.tools import search_and_save_news
import argh


def main():
    argh.dispatch_command(search_and_save_news)


if __name__ == "__main__":
    main()
