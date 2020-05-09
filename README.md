# ArxivParser
ArxivParser CLI and API -> uses arXiv search API v0.5.6 to generate CSV of requested articles.

# API

I have an API version of this parser running on my personal python box (mau.pythonanywhere.com). You can toy around with the API using this [shared REPL I made.](https://repl.it/@whymauri/QuestionableIncomparableDevices)

# CLI

'''
usage: cli.py [-h] [--amount AMOUNT] search_term start_date end_date

arXivParser CLI

positional arguments:
  search_term      The search term you're using to find articles.
  start_date       The start of the date range used to search for
                   articles.Default is today.
  end_date         The end of the date range used to search for
                   articles.Default is a week ago.

optional arguments:
  -h, --help       show this help message and exit
  --amount AMOUNT  Maximum number of articles to search for. Due to arXiv
                   limitations, this must be 25, 50, 100, or 200.
'''
