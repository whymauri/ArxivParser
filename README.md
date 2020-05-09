# ArxivParser
ArxivParser CLI and API -> uses arXiv search API v0.5.6 to generate CSV of requested articles.

# API

I have an API version of this parser running on my personal python box (mau.pythonanywhere.com). You can toy around with the API using this [shared REPL I made.](https://repl.it/@whymauri/QuestionableIncomparableDevices)

You can edit the shared REPL. REPL will redirect you to a personal copy of the REPL code. If it hangs, just give it a reload.

The REPL prompt will generate a link you can click that will download the CSV of the given search entry. It will be generically titled, `download.csv`.

# CLI

```
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
```

## Example Usage

`python cli.py "Contextual Bandits" 2013-06-02 2016-06-02 --amount 25`

Will search for the term "Contextual Bandits" from the date `2013-06-02` until the date `2016-06-02`. It will report at most 25 results.


`python cli.py "Neural Architecture Search" 2018-01-01 2020-05-02`

Will search for the term "Neural Architecture Search" from the date `2018-01-01` until the date `2020-05-02`. It will report at most 50 results, the default parameter for amount.

`python cli.py "Neural Architecture Search" 2018-01-01 2020-05-02 23`

Will return an error, since 23 is not a valid number of entries to search for according to arXiv search v0.5.6.

# Tests

Running `python run_tests.py` will run a comprehensive suite of tests for the Parser class.
