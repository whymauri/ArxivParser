import csv
import json
import os
import requests

from bs4 import BeautifulSoup
from hashlib import sha256
from parser import ArxivParser

'''
These are the utility functions used in both the arXiv scraper/parser
command line interface and API.
'''

'''
DEFAULT_ARGS are all the necessary GET request paremeters for
arXiv Search v0.5.6 released 2020-02-24.
'''
DEFAULT_ARGS = {"advanced":"",
                "terms-0-field":"all",
                "classification-physics_archives":"all",
                "classification-include_cross_list":"include",
                "date-year":"",
                "date-filter_by":"date_range",
                "date-date_type":"submitted_date",
                "abstracts":"show",
                "order":"-announced_date_first"}

DIRECTORY = os.path.abspath(os.path.dirname(__file__))

'''
These are the necessary parameters to search for results
within a given date range according to arXiv Search v0.5.6
released 2020-02-24.
'''
NECESSARY_PARAMS = {"terms-0-term",
                    "date-from_date",
                    "date-to_date",
                    "size"}

def arxiv_request(args):
  '''
    Joins the user-defined arguments and arXiv search default arguments
    into GET requests parameters. Then executes a GET request to arXiv
    search v0.5.6.
  '''
  valid_args, error = verify_arguments(args.keys())
  if not valid_args:
    return error

  params = args.copy()
  # Pythonic hashtable/dictionary updates are in-place.
  params.update(DEFAULT_ARGS)
  response = requests.get(
      "https://arxiv.org/search/advanced",
      params=params
  )

  return response

def get_csv(fname, directory=None):
  if not directory:
    directory = DIRECTORY
  return directory + fname

def parse_arxiv_entry(result, parser):
  '''
    Given one arXiv search result, we parse out the title, first author,
    last author, abstract, and URI. We then properly encode the result
    and strip unnecessary newlines.
  '''
  abstract = parser.parse_html_abstract(result)
  first_auth, last_auth = parser.parse_html_authors(result)
  title = parser.parse_html_titles(result)
  uri = parser.parse_html_uri(result)

  document = {"abstract": abstract,
              "first_auth": first_auth,
              "last_auth": last_auth,
              "title": title,
              "URI": uri}
  return utf_encode(document)

def process_response(page):
  '''
    Given the raw HTML response from an arXiv search GET request, we
    parse the page for individual arXiv search results. Those results
    are parsed accordingly.
  '''
  parser = ArxivParser()
  soup = BeautifulSoup(page.text, 'html.parser')
  results = soup.find_all("li", class_="arxiv-result")

  documents = []
  for result in results:
    document = parse_arxiv_entry(result, parser)
    documents.append(document)

  return documents

def store_as_csv(documents, fname=None):
  '''
    Determine a valid file name for the arXiv results csv. Once determined,
    we create this csv file and write each row based on the arXiv search
    results.

    keywords:
      fname: the file name for the stored CSV. If being called by CLI,
      we allow the CLI to define the file naming conventions. When called
      by the API, we assign a unique hash ID. (Default = None)

    Notes: a potential extension is to not write a CSV from API call if
    that csv already exists. Not currently implemented.
  '''
  if fname is None:
    search_hash = sha256(json.dumps(documents)).hexdigest()
    fname = "{}.csv".format(search_hash)

  with open(DIRECTORY + '/' + fname, 'w') as csv_file:
    fieldnames = ["title",
                  "first_auth",
                  "last_auth",
                  "abstract",
                  "URI"]

    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    for doc in documents:
      csv_writer.writerow(doc)

  return fname

def utf_encode(documents):
  '''
    This removes all special characters from arXiv results that
    might make writing to csv difficult. We also remove newlines.
  '''
  for key, value in documents.items():
    documents[key] = value.encode('utf-8').strip()
  return documents

def verify_arguments(args):
  '''
    Verify that the user-defined arguments include a search term, a
    start date, an end date, and an amount of results to return.
  '''
  not_included = []
  valid = True

  for param in NECESSARY_PARAMS:
    if param not in args:
      not_included.append(param)
      valid = False

  error = {"error":"Must include " + str(not_included)}
  return valid, error
