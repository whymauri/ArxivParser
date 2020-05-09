import os

from bs4 import BeautifulSoup

DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + '/'
SAMPLE_HTML = "ContextualBandits.html"

def green(_s):
  return "\033[1;32m{}\033[m".format(_s)

def red(_s):
    return "\033[1;31m{}\033[m".format(_s)

def test_parse_html_abstract(parser, page):
  abstract = parser.parse_html_abstract(page)
  assert(len(abstract) ==  785)
  try:
    assert('bandit' in abstract)
    assert('contextual' in abstract)
  except AssertionError:
    msg = "Expected to find: \'bandit\' or \'contextual\'."
    msg = "But neither was found!"
    print(red("Failed abstract test! {}".format(msg)))

def test_parse_html_authors(parser, page):
  first, last = parser.parse_html_authors(page)
  try:
    assert(first == u'Vasilis Syrgkanis') # Vasilis is awesome.
    assert(last == u'Robert E. Schapire')
  except:
    msg = "Expected to find: (u\'Vasilis Syrgkanis\', u\'Robert E. Schapire\')"
    msg += "Instead found: {}".format((first, last))
    print(red("Failed authors test!"))

def test_parse_html_titles(parser, page):
  title = parser.parse_html_titles(page)
  to_match = "Improved Regret Bounds for Oracle-Based Adversarial Contextual "+\
    "Bandits"
  try:
    assert(to_match == title.strip())
  except AssertionError:
    msg = "Expected to find title: {}".format(to_match)
    msg += "Instead found: {}".format(title.strip())
    print(red("Failed html titles test! {}".format(msg)))

def test_parse_html_uri(parser, page):
  uri = parser.parse_html_uri(page)
  to_match = "https://arxiv.org/abs/1606.00313"
  try:
    assert(to_match == uri)
  except AssertionError:
    msg = "Expected to find uri: {}".format(to_match)
    msg += "Instead found: {}".format(uri)
    print(red("Failed html uri test! {}".format(msg)))

def test_parser_all(parser):
  html = open(DIRECTORY + SAMPLE_HTML, 'r')
  soup = BeautifulSoup(html, 'html.parser')
  page = soup.find('li', class_='arxiv-result')

  test_parse_html_abstract(parser, page)
  test_parse_html_authors(parser, page)
  test_parse_html_titles(parser, page)
  test_parse_html_uri(parser, page)

  print(green("Passed parser tests!"))
