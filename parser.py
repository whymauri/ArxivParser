from bs4 import element

class ArxivParser(object):
  '''
    This parser can speficially handle entries for arXiv search v0.5.6.

    Args: None
  '''
  def __init__(self):
    self._abstract_class = 'abstract-full has-text-grey-dark mathjax'
    self._authors_class = 'authors'
    self._title_class = 'title is-5 mathjax'

  @staticmethod
  def __less_tag_mask(idx, content):
    '''
      Will filter the "less" tag that appears on arXiv abstracts.

      Note: a more OOP way of handling these masks is to have a mask
      class with overloaded methods and argument unpacking. This avoids
      having the __no_mask method take in two useless arguments.

      But for now, and for just two masks, this suffices.
      '''
    return idx != len(content) - 2

  @staticmethod
  def __no_mask(idx, content):
    '''
      Will not filter anything.
    '''
    return True

  @staticmethod
  def __parse_text(html_text, mask):
    '''
      Will do a one-level HTML expansion of all the text that is parsed.
      This is because the arXiv search API will highlight words or phrases
      that match the search term. The highlight needs to be removed so
      the content inside can be parsed.

      We also apply a generalized masking function to remove tags that are
      placed deterministically in an arXiv text entry i.e.: "less tag.
    '''
    text_str = ""
    for i, text in enumerate(html_text):
      mask_condition = mask(i, html_text)

      if isinstance(text, element.Tag) and mask_condition:
        text_str += text.contents[0]
      elif isinstance(text, element.NavigableString):
        text_str += text

    return text_str

  def parse_html_abstract(self, page):
    '''
      Finds the abstract in an arXiv entry and then parses the abstract.
    '''
    abstract = page.find('span', class_=self._abstract_class)
    return self.__parse_text(abstract, self.__less_tag_mask)

  def parse_html_authors(self, page):
    '''
      Finds the authors in an arXiv entry and then parses the authors.
      Returns first and last authors.

      Note: can be easily expanded to return all authors.
    '''
    authors = page.find('p', class_=self._authors_class)
    authors = authors.find_all('a')
    authors_plaintext = [a.contents[0] for a in authors]
    first, last = authors_plaintext[0], authors_plaintext[-1]
    return first, last

  def parse_html_titles(self, page):
    '''
      Finds the title of an arXiv entry and then parses the title.
    '''
    title = page.find("p", class_=self._title_class)
    return self.__parse_text(title, self.__no_mask)

  @staticmethod
  def parse_html_uri(page):
    '''
      Finds the uri of a given page and then returns it.
    '''
    uri = page.find('a')
    return uri.get('href')
