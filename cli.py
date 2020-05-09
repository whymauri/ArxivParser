import argparse
import sys
import utils

from datetime import datetime
from datetime import timedelta

'''Command Line Interface for arXivParser'''

'''
  arXiv API analogs of the CLI parameters. We don't use their naming
  schemes because they're slightly convoluted.
'''
CLI_TO_ARXIV_PARAMS = {'amount': 'size',
                       'search_term': 'terms-0-term',
                       'start_date': 'date-from_date',
                       'end_date': 'date-to_date'}

def args_stringify(args):
  '''
    Turns all the arguments into strings so they're properly
    formatted during the GET request.
  '''
  args['amount'] = str(args['amount'])
  args['end_date'] = args['end_date'].strftime("%Y-%m-%d")
  args['start_date'] = args['start_date'].strftime("%Y-%m-%d")
  return args

def args_to_name(args):
  '''
    Generates a name for a csv file based on the passed in arguments.
  '''
  term = args['search_term'].replace(' ', '_')
  date = args['start_date']
  amount = args['amount']

  return term + date + amount

def arxiv_request_and_save(args):
  '''
    arXiv search API request is executed, parsed, and then stored as
    a csv.
  '''
  # Convert readable CLI arguments to arXiv search v0.5.6 params
  args = args_stringify(args)
  arxiv_params = {CLI_TO_ARXIV_PARAMS[key]:value for key, value in args.items()}
  arxiv_response = utils.arxiv_request(arxiv_params)
  clean_documents = utils.process_response(arxiv_response)

  file_name = "{}.csv".format(args_to_name(args))
  utils.store_as_csv(clean_documents, fname=file_name)

def arxiv_valid_amount(_i):
  '''
    Checks that the number of arXiv entries requested by the user are
    compliant with the arXiv search API.
  '''
  valid_amounts = ['25', '50', '100', '200']
  if _i in valid_amounts:
    return _i
  else:
    msg = "Not a valid amount: {}. ".format(str(_i))
    msg += "Must be in {}.".format(str(valid_amounts))
    raise argparse.ArgumentTypeError(msg)

def arxiv_valid_date(_s):
  '''
    Verifies that dates are properly formatted.
  '''
  try:
    return datetime.strptime(_s, "%Y-%m-%d")
  except ValueError:
    msg = "Not a valid date: {}.".format(_s)
    msg += "Should be in (YYYY-MM-DD) zero-padded format"
    raise argparse.ArgumentTypeError(msg)

def make_command_line_parser():
  '''
    Initiates an argparse parser for the arXiv scraper CLI.
  '''
  parser = argparse.ArgumentParser(
      description='arXivParser CLI')

  parser.add_argument(
      'search_term',
      help='The search term you\'re using to find articles.',
      type=str
      )

  # We're going to adhere to the valid date format as specified by
  # arXiv Search v0.5.6 released 2020-02-24

  today = datetime.today()
  week_ago = today - timedelta(days=7)

  parser.add_argument(
      'start_date',
      help='The start of the date range used to search for articles.' +
      'Default is today.',
      default=week_ago.strftime('%Y-%m-%d'),
      type=arxiv_valid_date
      )

  parser.add_argument(
      'end_date',
      help='The end of the date range used to search for articles.' +
      'Default is a week ago.',
      default=today.strftime('%Y-%m-%d'),
      type=arxiv_valid_date
      )

  parser.add_argument(
      '--amount',
      help='Maximum number of articles to search for. Due to arXiv '+
      'limitations, this must be 25, 50, 100, or 200.',
      default=50,
      type=arxiv_valid_amount
      )

  return parser

def main():
  '''Command Line Interface for arXivParser'''
  args = sys.argv[1:]
  parser = make_command_line_parser()
  if not args or '-h' in args or '--help' in  args:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args(args)
  arxiv_request_and_save(vars(args))

if __name__ == '__main__':
  main()
