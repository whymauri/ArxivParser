import os
import utils

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

app = Flask(__name__)
app.secret_key = os.urandom(24)
DIRECTORY = os.path.abspath(os.path.dirname(__file__))

@app.route("/api/v0/arxiv/search", methods=["GET"])
def search_fulfill():
  '''
    Services API requests for arXiv searches. Is actually technically
    capable of servicing all arXiv search v0.5.6 requests.
  '''
  # Note: we don't need to unpack the arguments since superflous
  # arguments will be naturally ignored by the API.
  args = request.args.to_dict(flat=True)
  arxiv_response = utils.arxiv_request(args)
  clean_documents = utils.process_response(arxiv_response)

  fname = utils.store_as_csv(clean_documents)
  ret = {'entries': clean_documents}
  # Pythonic dictionary update is in-place
  ret.update({'csv_name': fname})
  return jsonify(ret)

@app.route("/api/v0/arxiv/download", methods=["GET"])
def download_fulfill():
  '''
    Will redirect the user to download the csv file that matches
    the given uid.

    Note: currently not secure, but the contents of the csv are not
    critical.
  '''
  uid = request.args.get('fname')
  to_serve = utils.get_csv(fname, directory=DIRECTORY)
  return send_file(to_serve)

if __name__ == "__main__":
  app.run(debug=True)
