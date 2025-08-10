import benepar
import nltk
import ssl
import warnings
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

ssl._create_default_https_context = ssl._create_unverified_context
benepar.download('benepar_en3')
nltk.download('punkt')
parser = benepar.Parser('benepar_en3')
logger.info("Benepar parser and NLTK punkt tokenizer initialized.")

def constituency_clauses(text):
    logger.info("Extracting constituency clauses.")
    clauses = []
    for sent in nltk.sent_tokenize(text):
        tree = parser.parse(sent)
        leaves = tree.leaves()
        subs = []
        for sub in tree.subtrees():
            if sub.label() == 'S' and sub.leaves() != leaves:
                raw = ' '.join(sub.leaves())
                cleaned = re.sub(r'\s+([,.;:!?])', r'\1', raw)
                subs.append(cleaned)
        if subs:
            clauses.extend(subs)
        else:
            raw = ' '.join(leaves)
            cleaned = re.sub(r'\s+([,.;:!?])', r'\1', raw)
            clauses.append(cleaned)
    return clauses
