from base import BaseSolver
import re
from functools import lru_cache
from pymorphy2 import MorphAnalyzer


class HeurisitcSolver(BaseSolver):
   
    def __init__(self, path: str, path_valid=None):
        super(HeurisitcSolver, self).__init__(path, path_valid)
        self.morph = MorphAnalyzer()
   
    def preprocess(self, columns):
      for column in columns:
        self.train[f"{column}_lemmas"] = self.train[column].apply(self.clean_text)
        self.valid[f"{column}_lemmas"] = self.valid[column].apply(self.clean_text)
 
    def words_only(self, text):
      rg = re.compile("[А-Яа-яA-z]+")
      try:
        return rg.findall(text.lower())
      except:
        return []

    @lru_cache(maxsize=128)
    def lemmatize_word(self, token):
      return self.morph.parse(token)[0].normal_form

    def lemmatize_text(self, text):
      return [self.lemmatize_word(w) for w in text]

    def clean_text(self, text):
      tokens = self.words_only(text)
      lemmas = self.lemmatize_text(tokens)  
      return lemmas
    
    def heuristics_all(self, final_decision=None):
        y_pred = []
        self.preprocess(columns=['premise', "hypothesis"])

        for i, row in self.valid.iterrows():
          
          hyp = row.hypothesis.lower()
          hyp_lem = set(row['hypothesis_lemmas'])
          prem_lem = set(row['premise_lemmas'])
          indic_non_ent = set(['только', 'мужчина']) #TODO add function to compute such words

          if hyp in row['premise'].lower():
             y_pred.append('entailment')
          elif len(prem_lem & hyp_lem)/len(hyp_lem) <= 1/3 or len(row['premise'].split()) < 29 or len(indic_non_ent & hyp_lem) > 0:
            y_pred.append('not_entailment')
          elif len(prem_lem & hyp_lem)/len(hyp_lem) == 0.75 or len(prem_lem & hyp_lem)/len(hyp_lem) == 1 or len(prem_lem & hyp_lem)/len(hyp_lem) == 2/3:
            y_pred.append('entailment')
          elif len(row['premise'].split()) > 32:
            y_pred.append('entailment')
          else:
            y_pred.append(final_decision(test_size=1)[0])
        
        return y_pred