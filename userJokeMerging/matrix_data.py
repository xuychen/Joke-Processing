import numpy as np
import pandas as pd
from scipy import sparse
import pickle

""" name of ratings file """
RATINGS_FILENAME = "data.csv"

""" name of joke features file """
JOKES_FILENAME = "jokes_info.csv"

"""" Enables verbose logging, for debugging """
_DEBUG = True

class matrix_object:
  rows = []
  cols = []
  values = []

  def __init__(self):
    self.rows = []
    self.cols = []
    self.values = []

  def add_value(self, row, col, value):
    self.rows.append(row)
    self.cols.append(col)
    self.values.append(value)

  def compile_matrix(self):
    npcols = np.array(self.cols)
    nprows = np.array(self.rows)
    npvals = np.array(self.values)
    matrix = sparse.csc_matrix((npvals, (nprows, npcols)))
    return matrix

  def to_df(self, matrix):
    df = pd.DataFrame(matrix.toarray())
    column_names = ['isAggressive', 'isIncongruence', 'isMillenial', 'isGenX', 'isGenZ']
    for idx in range(5, matrix.shape[1]):
      column_names.append("user" + str(idx))
    df.columns = column_names
    return df


""" creates user, movie matrix from ratings dataframe """
def create_matrix(ratings_df, jokes_df):
  """ create jokes x (joke features + user) matrix and populate with ratings """
  """ note: empty entries are populated with zeros """

  matrix_handler = matrix_object()

  num_joke_features = 5

  ''' add all joke features '''
  for row_idx in range(0, jokes_df.shape[0]):
    joke_idx = int(jokes_df.iloc[row_idx]["Idx"])
    isAggressive = jokes_df.iloc[row_idx]["isAggressive"]
    isIncongruence = jokes_df.iloc[row_idx]["isIncongruence"]
    generation = jokes_df.iloc[row_idx]["Generation"]
    isMillenial = (generation == "Millenial")
    isGenX = (generation == "Gen X")
    isGenZ = (generation == "Gen Z")

    if(int(isMillenial) == 1.0 and int(isGenX) == 1.0):
      raise Valueerror()

    matrix_handler.add_value(joke_idx - 1, 0, int(isAggressive))
    matrix_handler.add_value(joke_idx - 1, 1, int(isIncongruence))
    matrix_handler.add_value(joke_idx - 1, 2, int(isMillenial))
    matrix_handler.add_value(joke_idx - 1, 3, int(isGenX))
    matrix_handler.add_value(joke_idx - 1, 4, int(isGenZ))

  ''' add all ratings '''
  for row_idx in range(0, ratings_df.shape[0]):
    for joke_idx in range(1, 122):
      col_name = "joke" + str(joke_idx)
      matrix_handler.add_value(joke_idx - 1, row_idx + num_joke_features, ratings_df.iloc[row_idx][col_name])

  matrix = matrix_handler.compile_matrix()
  new_df = matrix_handler.to_df(matrix)

  return matrix, new_df


if __name__=='__main__':
  ratings_df = pd.read_csv(RATINGS_FILENAME)
  jokes_df = pd.read_csv(JOKES_FILENAME)
  matrix, df = create_matrix(ratings_df, jokes_df)
  print(matrix[0])
  print(df)
