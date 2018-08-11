import pandas as pd

def stringToBinaryDict (array):
    dictionary = {}
    uniq_array = array.unique()
    arraySize = uniq_array.size # find out the unique size of an array
    
    if arraySize > 2:
        # normal
        for i in range(arraySize):
            alist =  [-1 for j in range(arraySize)]
            alist[i] = 1
            dictionary[uniq_array[i]] = alist
    elif arraySize == 2:
        # gender
        dictionary[uniq_array[0]] = 1
        dictionary[uniq_array[1]] = -1
    
    return uniq_array, dictionary

def userJokeMerge(userCSV, jokeCSV):
    userData = pd.read_csv(userCSV)
    jokeData = pd.read_csv(jokeCSV)
    userInfo = userData.iloc[:,:66]
    
    jokeRating = jokeRatingPreprocessing(userData)
    jokeInfo = jokeInfoPreprocessing(jokeData)
    jokeInfoRating = jokeRating.merge(jokeInfo, how="left", left_on="index",right_on="Joke ID")
    
    # melting
    meltData = pd.melt(jokeInfoRating, id_vars=jokeInfoRating.columns[-6:], value_vars=jokeInfoRating.columns[1:-6], var_name="User", value_name="Joke Rating")
    
    # merge with userInfo
    userJoke = meltData.merge(userInfo, how="left", left_on="User", right_on="Respondent ID")
    
    # reorder
    newCols = list(userJoke.columns[8:]) + list(userJoke.columns[:6]) + ["Joke Rating"]
    userJoke = userJoke.loc[:, newCols]
    return userJoke

def jokeRatingPreprocessing(userData):
    jokeRating = pd.DataFrame(userData.iloc[:, 66:-1])
    jokeRating.index = userData["Respondent ID"]
    jokeRating = jokeRating.transpose()
    return jokeRating.reset_index()

def jokeInfoPreprocessing(jokeData):
    jokeInfo = jokeData.drop(["Idx", "Generation"], axis = 1)
    uniq, dictionary = stringToBinaryDict(jokeData["Generation"])
    jokeInfo[uniq] = pd.DataFrame(jokeData["Generation"].map(dictionary).values.tolist(), index=jokeInfo.index)
    return jokeInfo

userJokeMerge("cleaned.csv", "jokesInfo.csv").to_csv("userJoke.csv")
