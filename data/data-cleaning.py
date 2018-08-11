#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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

def clean_data(filename):
    gender_column = "What gender are you?"
    income_column = "What is your income range?"
    music_column = "What types of movies do you like? (select all that apply)"
    internet_column = "What is the average number of hours that you spend on the Internet per day?"
    children_column = "How many children do you have?"
    vehicle_column = "Do you or your family own a vehicle and if so, how many?"
    age_column = "Which age group are you in?"
    split_columns = ["Region of Ancestry", "Current Region of Occupancy", "Ethnicity", "Biological Gender", "Education", "Occupation", "Preferred Music Genre", "Marital Status"]

    # preprocessing
    data = pd.read_csv(filename)
    data = data.rename(index=str, columns = {data.columns[1]: "Gender Role", data.columns[28]: "Political Affiliation"})
    data = data.drop(data.index[117])
    split_columns_dictionary = {}
    
    # gender questions
    for i in range(1, 21):
        column = data.iloc[:, i].div(100)
        
        if (i == 1):
            data.iloc[:, i] = column
        else:
            data.iloc[:, 1] = data.iloc[:, 1] + column
    
    data.iloc[:, 1] = data.iloc[:, 1].div(20)
    
    data[gender_column] = data[gender_column].replace("Other", np.nan)
    data[gender_column] = data.apply(lambda row: ("Man" if row["Gender Role"] >= 0 else "Woman") 
                                     if pd.isnull(row[gender_column]) 
                                     else row[gender_column], axis=1)
    
    # categorical multiple choices
    categorical_range = range(21, 26)
    categorical_range.append(27)
    categorical_range.append(48)
    categorical_range.append(52)
    
    for index in range(0, len(categorical_range)):
        i = categorical_range[index]
        data.iloc[:, i] = data.iloc[:, i].fillna(data.iloc[:, i].mode()[0])
        uniq, dictionary = stringToBinaryDict(data.iloc[:, i])
        split_columns_dictionary[split_columns[index]] = [split_columns[index] + ": " + s for s in uniq]
        data.iloc[:, i] = data.iloc[:, i].map(dictionary)
    
    # income question
    income_dictionary = {"Below $10,000": 0, "$10,000 - $30,000": 1, "$30,000 - $50,000": 2, "$50,000 - $70,000": 3,
                         "$70,000 - $90,000": 4, "Above $90,000": 5}
    income = data[income_column].map(income_dictionary)
    data[income_column] = (income - income.mean()) / income.std()
    
    # internet question
    internet_dictionary = {"less than 1 hour": 1, "1-3 hours" : 2, "3-4 hours": 3, "4 or more hours": 4}
    internet = data[internet_column].map(internet_dictionary)
    data[internet_column] = (internet - internet.mean()) / internet.std()
    
    # children question
    children_dictionary = {"I do not have children": 0, "1": 1, "2": 2, "More than 2": 3}
    children = data[children_column].map(children_dictionary)
    data[children_column] = (children - children.mean()) / children.std()
    
    # vehicle question
    vehicle_dictionary = {"Neither me nor my family owns a vehicle.": 0, "1": 1, "2": 2, "3 or more": 3}
    vehicle = data[vehicle_column].map(vehicle_dictionary)
    data[vehicle_column] = (vehicle - vehicle.mean()) / vehicle.std()
    
    # age question
    age_dictionary = {"Under 20 years of age": 1, "20 - 30 years of age": 2, "30 - 40 years of age": 3, 
                      "40 - 50 years of age": 4, "50 or more years of age": 5}
    age = data[age_column].map(age_dictionary)
    data[age_column] = (age - age.mean()) / age.std()
    
    # politics questions
    for i in range(28, 38):
        column = data.iloc[:, i].div(100)
        
        if i == 28:
            data.iloc[:, i] = column
        else:
            data.iloc[:, 28] = data.iloc[:, 28] + column
    
    data.iloc[:, 28] = data.iloc[:, 1].div(20)
    
    
    # select all apply questions
    data.iloc[:, 39:48] = data.iloc[:, 39:48].fillna(-1).replace(regex=r'^\w+', value=1)
        
    # joke ratings
    for i in range(53, 174):
        data.iloc[:, i] = data.iloc[:, i].div(100)

    # drop unnecessary columns
    data = data.drop(data.columns[29: 38], axis = 1)
    data = data.drop(data.columns[2: 21], axis = 1)
    data = data.drop(data.columns[119], axis = 1)

    #change column name
    data.rename(columns={ data.columns[11]: "Preferred Film Genre: Romance", data.columns[12]: "Preferred Film Genre: Horror", data.columns[13]: "Preferred Film Genre: Comedy",
			 data.columns[14]: "Preferred Film Genre: Drama", data.columns[15]: "Preferred Film Genre: Historical", data.columns[16]: "Preferred Film Genre: Animation",
			 data.columns[17]: "Preferred Film Genre: Documentary", data.columns[18]: "Preferred Film Genre: Adventure", data.columns[19]: "Preferred Film Genre: Fiction" }, inplace=True)
    rename_dict = dict(zip(data.columns[25:-1], ["Joke " + str(index) for index in range(1, len(data.columns)-25)]))
    print rename_dict
    data.rename(columns=rename_dict, inplace=True)
    data.rename(columns={ data.columns[5]: "Biological Gender", data.columns[7]: "Income Range", data.columns[10]: "Age Group", data.columns[21]: "Average time spent on Internet (in hours)",
			 data.columns[22]: "Number of children", data.columns[23]: "Number of vehicles owned" }, inplace=True)

    #reconstruct dataframe
    new_data = pd.DataFrame(data.iloc[:, 0:2], index = data.index)
    new_data[split_columns_dictionary["Region of Ancestry"]] = pd.DataFrame(data.iloc[:, 2].values.tolist(), index = data.index)
    new_data[split_columns_dictionary["Current Region of Occupancy"]] = pd.DataFrame(data.iloc[:, 3].values.tolist(), index = data.index)
    new_data[split_columns_dictionary["Ethnicity"]] = pd.DataFrame(data.iloc[:, 4].values.tolist(), index = data.index)
    new_data = new_data.join(data.iloc[:, 5])
    new_data[split_columns_dictionary["Education"]] = pd.DataFrame(data.iloc[:, 6].values.tolist(), index = data.index)
    new_data = new_data.join(data.iloc[:,7])
    new_data[split_columns_dictionary["Occupation"]] = pd.DataFrame(data.iloc[:, 8].values.tolist(), index = data.index)
    new_data = new_data.join(data.iloc[:,9:11])
    new_data = new_data.join(data.iloc[:,11:20])
    new_data[split_columns_dictionary["Preferred Music Genre"]] = pd.DataFrame(data.iloc[:, 20].values.tolist(), index = data.index)
    new_data = new_data.join(data.iloc[:, 21:24])
    new_data[split_columns_dictionary["Marital Status"]] = pd.DataFrame(data.iloc[:, 24].values.tolist(), index = data.index)
    new_data = new_data.join(data.iloc[:, 25:])


    return new_data

clean_data("data.csv").to_csv("cleaned.csv", index=False)
