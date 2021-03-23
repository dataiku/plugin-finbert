# Code for custom code recipe fb2021 (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *
from dataiku import recipe
input_dataset = recipe.get_inputs_as_datasets()[0]
output_dataset = recipe.get_outputs_as_datasets()[0]

# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
# my_variable = get_recipe_config()['parameter_name']

text_column = get_recipe_config()['text_column_name']

# For optional parameters, you should provide a default value in case the parameter is not present:
# my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import nltk
nltk.download('punkt')
from finbert.finbert import *
import finbert.utils as tools

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
#examples = dataiku.Dataset("input_A_datasets")
examples_df = input_dataset.get_dataframe()


# Compute recipe outputs from inputs

##load pretrained model from Huggingface with the trasnformers package

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def score_to_json(text, model):
    result = predict(text, model)
    result_dict = result.to_dict(orient = "index")
    avg_sent_score = result['sentiment_score'].mean()
    top_pred = result['prediction'].mode()[0]
    return result_dict, avg_sent_score, top_pred

examples_df['bert_result'],examples_df['bert_avg_sent_score'],examples_df['bert_avg_pred'] = zip(*examples_df[text_column].apply(lambda x: score_to_json(x, model)))


examples_sentiment_df = examples_df # For this sample code, simply copy input to output


# Write recipe outputs
#examples_sentiment = dataiku.Dataset("output_A_datasets")
output_dataset.write_with_schema(examples_sentiment_df)