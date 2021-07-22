# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 18:00:00 2021

@author: anand
"""

##############################################################################
# Importing the Libraries
##############################################################################
import pandas as pd
import webbrowser
import pickle
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import random

##############################################################################
# Declaring the Global Variables
##############################################################################
project_name = None
app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP])

##############################################################################
# Defining my Functions
##############################################################################
# Function to create the pie chart of review analysis
def sec_1_reviews_analysis():
    global scrapped_reviews
    scrapped_reviews = pd.read_csv("amazon_jewlery_reviews_sentiment_analysed.csv")
    reviews = scrapped_reviews.Sentiment.value_counts()
    return reviews[0], reviews[1]
    
# Function to create the Word Cloud
def sec_2_generate_word_cloud():
    # Python program to generate WordCloud
    # Selecting first 2000 reviews for reducing computational burden
    reviews_wc = scrapped_reviews.iloc[:2000,:]
    reviews_wc = reviews_wc.review_body.to_list()

    string = ""
    
    for i in reviews_wc:
        string = string + i
    string = string.lower()
    
    stopwords = set(STOPWORDS)
    cloud = WordCloud(width = 800, height = 400,
                      background_color = 'white',
                      stopwords = stopwords,
                      min_font_size = 10).generate(string)
    cloud.to_file('assets/wordcloud.png')

# Function to Generate 1000 random dropdown reviews
def sec_3_generate_dropdown_values():
    global dropdown_values_1000
    dropdown_values_1000 = {}
    # Selecting first 10000 reviews for reducing computational burden
    reviews_for_dropdown = scrapped_reviews.iloc[:100000,:]
    # Generating 1000 random values between 0 and 100000
    random_numbers = random.sample(range(0, 100000), 1000)
    
    for i in random_numbers:
        dropdown_values_1000[reviews_for_dropdown['review_body'][i]] = reviews_for_dropdown['review_body'][i]
    dropdown_values_1000 = [{'label' : key, 'value' : values} for key, values in dropdown_values_1000.items()]
    
# Function to load the model
def load_model():
    global pickle_model
    model_file = open("pickle_model.pkl", "rb")
    pickle_model = pickle.load(model_file)
    
    global vocab 
    vocab_file = open("features.pkl", "rb")
    vocab = pickle.load(vocab_file)

# Check the review and apply prediction
def check_review(review_text):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error = "replace", vocabulary = vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([review_text]))
    
    return pickle_model.predict(vectorised_review)

# Automating the tab opening process
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8080/")
    
# Create the app interface
def create_app_ui():
    # Analyse the reviews and save the positive and negative reviews numbers
    neg, pos = sec_1_reviews_analysis()
    
    # Create the WordCloud and Save it in png format
    sec_2_generate_word_cloud()
    
    # Generate 1000 random reviews for dropdown
    sec_3_generate_dropdown_values()
    
    # Start building the layout
    main_layout = html.Div(
        [
            # Create the Heading
            html.H1(id='Main_title', children = "Sentiment Analysis with Insights",style={'text-align':'center'}),
            
            #######################################################################################################
            # Section-1: Creating the Pie Chart and Displaying it
            #######################################################################################################
            
            # Create the Horizontal line
            html.Hr(style={'background-color':'black'}),
            
            # Create the Pie Chart for reveiw analysis
            html.H2(id = 'pie-chart', children = 'Pie Chart', style = {'text-align' : 'center', 'text-decoration' : 'underline'}),
            html.P(
                dcc.Graph(
                    id = 'reviews-bar-graph',
                    figure = go.Figure(
                        data = [
                            go.Pie(
                                labels = ['Negative', 'Positive'],
                                values = [neg, pos],
                                marker = dict(
                                    colors = ['#FF0000', '#00FF00']
                                )
                            )
                        ]
                    ), style = {'width':'100%','height':'100%', 'text-align' : 'center'}              
                ),
                style = {'text-align' : 'center'}
            ),
                        
            # Create the Horizontal line
            html.Hr(style={'background-color':'black'}),
            
            #######################################################################################################
            # Section-2: Creating the Word Cloud and Displaying it
            #######################################################################################################
            # Create the WordCloud
            html.H2(
                id = 'word-cloud', 
                children = 'Word Cloud', 
                style = {'text-align' : 'center', 'text-decoration' : 'underline'}
            ),
            html.Br(), 
            html.P(
                [
                    html.Img(
                        src=app.get_asset_url('wordcloud.png'),
                        style={'width':'700px','height':'400px'}
                    )
                ],
                style={'text-align':'center'}
            ),
            html.Hr(style={'background-color':'black'}),
            
            #######################################################################################################
            # Section-3: Show 1000 random reviews in Dropdown selector format
            #######################################################################################################
            html.H2(
                children = 'Select a Review',
                style = {'text-align':'center','text-decoration':'underline'}
            ),
            html.Br(),
            # Creating the Dropdown
            dcc.Dropdown(
                id = 'dropdown-1000',
                options = dropdown_values_1000,
                placeholder = 'Select a Review',
                style = {'font-size':'22px','height':'70px'}
            ),
            html.Br(),
            html.H3(
                id='dropdown-sentiment',
                children = 'Missing',
                style={'text-align':'center'}
            ),
            html.Hr(style={'background-color':'black'}),
            
            #######################################################################################################
            # Section-4: Taking a review text from customer and analysing it
            #######################################################################################################
            html.H2(
                children = 'Generate the Sentiment of Your Review',
                style = {'text-align':'center','text-decoration':'underline'}
            ),
            html.Br(),
            # Creating the Text input
            dcc.Textarea(
                id = 'textarea_review',
                placeholder = 'Enter Your Review here...',
                style = {'width' : '100%', 'height' : 150, 'font-size' : '22px'}
            ),
            # Create the submit button
            dbc.Button(
                id = 'submit_review',
                children = 'Analyse my Review',
                style = {'width' : '100%'}
            ),
            html.H3(
                id = 'button-sentiment',
                children = 'None',
                style = {'text-align' : 'center'}
            )            
        ]
    )
    
    return main_layout

# Callbak mechanism for dropdown
@app.callback(
    Output('dropdown-sentiment', 'children'),
    [Input('dropdown-1000', 'value')]
)
def update_dropdown_sentiment(dropdown_review):
    sentiment = []
    if dropdown_review:
        if (check_review(dropdown_review) == 0):
            sentiment = 'Negative'
        elif (check_review(dropdown_review) == 1):
            sentiment = 'Positive'
        else:
            sentiment = 'None'
    else:
        sentiment = 'None'
        
    return sentiment

# Callbak mechanism for textarea    
@app.callback(
    Output('button-sentiment', 'children'),
    [Input('submit_review', 'n_clicks')],
    [State('textarea_review', 'value')]
)
def update_textarea_sentiment(n_clicks, textarea_value):    
    if (n_clicks > 0):       
        response = check_review(textarea_value)
        if (response[0] == 0):
            result = 'Negative'
        elif (response[0] == 1):
            result = 'Positive'
        else:
            result = 'Unknown'
        return result
    else:
        return ""
    
##############################################################################
# Main Function to control the flow of Project
##############################################################################
def main():
    print("Starting of the Project...")
    
    # Loaidng the model
    load_model()
    
    # Open the link
    open_browser()
        
    # Declaring the Global Values
    global project_name
    global app
    
    project_name = "Sentiment Analysis with Insights"
    
    # APP details
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server(port = "8080")
    
    print("End of the Project")
    
    # Reverting back variables values to None
    project_name = None
    scrappedReviews = None
    app = None
    dropdown_values_1000 = None
    pickle_model = None
    vocab = None

##############################################################################
# Calling the main Function
##############################################################################
if __name__ == '__main__':
    main()