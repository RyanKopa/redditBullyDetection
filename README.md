# Reddit Bully Detection

Algorithm that uses natural language processing (NLP) and machine learning (ML) to 
determine if a comment made on social media (specifically reddit).
Data was gathered from Kaggle's Reddit May 2015 hosted data off of a sql server which 
can be found here: https://www.kaggle.com/reddit/reddit-comments-may-2015.
Required packages: numpy, pandas, sklearn (jupiter notebook for .ipynb)

The redditBullies.py script was my first model developed using csv files I downloaded from the kaggle
hosted Reddit SQL database.  I used a decission tree classifier model to determine whether a post was
considered bullying or not (using its controversiality metric).  This resulted in a roughly __53%__ accuracy
(*yay for beating random guessing!*).

The redditBullyDetection.py script was the second model I made with the help of Dave Fernig's script, 
"the lowest form of sarcasm."  Being inspired from his way of developing his corpus for NLP and 
analyzing feature importance in a model, I continued with his way and used a logistic regression
model to obtain about __67%__ accuracy (*yay for being a lot better than random guessing!*).

Future direction for this project will be to develop a webapp or web browser extension to host this model
for people to test if their comments on the internet are hateful or not.  Also, users will be able to
recognize when their friends are being bullied as well, and seek the appropriate help.
