# Disaster Message Classifier
Disaster Messages (such as tweets, facebook posts, etc) are good indicators of people's need in emergency. The aim of emergency response is to provide immediate assistance to maintain life, improve health and support the morale of the affected population. Therefore, being able to categorize the messages to match people's demand brings about huge socio-economic benefits.
In this project, I will a simple web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display some visualizations of the dataset in general.

![Disaster Response](https://github.com/Hieuqng/Disaster-Message-Classifier/blob/master/images/front_page.jpg = 500x500)
<img src="https://github.com/Hieuqng/Disaster-Message-Classifier/blob/master/images/front_page.jpg" width="500" height="400" />


## Getting Started

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/nbsvm.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Prerequisites

What things you need to install the software and how to install them
1. sqlalchemy
2. nltk: download
3. wordcloud (if you want to run the notebook)
```
git clone https://github.com/amueller/word_cloud.git
cd word_cloud
pip install .
```
4. pickle
5. sklearn, pandas, numpy, plotly
6. flask

## Authors

* **Hieu Nguyen** - [Github](https://github.com/Hieuqng)

