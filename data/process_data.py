# Data preprocessing
import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load csv data from specified paths and return the joined dataframe

    INPUT:
        messages_filepath: path to messages file
        categories_filepath: path to categories file

    OUTPUT:
        df: joined dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, how='outer', on='id')
    return df


def clean_data(df):
    '''
    Split categories into separate category columns.

    INPUT:
        df: data frame (raw data)

    OUTPUT:
        df: data frame (cleaned)
    '''

    # ceparate each value into its own column
    categories = df['categories'].str.split(';', expand=True)
    categories.columns = [x.split('-')[0] for x in categories.iloc[0]]
    df = df.drop('categories', axis=1)

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    # combine cleaned categories dataframe with the original
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    # "related" column has 3 categories (i.e. 0,1,2). Category 2 makes little sense here
    # so they are converted to 1
    df['related'] = df['related'].replace(2, 1)

    # child_alone column has only one categories (0)
    # so our model will always produce one value for it
    df.drop('child_alone', axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Save cleaned data to SQLite database using SQLAlchemy engine

    INPUT:
        df: data frame
        database_filename: path to save to database

    return: None
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    return


def main():
    if len(sys.argv) != 4:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')

    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')

    return


if __name__ == '__main__':
    main()
