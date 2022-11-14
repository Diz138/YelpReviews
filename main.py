import pandas as pd
import json


def convert_to_csv():
    # Business
    # yelp_business = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)
    # filtered_business = yelp_business[yelp_business['review_count'] >= 50]
    # filtered_business.to_csv('yelp_filtered_business.csv')

    # businesses = filtered_business['business_id'].values.tolist()

    # Open and convert rule dataset
    # with open('data/yelp_academic_dataset_review.json', encoding='utf-8') as json_file:
    #     data = json_file.readlines()
    #     data = list(map(json.loads, data))
    #
    # yelp_review = pd.DataFrame(data)
    #
    # filtered_review = yelp_review[yelp_review['business_id'].isin(businesses)]
    #
    # filtered_review.to_csv('yelp_filtered_review.csv')

    # Open business and filter by attribute count
    yelp_business_filtered = pd.read_csv('data/yelp_filtered_business.csv')
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 1000)
    print(yelp_business_filtered.head(1)['attributes'])

    return

def load_data():

    # Define desired columns
    business_cols = ["business_id", "stars", "review_count"]
    review_cols = ["business_id", "stars", "useful", "funny", "cool"]
    # tip_cols = ["business_id", "text"]

    # Load files into dataframes
    yelp_review = pd.read_csv('data/yelp_filtered_review.csv', usecols=review_cols)
    yelp_business = pd.read_csv('data/yelp_filtered_business.csv', usecols=business_cols)
    # Need to use different method since the json is a bunch of different individual json objects
    # with open('data/yelp_academic_dataset_review.json', encoding='utf-8') as json_file:
    #     data = json_file.readlines()
    #     data = list(map(json.loads, data))
    #
    # yelp_review = pd.DataFrame(data)
    #yelp_tip = pd.read_json('data/yelp_academic_dataset_tip.json', lines=True)

    #print(yelp_review.info())
    # grouped_tips = yelp_tip.groupby(['business_id'])
    pd.set_option('display.max_columns', 10)

    # user_agg = reviews.groupby('user_id').agg({'review_id': ['count'], 'date': ['min', 'max'],
    #                                            'useful': ['sum'], 'funny': ['sum'], 'cool': ['sum'],
    #                                            'stars': ['mean']})

    business_agg = yelp_review.groupby('business_id').agg({'business_id': ['count'],
                                               'useful': ['sum'], 'funny': ['sum'], 'cool': ['sum'],
                                               'stars': ['mean']})
    business_agg = business_agg.sort_values([('business_id', 'count')], ascending=False)
    sorted_business = yelp_business.sort_values(by=['review_count'], ascending=False)
    print("Top 10 Reviewed Businesses in Yelp")
    print(business_agg.head(5))
    print("Businesses sorted by review count")
    print(sorted_business.head(5))

if __name__ == '__main__':
    #convert_to_csv()
    load_data()