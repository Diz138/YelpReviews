import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans


def convert_to_csv():
    # Business
    yelp_business = pd.read_json('data/yelp_academic_dataset_business.json', lines=True)
    filtered_business = yelp_business[yelp_business['review_count'] >= 50]
    # filtered_business.to_csv('yelp_filtered_business.csv')

    businesses = filtered_business['business_id'].values.tolist()

    # Check-in
    # yelp_checkin = pd.read_json('data/yelp_academic_dataset_checkin.json', lines=True)
    # filtered_checkin = yelp_checkin[yelp_checkin['business_id'].isin(businesses)]
    # filtered_checkin.to_csv('yelp_filtered_checkin.csv')

    # Tip
    # yelp_tip = pd.read_json('data/yelp_academic_dataset_tip.json', lines=True)
    # yelp_tip_grouped = yelp_tip.groupby('business_id')['text'].apply('|'.join).reset_index()
    # filtered_tip = yelp_tip_grouped[yelp_tip_grouped['business_id'].isin(businesses)]
    # filtered_tip.to_csv('yelp_filtered_tip.csv')

    # Open and convert review dataset
    # with open('data/yelp_academic_dataset_review.json', encoding='utf-8') as json_file:
    #     data = json_file.readlines()
    #     data = list(map(json.loads, data))
    #
    # yelp_review = pd.DataFrame(data)
    #
    # yelp_review_grouped = yelp_review.groupby('business_id')['text'].apply('|'.join).reset_index()
    # #filtered_review = yelp_review[yelp_review['business_id'].isin(businesses)]
    # filtered_review = yelp_review_grouped[yelp_review_grouped['business_id'].isin(businesses)]
    #
    # filtered_review.to_csv('yelp_filtered_review.csv')

    # Open business and filter by attribute count
    # yelp_business_filtered = pd.read_csv('data/yelp_filtered_business.csv')
    # pd.set_option('display.max_columns', 10)
    # pd.set_option('display.max_colwidth', 1000)
    # print(yelp_business_filtered.head(1)['attributes'])

    return


def load_data():
    # Define desired columns
    business_cols = ["business_id", "stars", "review_count"]
    review_cols = ["business_id", "text",
                   # "stars", "useful", "funny", "cool"
                   ]
    tip_cols = ["business_id", "text"]

    # Load files into dataframes
    yelp_review = pd.read_csv('data/yelp_filtered_review.csv', usecols=review_cols)
    yelp_business = pd.read_csv('data/yelp_filtered_business.csv', usecols=business_cols)
    yelp_checkin = pd.read_csv('data/yelp_filtered_checkin.csv', index_col=0)
    yelp_tip = pd.read_csv('data/yelp_filtered_tip.csv', usecols=tip_cols)
    # Need to use different method since the json is a bunch of different individual json objects
    # with open('data/yelp_academic_dataset_review.json', encoding='utf-8') as json_file:
    #     data = json_file.readlines()
    #     data = list(map(json.loads, data))
    #
    # yelp_review = pd.DataFrame(data)
    # yelp_tip = pd.read_json('data/yelp_academic_dataset_tip.json', lines=True)

    # print(yelp_review.info())
    # grouped_tips = yelp_tip.groupby(['business_id'])
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 1000)

    # Summarize info of reviews and businesses
    # business_agg = yelp_review.groupby('business_id').agg({'business_id': ['count'],
    #                                            'useful': ['sum'], 'funny': ['sum'], 'cool': ['sum'],
    #                                            'stars': ['mean']})
    # business_agg = business_agg.sort_values([('business_id', 'count')], ascending=False)
    # sorted_business = yelp_business.sort_values(by=['review_count'], ascending=False)
    # print("Top 10 Reviewed Businesses in Yelp")
    # print(business_agg.head(5))
    # print("Businesses sorted by review count")
    # print(sorted_business.head(5))
    #
    # print("Average reviews: ", yelp_business['review_count'].mean())
    # print("Minimum reviews: ", yelp_business['review_count'].min())
    # print("Maximum reviews: ", yelp_business['review_count'].max())

    # # Summarize info of check-ins
    # yelp_checkin['checkin_count'] = yelp_checkin['date'].apply(
    #     lambda text: text.count("-") / 2
    # )
    # yelp_checkin['first_visit'] = yelp_checkin['date'].apply(
    #     lambda text: text[0:11]
    # )
    # print("Average checkins: ", yelp_checkin['checkin_count'].mean())
    # print("Minimum checkins: ", yelp_checkin['checkin_count'].min())
    # print("Maximum checkins: ", yelp_checkin['checkin_count'].max())
    #
    # yelp_checkin_sorted = yelp_checkin.sort_values(by='date')
    # print(yelp_checkin['date'].head(5))
    # print(yelp_checkin_sorted['date'].head(5))

    # Combine
    print(yelp_checkin.info())
    print(yelp_business.info())
    print(yelp_tip.info())
    print(yelp_review.info())

    yelp_checkin['dates'] = yelp_checkin['date'].apply(
        lambda text: text.split(', ')
    )
    yelp_checkin = yelp_checkin.drop(['date'], axis=1)

    yelp_tip['tips'] = yelp_tip['text'].apply(
        lambda text: text.split('|')
    )
    yelp_tip = yelp_tip.drop(['text'], axis=1)

    yelp_review['reviews'] = yelp_review['text'].apply(
        lambda text: text.split('|')
    )

    yelp_review = yelp_review.drop(['text'], axis=1)

    # print(yelp_review.info())
    # print(yelp_review['reviews'].head(1))
    #
    # print(yelp_checkin.info())
    # print(yelp_checkin['dates'].head(1))
    #
    # date_ex = yelp_checkin.iloc[0]['dates']
    # print(date_ex[0])

    merged_dataframe = pd.merge(yelp_business, yelp_tip, how='left', on="business_id")
    merged_dataframe = merged_dataframe.rename(columns={"text": "tips"})
    merged_dataframe = pd.merge(merged_dataframe, yelp_checkin, how='left', on="business_id")
    merged_dataframe = pd.merge(merged_dataframe, yelp_review, how='left', on="business_id")

    print(merged_dataframe.info())
    print(merged_dataframe.head(1))
    merged_dataframe.to_pickle("./yelp_merged.pkl")

    # merged_dataframe.to_csv('yelp_filtered_merged.csv')


def lowest_date(x):
    # print(f"x: {x} type of x: {type(x)}")
    year = int(x[0][0:4])

    return year


def plot_opening_dates(df):
    plt.hist(df['opening'], edgecolor='black', linewidth=1.2, bins=np.arange(2009, 2023) - 0.5)
    plt.xticks(range(2009, 2023, 1))
    plt.xticks(rotation='vertical')
    plt.title("Opening Dates Histogram")
    plt.xlabel("Year")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.show()

    counts, bin_edges = np.histogram(df['opening'], bins=12)

    print(counts)
    print(bin_edges)


def plot_visits(df):
    # plt.hist(df['visits'], edgecolor='black', linewidth=1.2)
    # plt.xticks(range(2009, 2023, 1))
    # plt.xticks(rotation='vertical')
    # df_no_outliers = df[df['visits'] <= 15000]
    # df_pre_2015 = df_no_outliers[df_no_outliers['opening'] <= 2015]
    # df_post_2015 = df_no_outliers[df_no_outliers['opening'] > 2015]
    # plt.scatter(df['opening'], df['visits'], c="blue")
    # plt.xticks(range(2009, 2023, 1))
    # plt.xticks(rotation='vertical')
    # plt.title("Visits Histogram")
    # plt.xlabel("Opening Year")
    # plt.ylabel("Num Visits")
    # plt.tight_layout()
    # plt.show()

    # plt.scatter(df_pre_2015['opening'], df_pre_2015['visits'], c="blue")
    # plt.xticks(range(2009, 2016, 1))
    # plt.xticks(rotation='vertical')
    # plt.title("Visits Histogram pre-2016")
    # plt.xlabel("Opening Year")
    # plt.ylabel("Num Visits")
    # plt.tight_layout()
    # plt.show()
    #
    # plt.scatter(df_post_2015['opening'], df_post_2015['visits'], c="blue")
    # plt.xticks(range(2016, 2023, 1))
    # plt.xticks(rotation='vertical')
    # plt.title("Visits Histogram post-2015")
    # plt.xlabel("Opening Year")
    # plt.ylabel("Num Visits")
    # plt.tight_layout()
    # plt.show()

    plt.scatter(df['opening'], df['visits_normalized'], c="blue")
    plt.xticks(range(2009, 2023, 1))
    plt.xticks(rotation='vertical')
    plt.title("Visits Scatter Plot")
    plt.xlabel("Opening Year")
    plt.ylabel("Num Visits Normalized by Year")
    plt.tight_layout()
    plt.show()



def evaluate_merged():
    yelp_merged = pd.read_pickle("data/yelp_merged.pkl")

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 1000)

    yelp_merged_dropped = yelp_merged.dropna(axis='rows')

    # print(yelp_merged.info())
    print(yelp_merged_dropped.info())
    # print(yelp_merged['tips'].head(1))


    yelp_merged_dropped['opening'] = yelp_merged_dropped['dates'].apply(
        lambda x: lowest_date(x)
    )
    yelp_merged_dropped['age'] = yelp_merged_dropped['opening'].apply(
        lambda x: 2022 - x
    )
    yelp_merged_dropped['visits'] = yelp_merged_dropped['dates'].apply(
        lambda x: len(x)
    )
    yelp_merged_dropped['visits_normalized'] = yelp_merged_dropped['visits'] / yelp_merged_dropped['age']

    print(yelp_merged_dropped['opening'].head(3))
    print(yelp_merged_dropped['age'].head(3))

    print(yelp_merged_dropped.info())
    print("Earliest opening ", yelp_merged_dropped['opening'].min())
    print("Latest opening ", yelp_merged_dropped['opening'].max())
    print("Minimum visits ", yelp_merged_dropped['visits'].min())
    print("Maximum visits ", yelp_merged_dropped['visits'].max())
    print("Average visits ", yelp_merged_dropped['visits'].mean())

    print("Minimum normalized visits ", yelp_merged_dropped['visits_normalized'].min())
    print("Maximum normalized visits ", yelp_merged_dropped['visits_normalized'].max())
    print("Average normalized visits ", yelp_merged_dropped['visits_normalized'].mean())
    # print(yelp_merged_dropped['dates'].head(5))
    # print(yelp_merged_dropped['opening'].head(5))

    #yelp_merged_dropped.to_pickle("./yelp_merged_filtered.pkl")

    return yelp_merged_dropped


def get_clusters(df):
    data = df['visits_normalized'].values
    km = KMeans(n_clusters=3)
    km.fit(data.reshape(-1, 1))
    print(km.cluster_centers_)



if __name__ == '__main__':
    # convert_to_csv()
    # load_data()
    df = evaluate_merged()
    # plot_opening_dates(df)
    plot_visits(df)
    get_clusters(df)
