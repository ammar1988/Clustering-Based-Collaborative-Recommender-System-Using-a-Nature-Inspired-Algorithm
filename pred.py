import numpy as np
def predict(clusters_rate, test_actual):

    #build prediction dataframe
    test_pred = test_actual.drop(['rating','relevent'], axis=1)
    test_pred['rating'] = ""
    test_pred['relevent'] = ""

    #fill (rating, relevent) prediction dataframe
    for index, row in test_pred.iterrows():    
        user_rate = test_pred.iloc[index]
        user_id = user_rate["user_id"]
        movie_id = user_rate["movie_id"]
        user_cluster = clusters_rate.loc[clusters_rate['user_id'] == user_id, 'cluster'].iloc[0]
        users_in_cluster = clusters_rate.loc[(clusters_rate['cluster'] == user_cluster) & (clusters_rate['movie_id'] == movie_id)] ##get all users in target cluster
        movie_rates = users_in_cluster["rating"].to_numpy()
        if movie_rates.size == 0:
            av = 3
        else:
            av = np.average(movie_rates)
        test_pred.iloc[index, 2] = av
        
        if av >= 3.5:
            test_pred.iloc[index, 3] = '1'
        else:
            test_pred.iloc[index, 3] = '0'
    return test_pred