# collaborative_filtering.py

import pandas as pd
import numpy as np
from surprise import dump
from collections import defaultdict

def collaborativefiltering_recommendations(user_id, filtered_df, file_name='CoClustering_tuned', k=10):
    _, loaded_algo = dump.load(file_name)
    if not loaded_algo:
        raise RuntimeError("추천시스템 로드 실패")

    all_pred_scores = []
    for r_id in filtered_df['route_id']:
        pred = loaded_algo.predict(user_id, r_id)
        similarity_score = float(filtered_df.loc[filtered_df['route_id'] == r_id]['similarity'])
        all_pred_scores.append({'route_id': r_id, 'recom_score': pred.est, 'total_similarity_score': similarity_score + pred.est})
    
    result_df = pd.DataFrame(all_pred_scores)
    total_df = pd.merge(filtered_df, result_df, on='route_id')

    total_df.sort_values(by='total_similarity_score', ascending=False, inplace=True)
    return total_df.iloc[:k]

