# python 6_3-disjoint_NeuralLinUCB_experiment_results.py 2026-02-09 --validation_run 600
import pandas as pd
from tqdm import tqdm
from collections import Counter
import math, os
from sklearn.metrics import ndcg_score
import numpy as np
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=4, progress_bar=True)
import argparse, pickle

def load_popularity_dict(path):
    content_popularity_tag = pd.read_csv(path)
    content_popularity_tag.drop_duplicates(subset=['sub_title_id'], inplace = True)
    content_popularity_tag.reset_index(drop = True, inplace = True)
    content_popularity_dict = content_popularity_tag.set_index('sub_title_id')['popularity_tag'].to_dict()    
    return content_popularity_dict

def calculate_ndcg(group, predictions, labels, k_val = [1, 3, 5]):
    if len(group) < k_val[-1]:
        return {}
    
    y_true = np.asarray([group[labels].values.astype(float)])
    y_score = np.asarray([group[predictions].values.astype(float)])
    
    if len(np.unique(y_true)) == 1:
        return {f'NDCG@{k}': 0 for k in k_val}
            
    return {f'NDCG@{k}': ndcg_score(y_true, y_score, k=k) for k in k_val}

def popularity_dist_at_rank(df, score, k_val = [1, 3, 5]):
    df[score] = df[score].astype(float)
    df['rank'] = df.groupby(['dw_p_ids', 'timestamps'])[score].rank(ascending = False)
    
    popularity_dist_dict = {}
    for k in k_val:
        tmp = df[df['rank'] <= k].reset_index(drop = True)
        distribution_dict = tmp.groupby('popularity_category')['content_ids'].count().to_dict()
        popularity_dist_dict[f'Popularity distribution @ {k}'] = {k: round(100 * (v/sum(distribution_dict.values())), 2) for k, v in distribution_dict.items()}
    return popularity_dist_dict

def unique_catalog_at_rank(df, score, k_val = [1, 3, 5]):
    df[score] = df[score].astype(float)
    df['rank'] = df.groupby(['dw_p_ids', 'timestamps'])[score].rank(ascending = False)
    
    catalog_share_dict = {}
    for k in k_val:
        tmp = df[df['rank'] <= k].reset_index(drop = True)
        unique_catalog = tmp['content_ids'].nunique()
        catalog_share_dict[f'Catalog Length @ {k}'] = unique_catalog
    return catalog_share_dict
    
def perplexity_at_rank(df, score, k_val = [1, 3, 5]):
    df[score] = df[score].astype(float)
    df['rank'] = df.groupby(['dw_p_ids', 'timestamps'])[score].rank(ascending = False)
    
    perplexity_at_k_dict = {}
    for k in k_val:
        content_list = df[df['rank'] == k]['content_ids'].values
        p_dist = {k: v/len(content_list) for k,v in dict(Counter(content_list)).items()}
        cross_entropy = -sum([x*math.log2(x) for x in p_dist.values()])
        perplexity = 2**cross_entropy
        perplexity_at_k_dict[f'Perplexity @ {k}'] = perplexity
    return perplexity_at_k_dict

def evaluation_summary(df, alpha = 1, beta = 1, gamma = 0, k_val = [1, 3, 5]):
    print('*' * 80)
    print('Evaluation metrics creation started')
    df['new_score'] = alpha * df['deepFMpredictions'] + beta * df['variances'] + gamma * df['means']

    ndcg_results = result_df.groupby(['dw_p_ids', 'timestamps']).parallel_apply(
        lambda x: calculate_ndcg(x, predictions = 'new_score', labels = 'labels')
    ).reset_index(name=f'ndcg_score')
    ndcg_dict = pd.json_normalize(ndcg_results['ndcg_score']).mean().to_dict()
    
    print(f'NDCG for this approach : {ndcg_dict}')

    popularity_dist_dict = popularity_dist_at_rank(df.copy(), 'new_score', k_val = k_val)
    print(f'Popularity category distribution: {popularity_dist_dict}')
    
    catalog_dict = unique_catalog_at_rank(result_df, 'new_score', k_val = k_val)
    print(f'Catalog length distribution: {catalog_dict}')
    
    perplexity_at_k = perplexity_at_rank(result_df, 'new_score', k_val = k_val)
    print(f'Perplexity distribution: {perplexity_at_k}')
    print('*' * 80)
    return {
        "ndcg_results": ndcg_dict,
        "popularity_dist_dict": popularity_dist_dict,
        "catalog_dict": catalog_dict,
        "perplexity_at_k": perplexity_at_k
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TPFY Exploration Progressive Validation")
    parser.add_argument("date", type=str)
    parser.add_argument("--validation_run", default=600, type=int)
    parser.add_argument("--checkpoint", default='1770723470', type=str)

    args = parser.parse_args()

    result_df = []
    valdiation_metrics_dict = {}

    date = args.date
    
    for run in tqdm(range(0, args.validation_run, 100)):
        result_dict = pd.read_pickle(f'disjoint_neural_linucb/disjoined_neural_linUCB_offline_matrices_{args.date}_{args.checkpoint}/validation_dumping_dict_{args.date}/validation_stats_run_{run}.pkl')
        df = pd.DataFrame.from_dict(result_dict, orient='index').T
        result_df.append(df)

    result_df = pd.concat(result_df).reset_index(drop = True)
    content_popularity_dict = load_popularity_dict('Content_popularity_tag_26Jan_to_1Feb.csv')
    result_df['popularity_category'] = result_df['content_ids'].apply(lambda x: content_popularity_dict.get(int(x)))
    result_df['popularity_category'] = result_df['popularity_category'].fillna('Unmapped')

    print("Unique userids in the validation sample: ", result_df['dw_p_ids'].nunique())

    result_df.dropna(inplace = True)

    tqdm.pandas()

    mu_variances = result_df['variances'].mean()
    sigma_variances = result_df['variances'].std()

    mu_means = result_df['means'].mean()
    sigma_means = result_df['means'].std()
    
    mu_deepFMscore = result_df['deepFMpredictions'].mean()
    sigma_deepFMscore = result_df['deepFMpredictions'].std()

    print('Mean and std of deepFM score: ', (mu_deepFMscore, sigma_deepFMscore))

    result_df['variances'] = result_df['variances'].progress_apply(lambda x: ((x - mu_variances) / sigma_variances))
    result_df['means'] = result_df['means'].progress_apply(lambda x: ((x - mu_means) / sigma_means))
    result_df['deepFMpredictions'] = result_df['deepFMpredictions'].progress_apply(lambda x: ((x - mu_deepFMscore) / sigma_deepFMscore))

    deepFM_valid_results = evaluation_summary(result_df, alpha=1, beta=0)
    explo_valid_results = evaluation_summary(result_df, alpha=0, beta=1)
    means_valid_results = evaluation_summary(result_df, alpha=0, beta=0, gamma = 1)
    balanced_deepfm_valid_results = evaluation_summary(result_df, alpha=1, beta=1)
    balanced_lincub_valid_results = evaluation_summary(result_df, alpha=0, beta=1, gamma = 1)

    valdiation_metrics_dict['unique_users'] = result_df['dw_p_ids'].nunique()
    valdiation_metrics_dict['deepFM_params'] = [mu_deepFMscore, sigma_deepFMscore]
    valdiation_metrics_dict['variances_params'] = [mu_variances, sigma_variances]
    valdiation_metrics_dict['deepFM_valid_results'] = deepFM_valid_results
    valdiation_metrics_dict['explo_valid_results'] = explo_valid_results
    valdiation_metrics_dict['means_valid_results'] = means_valid_results
    valdiation_metrics_dict['balanced_deepfm_valid_results'] = balanced_deepfm_valid_results
    valdiation_metrics_dict['balanced_lincub_valid_results'] = balanced_lincub_valid_results

    os.makedirs(f'disjoint_neural_linucb/disjoined_neural_linUCB_offline_matrices_{args.date}_{args.checkpoint}/validation_dumping_dict_{args.date}/validation_metrics', exist_ok=True)

    with open(f'disjoint_neural_linucb/disjoined_neural_linUCB_offline_matrices_{args.date}_{args.checkpoint}/validation_dumping_dict_{args.date}/validation_metrics/valid_metrics_{date}', 'wb') as handle:
        pickle.dump(valdiation_metrics_dict, handle)
