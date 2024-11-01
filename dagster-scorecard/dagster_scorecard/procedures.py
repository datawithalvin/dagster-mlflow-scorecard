from sklearn.metrics import auc, roc_auc_score, precision_score, recall_score

def calc_credit_metrics(y_true, y_pred, y_prob):
    """Calculate beberapa  metrics essential untuk credit scoring"""
    from scipy.stats import ks_2samp
    
    # Calculate Gini
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    
    # Calculate KS statistic
    ks_stat = ks_2samp(y_prob[y_true==0], y_prob[y_true==1]).statistic
    
    # Calculate bad rate metrics
    precision_bad = precision_score(y_true, y_pred, pos_label=1)
    recall_bad = recall_score(y_true, y_pred, pos_label=1)
    
    return {
        "gini_coefficient": gini,
        "ks_statistic": ks_stat,
        "precision_bad_rate": precision_bad,
        "recall_bad_rate": recall_bad
    }
