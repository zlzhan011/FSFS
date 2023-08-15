from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import pandas as pd


from pathlib import Path


home = str(Path.home())
home = ''
capuchin_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/capuchin'
results_path = Path(home + '/Complexity-Driven-Feature-Construction/results')
results_path.mkdir(parents=True, exist_ok=True)

COMPAS_path = home + '/Finding-Fair-Representations-Through-Feature-Construction/data/compas-analysis'

df = pd.read_csv(COMPAS_path + '/compas-scores-two-years.csv')

df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
                    'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]
ix = df['days_b_screening_arrest'] <= 30
ix = (df['days_b_screening_arrest'] >= -30) & ix
ix = (df['is_recid'] != -1) & ix
ix = (df['c_charge_degree'] != "O") & ix
ix = (df['score_text'] != 'N/A') & ix
df = df.loc[ix, :]
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-pd.to_datetime(df['c_jail_in'])).apply(lambda x: x.days)

dfcut = df.loc[~df['race'].isin(['Native American', 'Hispanic', 'Asian', 'Other']), :]

dfcutQ = dfcut[['sex', 'race', 'age_cat', 'c_charge_degree', 'score_text', 'priors_count', 'is_recid',
                'two_year_recid', 'length_of_stay']].copy()


# Quantize priors count between 0, 1-3, and >3
def quantizePrior(x):
    if x <= 0:
        return '0'
    elif 1 <= x <= 3:
        return '1 to 3'
    else:
        return 'More than 3'


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return '<week'
    if 8 < x <= 93:
        return '<3months'
    else:
        return '>3 months'


# Quantize length of stay
def adjustAge(x):
    if x == '25 - 45':
        return '25 to 45'
    else:
        return x


# Quantize score_text to MediumHigh
def quantizeScore(x):
    if (x == 'High') | (x == 'Medium'):
        return 'MediumHigh'
    else:
        return x


def generate_binned_df(df):
    df_ = df.copy()
    for i in list(df_):
        if i != target and df_[i].dtype in (float, int):
            out, bins = pd.qcut(df_[i], q=2, retbins=True, duplicates='drop')
            if bins.shape[0] == 2:
                out, bins = pd.cut(df_[i], bins=2, retbins=True, duplicates='drop')
            df_.loc[:, i] = out.astype(str)
    return df_


dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))
dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))
dfcutQ['score_text'] = dfcutQ['score_text'].apply(lambda x: quantizeScore(x))
dfcutQ['age_cat'] = dfcutQ['age_cat'].apply(lambda x: adjustAge(x))

features = ['race', 'age_cat', 'c_charge_degree', 'priors_count', 'is_recid']

# Pass vallue to df
COMPAS_binned = dfcutQ[features]
COMPAS = dfcut[features]

f1 = make_scorer(f1_score, greater_is_better=True, needs_threshold=False)

kf1 = KFold(n_splits=5, random_state=42, shuffle=True)

target = 'is_recid'
sensitive_feature = 'race'
inadmissible_feature = ''
protected = 'African-American'
sensitive_features = [sensitive_feature]
admissible_features = [i for i in list(COMPAS) if i not in sensitive_features and i != target]



