import pandas as pd

def data_init():
    return pd.DataFrame(data={
        'time_stamp':[],
        'gender':[],
        'gender_conf':[],
        'age':[],
        'age_conf':[]
    })

def save(df, timestamp, gender,age):
    if gender is None: gender = (None,None)
    if age is None: age = (None,None)
    return df.append({
        'time_stamp':timestamp,
        'gender':gender[0],
        'gender_conf':gender[1],
        'age':age[0],
        'age_conf':age[1]
    }, ignore_index=True)
    
def store(df,path):
    df.to_csv(path, index_label='id')