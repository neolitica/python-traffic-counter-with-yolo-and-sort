import pandas as pd

def data_init():
    return pd.DataFrame(data={
        'time_stamp':[],
        'gender':[],
        'age':[]
    })

def save(df, timestamp, gender,age):
    return df.append({
        'time_stamp':timestamp,
        'gender':gender,
        'age':age,
    }, ignore_index=True)
    
def store(df,path):
    df.to_csv(path, index_label='id')