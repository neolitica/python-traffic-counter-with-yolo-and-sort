import pandas as pd


class Storage:

    def __init__(self):
        self.data = pd.DataFrame(data={
            'time_stamp':[],
            'gender':[],
            'gender_conf':[],
            'age':[],
            'age_conf':[]
        })

    def save(self, timestamp, gender,age):
        if gender is None: gender = (None,None)
        if age is None: age = (None,None)
        self.data = self.data.append({
            'time_stamp':timestamp,
            'gender':gender[0],
            'gender_conf':gender[1],
            'age':age[0],
            'age_conf':age[1]
        }, ignore_index=True)
        
    def store(self,path):
        self.data.to_csv(path, index_label='id', float_format='%.3f')