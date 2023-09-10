import pandas as pd

file_names = ['./data/mug_simulation/grasp_score_0.01.csv',
              './data/mug_simulation/grasp_score_0.03.csv',
              './data/mug_simulation/grasp_score_0.05.csv']

# open the csv files and divide all data by 100:
for file_name in file_names:
    df = pd.read_csv(file_name, index_col=0)
    df = df / 100
    df.to_csv(file_name)