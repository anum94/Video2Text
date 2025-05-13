import pandas as pd
import matplotlib.pyplot as mp

file_path = "/Users/anumafzal/PycharmProjects/video2Text/utils/plot_data/step_coorelation.csv"
df = pd.read_csv(file_path)

#print (df.columns)
series = dict(df[(df['# frame'] == 1) & (df['k'] == 6) & (df['step'] == 2)].iloc[0])

x = [i for i in range(10)]
baseline_rouge_columns = ['baseline_0-10%', 'baseline_10-20%', 'baseline_20-30%',
       'baseline_30-40%', 'baseline_40-50%', 'baseline_50-60%',
       'baseline_60-70%', 'baseline_70-80%', 'baseline_80-90%',
       'baseline_90-100%',]
feedback_rouge_columns = ['feedback_0-10%',
       'feedback_10-20%', 'feedback_20-30%', 'feedback_30-40%',
       'feedback_40-50%', 'feedback_50-60%', 'feedback_60-70%',
       'feedback_70-80%', 'feedback_80-90%', 'feedback_90-100%',
       ]
icl_rouge_columns = ['icl_0-10%', 'icl_10-20%', 'icl_20-30%',
       'icl_30-40%', 'icl_40-50%', 'icl_50-60%', 'icl_60-70%', 'icl_70-80%',
       'icl_80-90%', 'icl_90-100%',]
baseline_rouge_intervals = [series[col] for col in baseline_rouge_columns]
feedback_rouge_intervals = [series[col] for col in feedback_rouge_columns]
icl_rouge_intervals = [series[col] for col in icl_rouge_columns]

df = pd.DataFrame()
df['x'] = x
df['baseline_rouge_intervals'] = baseline_rouge_intervals
df['feedback_rouge_intervals'] = feedback_rouge_intervals
df['icl_rouge_intervals'] = icl_rouge_intervals


df.plot(x="x", y=["baseline_rouge_intervals", "feedback_rouge_intervals", "icl_rouge_intervals"],
        kind="line", figsize=(10, 10))

# display plot
mp.show()
print()



