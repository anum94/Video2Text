import pandas as pd
import matplotlib.pyplot as mp

file_path = "/Users/anumafzal/PycharmProjects/video2Text/utils/plot_data/step_coorelation.csv"
df = pd.read_csv(file_path)

#
print (df.columns)
df = df[df['# frame'] == 1]
# plot multiple columns such as population and year from dataframe
df.plot(x="step", y=["baseline_correlation", "icl_correlation"],
        kind="line", figsize=(10, 10))

# display plot
mp.show()
print()

