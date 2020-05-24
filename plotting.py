
from bokeh.plotting import figure, show,output_file
import pandas as pd
df = pd.read_csv("NN1.csv")
df1 = df.copy()
df1.loc[4] = [df.iloc[1,0]-df.iloc[2,0],df.iloc[1,1]-df.iloc[2,1],df.iloc[1,2]-df.iloc[2,2], df.iloc[1,3]-df.iloc[2,3], df.iloc[1,4]-df.iloc[2,4]]
df1.loc[5] = [df1.iloc[4,0]/df1.iloc[1,0]*100,df1.iloc[4,1]/df1.iloc[1,1]*100,df1.iloc[4,2]/df1.iloc[1,2]*100,df1.iloc[4,3]/df1.iloc[1,3]*100,df1.iloc[4,4]/df1.iloc[1,4]*100]

# df1.loc[5] = [df.iloc[4,0]/df.iloc[1,0]*100,df.iloc[4,1]/df.iloc[1,1]*100,df.iloc[4,2]/df.iloc[1,2]*100,df.iloc[4,3]/df.iloc[1,3]*100,df.iloc[4,4]/df.iloc[1,4]*100]# adding a row
# df1.index = df1.index + 1  # shifting index
# df1 = df1.sort_index()

output_file("line.html", title="question 1 k=1, error rate vs number of training sets")

p = figure(plot_width=800, plot_height=400)
p.xaxis.axis_label = "number of training samples"
p.yaxis.axis_label = "Error rate %"
# add a line renderer
p.line(df1.iloc[0], df1.iloc[5], line_width=2)

show(p)
