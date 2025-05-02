import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# just fake data for testing
testData = {
	'A': [1, 2, 3, 4],
    'B': [25, 13, 36, 39], 
    'C': [12, 7, 18, 20]
}
# create a DataFrame from the testData
df = pd.DataFrame(testData)

# create a figure 
fig = plt.figure()

# create a subplot which is a 1x1 grid placed at 1st position)
ax = fig.add_subplot(1,1,1)

# create a list of the positions of the bars
pos = list(range(len(df['A'])))
# set the width of the bars
width=0.3

# create the bars
ax.bar(pos, df['B'], width,alpha=0.5,color="blue",label=df['A'])

# define the title of the figure
plt.title(" Total profit ")

# name the x-axis
plt.xlabel('Quarter')

# create a space beteen the bars
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticks(pos)
ax.set_xticklabels(df['A'])

# name the y-axis
plt.ylabel('Total Profit')

##############################################
# create a second list of the positions of the bars
pos = list(range(len(df['A']))) 
width = 0.15

# print the positions
print(pos)

# draw both figures
fig, ax=plt.subplots(figsize=(10,5))
print(type(ax))

# create the bars, one for B and one for C, the B is yellow and the C is green
plt.bar(pos,df['B'], width,alpha=0.5,color="yellow",label='B') 
plt.bar([p + width for p in pos],df['C'], width,alpha=0.5,color="green",label='C') 

# set the label of the y-axis
ax.set_ylabel('Sales')

# set the title of the figure
ax.set_title('Sales of all different items per quarter')

# define the position of the x-axis
ax.set_xticks([p + 1.5 * width for p in pos])

# set the name of the x-axis
ax.set_xticklabels(df['A'])
       
# create a legend to understand the colors
plt.legend(loc= "upper left")

# show the figures
plt.show()

#######################"

# define a new chart in type of bar chart, the x-axis is A and the y-axis is B
df.plot(x='A', y='B', kind='bar', rot=45)

# define a new chart in type of bar chart, the x-axis is A and the y-axis is B, with two colors red and green
df.plot(x='A', kind='bar', rot=45, color=['red','green'])

# define a new chart in type of bar chart, the x-axis is A and the y-axis is B, with stacked bars
df.plot(x='A', kind='bar', rot=45, stacked=True)

# show the figures
plt.show()

#######################"

# same charts as the previous ones but these are lines and not bars
df.plot(x='A', y='B', kind='line', rot=45)
df.plot(x='A', kind='line', rot=45, color=['red','green'])
df.plot(x='A', kind='line', rot=45, stacked=True)

plt.show()

#######################

# same charts as the previous ones but these are horizontal bars and not bars
df.plot(x='A', y='B', kind='barh', rot=45)
df.plot(x='A', kind='barh', rot=45, color=['red','green'])
df.plot(x='A', kind='barh', rot=45, stacked=True)

plt.show()

# #######################"
# same charts as the previous ones but these are boxplots and not bars
df.boxplot(column='C', by='A', rot=45)
# same charts but this is a scatter plot (points)
df.plot(x='A', y='B', kind='scatter', rot=45)
# # same chart based on points but with different colors
df.plot(x='A', y='B', kind='scatter', rot=45, color='red')
# # same chart based on points but with stacked points
df.plot(x='A', y='B', kind='scatter', rot=45, stacked=True)

# from the b column, create a new dataframe
subDf=df['B']
# create a bar chart from the new dataframe in bar type but without the index 
subDf.plot(kind='bar', use_index=False)
plt.show()


#######################"
# same chart based on points but with a pie chart
df.plot(x='A', y='B', kind='pie', rot=45)
# same chart based on points but with a pie chart and two columns
df.plot(x='A', kind='pie', subplots=True, rot=45)
# same chart based on points but with a pie chart and two columns and labels
df.set_index('A')[['B', 'C']].plot(kind='pie', subplots=True, labels=df['A'])
# same chart based on points but with a pie chart and two columns and labels and without legend and with percentage values
df.set_index('A')[['B', 'C']].plot(kind='pie', subplots=True, labels=df['A'], legend=False, autopct='%1.1f%%', rot=45)


plt.show()
# ###################################
#Essayer 5 plots parmi d'autres plots : line, barh, hist, scatter, pie, a rea, box, etc
df.plot(x='A', kind='hist', rot=45, stacked=True)
df.plot(x='A', y='B', kind='scatter', rot=45, stacked=True)
df.plot(x='A', kind='pie', rot=45, subplots=True)
df.plot(x='A', kind='area', rot=45, stacked=True)
df.plot(x='A', kind='box', rot=45, stacked=True)

plt.show()