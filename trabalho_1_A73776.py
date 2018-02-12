#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 00:27:44 2017

@author: luislima
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import numpy.random as rnd


# Importing the dataset
dataset = pd.read_csv('dataset.csv')

# Describe dataset
dataset.describe()


# Create dataframes and vars

# Create a dataframe with Employment and Mean Wage 
d = {'Employment': dataset['Employment'], 'Mean Wage': dataset['Mean Wage']}
dfEmplotmentMeanWage = pd.DataFrame(data=d)

# This dataframe group the data per Area Name
byarea = dataset.groupby('Area Name').mean()
# Area Name |  Eployment Mean Wage |  Entry Wage | Experienced Wage  

# This var is float type to separate Mean Wage
byareas_mean_wage = byarea.iloc[:, 1].values
# This var is float type to separate Employment
byareas_Employment = byarea.iloc[:, 0].values

# Create a dataframe with Employment and Mean Wage 
d = {'Employment': byareas_Employment, 'Mean Wage': byareas_mean_wage}
dfByArea = pd.DataFrame(data=d)
# Create a dataframe for Mean Wage colmn
d = {'Mean Wage': byareas_mean_wage}
dfMeanWage = pd.DataFrame(data=d)

# filter data for one Occupational Title 
byOcu_CP =dataset[(dataset['Occupational Title'] == 'Computer Programmers')]
byOcu_ISA =dataset[(dataset['Occupational Title'] == 'Information Security Analysts')]

y = byOcu_CP.iloc[:, 2].values
x = byOcu_CP.iloc[:, 3].values

## Report Point -> 1.4
# Create a plot fig 1 

plt.rcdefaults()
fig, ax = plt.subplots()

regions = byarea.index
y_pos = np.arange(len(byarea.index))

ax.barh(y_pos, byareas_mean_wage, align='center',
        color='#75bbfd', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(regions)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Mean Wage')
ax.set_title('Averages annual salaries by region')


# More plots of dataset n
# correlation
 
# with regression
sns.pairplot(dfEmplotmentMeanWage, kind="reg")
plt.show()
 
# without regression
sns.pairplot(dfEmplotmentMeanWage, kind="scatter")
plt.show()

# Distribution
# Density
sns.pairplot(dfEmplotmentMeanWage, diag_kind="kde")
plt.show()
# Histogram
sns.pairplot(dfEmplotmentMeanWage, diag_kind="hist")
plt.show() 
# You can custom it as a density plot or histogram so see the related sections
sns.pairplot(dfEmplotmentMeanWage, diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False) )
plt.show()

plt.show()

# Create the plot fig 2 

bars = (range(len(byareas_Employment)))
y_pos = np.arange(len(bars))
height = byareas_Employment
plt.bar(y_pos, height, color='#75bbfd')
 
# If we have long labels, we cannot see it properly
names = byarea.index
plt.xticks(y_pos, names, rotation=90)
 
# Thus we have to give more margin:
plt.subplots_adjust(bottom=0.4)
 
# It's the same concept if you need more space for your titles
plt.title("Employment of Area")
plt.subplots_adjust(top=0.7)
plt.show()

# ## Report Point -> Part I 
# Create the plot fig 3

# function for create normed plot
def normedplot(data, nbins, title):
    mu = sp.mean(data)
    sigma = sp.std(data)
    n,bins,patches = plt.hist(data, nbins, normed = True,  color='#75bbfd')
    plt.plot(bins, 1/(sigma * sp.sqrt(2 * sp.pi)) *sp.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
    plt.title(title)
    plt.show()
    print (mu)
    print (sigma)

# call function normedplot for Mean Wage (all data)
normedplot(dataset['Mean Wage'], nbins = 12, title='Normal distribution for the Mean Wage variable (all)')


# Create the plot fig 4
# CDF plot
# this is not included in report
#sns.kdeplot(byareas_mean_wage, cumulative=True)

# ECDF vd CDF plot

sorted_data = np.sort(byareas_mean_wage)

yvals=np.arange(len(sorted_data))/float(len(sorted_data)-1)

ecdf = sm.distributions.ECDF(byareas_mean_wage)

x = np.linspace(min(byareas_mean_wage), max(byareas_mean_wage))
y = ecdf(x)
plt.step(x, y, label='ECDF')
plt.plot(sorted_data,yvals, label='CDF',  color='#75bbfd')
plt.legend()
plt.title('ECDF vs CDF')
plt.title('Mean of the variable (Mean Wage) grouped region')
plt.show()





# function for create plot violin

def violinplot(data):
    #  Boxplot for Mean Wage by Area
    sns.violinplot( y=data )
    plt.show()
    print(data.mean())
    

violinplot(byOcu_CP['Mean Wage'])
violinplot(byOcu_ISA['Mean Wage'])

def boxplot(data):
    #  Boxplot for Mean Wage by Area
    sns.boxplot( y=data )
    plt.show()
    print(data.mean())


boxplot(byOcu_CP['Mean Wage'])
boxplot(byOcu_ISA['Mean Wage'])



# This is not be in report
#In a normal distribution, the interval [μ - 2σ, μ + 2σ] 
#covers 95.5 %, so you can use 2 * std to estimate the 95 % interval:

mean = byareas_mean_wage.mean()
std = byareas_mean_wage.std()

plt.errorbar(mean, mean, xerr=0.5, yerr=2*std, linestyle='')
plt.show()



# Pearson vs Spearman
x = byOcu_CP.iloc[:, 2].values
y = byOcu_CP.iloc[:, 3].values


print ('\n')
print ('Spearman: ',st.spearmanr(x,y)[0])
print ('\n')
print ('Pearson: ',st.pearsonr(y,y)[0])


#Linear regression
slope, intercept, r_value, p_value, std_err = st.linregress(x, y)



#To get coefficient of determination (r_squared)

print("r-squared:", r_value**2, "\n")



#Plot the data along with the fitted line

plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.legend()
plt.show()



# KS test for variable Mean Wage

x = dataset.iloc[:, 3].values
mu = sp.mean(x)
sigma = sp.std(x)
nsize = len(x)


#Let's kick off by making a histogram of the data:
nbins = 15

#As before, the histogram function does everything for you
n,bins,patches = plt.hist(x, nbins, normed = True)

#The data looks normal-ish, so let's work out what the parameters of the normal distribution would be:


#If the data does indeed come from a normal distribution with mu and sigma, then the data should fit nicely with the following line
n,bins,patches = plt.hist(x, nbins, normed = True) #Dataset
plt.plot(bins, 1/(sigma * sp.sqrt(2 * sp.pi)) *sp.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')#Model


#First put your data in order
x.sort()

#Then create a list of the same length as your data
ecdf = sp.arange(nsize, dtype = "float")

print ("This is what ecdf looks like", ecdf)

#And now normalise it by the final value to give you a fraction of how far through the dataset you are at each point
ECDF = ecdf/ecdf[-1]

print ("And this is what the normalised ECDF looks like", ECDF)

#And plot it
plt.plot(x, ECDF)
plt.show()

#We want to compare this to a perfect normal distribution with mu and sigma as the parameters
CDF = st.norm(mu,sigma).cdf(x) #Evaluates the cdf at the points listed in "scores"

#Now plot them both to compare:
plt.plot(x,CDF, label='CDF')
plt.plot(x, ECDF, label='ECDF')
plt.legend()
plt.title('Kolmogorov–Smirnov test')
plt.show()

#The KS test is going to tell you the maximum distance between these lines:
ks = max(abs(CDF-ECDF))

print (ks)

print (st.kstest(x,'norm',args=(mu,sigma)))


# KS test for variable Employment

# Este codigo faz parte da revisão solicitada pelo professor no dia 10/12/17

x = dataset.iloc[:, 2].values

mu = sp.mean(x)
sigma = sp.std(x)
nsize = len(x)


#Let's kick off by making a histogram of the data:
nbins = 15

#As before, the histogram function does everything for you
n,bins,patches = plt.hist(x, nbins, normed = True)

#The data looks normal-ish, so let's work out what the parameters of the normal distribution would be:


#If the data does indeed come from a normal distribution with mu and sigma, then the data should fit nicely with the following line
n,bins,patches = plt.hist(x, nbins, normed = True) #Dataset
plt.plot(bins, 1/(sigma * sp.sqrt(2 * sp.pi)) *sp.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')#Model


#First put your data in order
x.sort()

#Then create a list of the same length as your data
ecdf = sp.arange(nsize, dtype = "float")

print ("This is what ecdf looks like", ecdf)

#And now normalise it by the final value to give you a fraction of how far through the dataset you are at each point
ECDF = ecdf/ecdf[-1]

print ("And this is what the normalised ECDF looks like", ECDF)

#And plot it
plt.plot(x, ECDF)
plt.show()

#We want to compare this to a perfect normal distribution with mu and sigma as the parameters
CDF = st.norm(mu,sigma).cdf(x) #Evaluates the cdf at the points listed in "scores"

#Now plot them both to compare:
plt.plot(x,CDF, label='CDF')
plt.plot(x, ECDF, label='ECDF')
plt.legend()
plt.title('Kolmogorov–Smirnov test')
plt.show()

#The KS test is going to tell you the maximum distance between these lines:
ks = max(abs(CDF-ECDF))

print (ks)

print (st.kstest(x,'norm',args=(mu,sigma)))

