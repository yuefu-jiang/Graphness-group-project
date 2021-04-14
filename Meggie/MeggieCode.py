import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a Pandas DataFrame from adult.data called adult_df
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
             'hours-per-week', 'native-country', 'salary']
adult_df = pd.read_csv('adult.data', names=col_names, index_col=False, skipinitialspace=True)


### SEX AND SALARY MULTIVARIATE BAR GRAPH ###
# From adult_df, create a DataFrame with just columns needed; sex and salary
sex_salary_df = adult_df[['sex', 'salary']]

# For each sex, count how many people make <=50K and >50K
filt = (sex_salary_df['sex'] == 'Male') & (sex_salary_df['salary'] == '<=50K')
M_less50 = sex_salary_df[filt].count().values[0]
filt = (sex_salary_df['sex'] == 'Male') & (sex_salary_df['salary'] == '>50K')
M_over50 = sex_salary_df[filt].count().values[0]
filt = (sex_salary_df['sex'] == 'Female') & (sex_salary_df['salary'] == '<=50K')
F_less50 = sex_salary_df[filt].count().values[0]
filt = (sex_salary_df['sex'] == 'Female') & (sex_salary_df['salary'] == '>50K')
F_over50 = sex_salary_df[filt].count().values[0]

# Create a multivariate bar graph for sex/salary
N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(111)
yvals = [M_less50, F_less50]
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = [M_over50, F_over50]
rects2 = ax.bar(ind+width, zvals, width, color='r')
ax.set_ylabel('Number of People')
ax.set_xticks(ind+width/2)
ax.set_xticklabels(('Male', 'Female'))
ax.legend((rects1[0], rects2[0]), ('<=50K', '>50K'))


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')


# Display the multivariate bar graph for sex/salary
autolabel(rects1)
autolabel(rects2)
plt.title("Salary Divided By Sex")
plt.show()


### SEX AND SALARY PIE CHART ###
# Calculate the total number of people by sex
total_males = M_less50 + M_over50
total_females = F_less50 + F_over50
totals = [total_males, total_females]

# Display the pie chart for sex
plt.pie(totals, labels=['Males', 'Females'], autopct='%0.2f%%')
plt.axis('equal')
plt.title("Proportion of Total Separated by Sex")
plt.show()


### WORKCLASS AND SALARY MULTIVARIATE BAR GRAPH ###
# From adult_df, create a DataFrame with just columns needed; workclass and salary
workclass_salary_df = adult_df[['workclass', 'salary']]

# Group by each workclass
workclass_grouped_dfs = workclass_salary_df.groupby(workclass_salary_df['workclass'])

workclass_names = []
less50 = []
over50 = []

# Add each workclass's name to a list, workclass_names
for workclass in workclass_grouped_dfs:
    workclass_names.append(workclass[0])

# For each workclass, count the number of people who make <=50K and >50K
for i in range(len(workclass_names)):
    filt = ((workclass_salary_df['workclass'] == workclass_names[i]) & (workclass_salary_df['salary'] == '<=50K'))
    count = workclass_salary_df[filt].count().values[0]
    less50.append(count)
    filt = ((workclass_salary_df['workclass'] == workclass_names[i]) & (workclass_salary_df['salary'] == '>50K'))
    count = workclass_salary_df[filt].count().values[0]
    over50.append(count)

# Create a multivariate bar graph for workclass/salary
N = len(workclass_names)
ind = np.arange(N)
width = 0.27
fig = plt.figure()
ax = fig.add_subplot(111)
yvals = less50
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = over50
rects2 = ax.bar(ind+width, zvals, width, color='r')
ax.set_ylabel('Number of People')
ax.set_xticks(ind+width/9)
ax.set_xticklabels(workclass_names)
ax.legend((rects1[0], rects2[0]), ('<=50K', '>50K'))


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')

# Display the multivariate bar graph for workclass/salary
autolabel(rects1)
autolabel(rects2)
plt.title("Salary Divided By Workclass", y=1.08)
plt.show()


### WORKCLASS AND SALARY PIE CHART ###
# Count the number of people in each workclass
workclass_counts = []
for i in range(len(workclass_names)):
    count = less50[i] + over50[i]
    workclass_counts.append(count)

# Display the pie chart for workclass
plt.pie(workclass_counts, labels=workclass_names, autopct='%0.2f%%')
plt.axis('equal')
plt.title("Proportion of Total Separated by Workclass")
plt.show()


### NATIVE COUNTRY AND SALARY MULTIVARIATE BAR GRAPH INCLUDING THE US###
# From adult_df, create a DataFrame with just columns needed; native-country and salary
country_salary_df = adult_df[['native-country', 'salary']]

# Group by native country
country_grouped_dfs = country_salary_df.groupby(country_salary_df['native-country'])

country_names = []
less50 = []
over50 = []

# Add each native country's name to a list, country_names
for country in country_grouped_dfs:
    country_names.append(country[0])

# For each native country, count the number of people who make <=50K and >50K
for i in range(len(country_names)):
    filt = ((country_salary_df['native-country'] == country_names[i]) & (country_salary_df['salary'] == '<=50K'))
    count = country_salary_df[filt].count().values[0]
    less50.append(count)
    filt = ((country_salary_df['native-country'] == country_names[i]) & (country_salary_df['salary'] == '>50K'))
    count = country_salary_df[filt].count().values[0]
    over50.append(count)

# Create a multivariate bar graph for native country/salary
N = len(country_names)
ind = np.arange(N)
width = 0.27
fig = plt.figure()
ax = fig.add_subplot(111)
yvals = less50
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = over50
rects2 = ax.bar(ind+width, zvals, width, color='r')
ax.set_ylabel('Number of People')
ax.set_xticks(ind+width/N)
ax.set_xticklabels(country_names)
ax.legend((rects1[0], rects2[0]), ('<=50K', '>50K'))


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')

# Display the multivariate bar graph for native country/salary including the US
autolabel(rects1)
autolabel(rects2)
plt.title("Salary Divided By Native Country Including the US")
plt.xticks(rotation=90)
plt.show()


### NATIVE COUNTRY AND SALARY PIE CHART INCLUDING THE US###
# Count the number of people in each native country
country_counts = []
for i in range(len(country_names)):
    count = less50[i] + over50[i]
    country_counts.append(count)

#Display the pie chart for native country including the US
plt.pie(country_counts, labels=country_names, autopct='%0.2f%%')
plt.axis('equal')
plt.title("Proportion of Total Separated by Native Country Including the US")
plt.show()


### NATIVE COUNTRY AND SALARY MULTIVARIATE BAR GRAPH EXCLUDING THE US###
# From adult_df, create a DataFrame with just columns needed; native-country and salary
country_salary_df = adult_df[['native-country', 'salary']]

# Group by native country
country_grouped_dfs = country_salary_df.groupby(country_salary_df['native-country'])

country_names = []
less50 = []
over50 = []

# Add each native country's name to a list, country_names
for country in country_grouped_dfs:
    country_names.append(country[0])

# Delete the US from the list
del country_names[-3]

# For each native country except for the US, count the number of people who make <=50K and >50K
for i in range(len(country_names)):
    filt = ((country_salary_df['native-country'] == country_names[i]) & (country_salary_df['salary'] == '<=50K'))
    count = country_salary_df[filt].count().values[0]
    less50.append(count)
    filt = ((country_salary_df['native-country'] == country_names[i]) & (country_salary_df['salary'] == '>50K'))
    count = country_salary_df[filt].count().values[0]
    over50.append(count)

# Create a multivariate bar graph for native country/salary excluding the US
N = len(country_names)
ind = np.arange(N)
width = 0.27
fig = plt.figure()
ax = fig.add_subplot(111)
yvals = less50
rects1 = ax.bar(ind, yvals, width, color='g')
zvals = over50
rects2 = ax.bar(ind+width, zvals, width, color='r')
ax.set_ylabel('Number of People')
ax.set_xticks(ind+width/N)
ax.set_xticklabels(country_names)
ax.legend((rects1[0], rects2[0]), ('<=50K', '>50K'))


def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h), ha='center', va='bottom')

#Display the pie chart for native country excluding the US
autolabel(rects1)
autolabel(rects2)
plt.title("Salary Divided By Native Country Excluding the US", y=1.08)
plt.xticks(rotation=90)
plt.show()


### NATIVE COUNTRY AND SALARY PIE CHART EXCLUDING THE US###
# Count the number of people in each native country except the US
country_counts = []
for i in range(len(country_names)):
    count = less50[i] + over50[i]
    country_counts.append(count)

#Display the pie chart for native country excluding the US
plt.pie(country_counts, labels=country_names, autopct='%0.2f%%')
plt.axis('equal')
plt.title("Proportion of Total Separated by Native Country Excluding the US", y=1.08)
plt.show()
