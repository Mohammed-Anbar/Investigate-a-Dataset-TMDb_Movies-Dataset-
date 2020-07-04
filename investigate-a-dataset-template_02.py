#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset (TMDb_Movies Dataset)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > I'm going to investigate the (TMDb_Movies Dataset) which I dowloaded from Kaggle web  page. 
# >
# > The dataset has information about 10,000 movies and consist of 21 columns such as popularity, budget, 	revenue,	original_title, cast ...etc.
# >
# >I'm lookingforward to figure out which genres are most popular from year to year? and what kinds of properties are associated with movies that have high revenues?

# In[357]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you document your steps carefully and justify your cleaning decisions.
# 
# ### General Properties

# In[358]:


# Load tmdb-movies.csv dataset file
# Change release date column into date format.
tmdb = pd.read_csv('tmdb-movies.csv' ,         parse_dates = ['release_date'])
tmdb.head(3)


# In[359]:


# Count number of rows and columns
tmdb.shape


# In[360]:


tmdb.describe()


# In[361]:


tmdb.info()


# In[362]:


# Check duplicate data
duplicats = tmdb[tmdb.duplicated(keep='last')]
duplicats


# In[363]:


# Count all zero value in each colunms
(tmdb == 0).sum()


# In[364]:


# Count all null value in each colunms
tmdb.isnull().sum()


# > **Tip**: You should _not_ perform too many operations in each cell. Create cells freely to explore your data. One option that you can take with this project is to do a lot of explorations in an initial notebook. These don't have to be organized, but make sure you use enough comments to understand the purpose of each code cell. Then, after you're done with your analysis, create a duplicate notebook where you will trim the excess and organize your steps so that you have a flowing, cohesive report.
# 
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
# 
# ### Data Cleaning (Delete unnecessary information)
# 
# 
# >**From the data wrangling results I confiermed that there are a unesessary columns and duplicated data and there are some rows withe null data and others with zero budget, zero revenue and zero runtime, so these data need to be cleaned by doing the following steps:**
# >1. Delet unesessary columns which are (id, imdb_id, homepage, keywords, overview, production_companies, vote_count)
# >2. Delet duplicated rows.
# >3. Delet all rows which have zero value.
# >4. Replace zero with NAN value.
# 
# 

# In[365]:


# Delete unesessary columns

del_columns=[ 'id','imdb_id', 'homepage', 'keywords', 'overview', 'production_companies', 'vote_count']
tmdb_clean_columns= tmdb.drop(del_columns,1) 

tmdb_clean_columns.head()


# In[366]:


tmdb.shape


# In[367]:


tmdb_clean_columns.shape


# In[368]:


# Delete duplicate rows
tmdb_clean_columns.drop_duplicates(keep='last', inplace=True)
tmdb_clean_columns.shape


# In[369]:


tmdb_clean_columns.dtypes


# In[370]:


(tmdb_clean_columns == 0).sum()


# In[371]:


tmdb_clean_columns.isnull().sum()


# In[372]:


# Delete all rows with zero values.

budget_revenue_runtime = [' budget ', ' revenue ', 'budget_adj', 'revenue_adj', 'runtime'  ]

# This will replace all zero values from '0' to NAN.
tmdb_clean_columns[budget_revenue_runtime]= tmdb_clean_columns[budget_revenue_runtime].replace(0, np.NAN)


# Removing all row which has NaN value in budget_revenue
tmdb_clean_columns.dropna(subset = budget_revenue_runtime, inplace = True)

tmdb_clean_columns.shape


# In[373]:


(tmdb_clean_columns == 0).sum()


# In[374]:


tmdb_clean_columns=tmdb_clean_columns.fillna(" ")


# In[375]:


tmdb_clean_columns.isnull().sum()


# In[376]:


tmdb_clean_columns


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > ## **Questions that can analyised from this data set:**
# > 1. Find the top 10 Highest Runtime Movies?
# > 2. Find the top 10 Highest Revenues Movies?
# > 3. Find the top 10 Highest Budgets Movies?
# > 4. Find the top 10 Highest Rating Movies?
# > 5. Find the top 10 Highest Net Profits Movies?
# 
# 
# 
# 

# In[377]:


# Change the dtype fo the fields budget and revenue from object to float64

tmdb_clean_columns[[' budget ', ' revenue ']] = tmdb_clean_columns[[' budget ', ' revenue ']].apply(pd.to_numeric, errors='coerce')


# In[378]:


tmdb_clean_columns.dtypes


# In[379]:


# Add new column for the Net Profit of the each movie

tmdb_clean_columns.insert(2,'net_profit',tmdb_clean_columns['revenue_adj'] - tmdb_clean_columns['budget_adj'])
   


# In[380]:


tmdb_clean_columns.head()


# In[381]:


#tmdb_clean_columns.hist(figsize=(15,8));
tmdb_clean_columns.hist (bins = 10, rwidth = 0.95 , figsize=(15,8));


# ## Research Question 01 - Top 10 Highest Runtime Movies

# In[382]:


by_runtime = tmdb_clean_columns.sort_values(['release_year','runtime'], ascending=[True, False])


# In[383]:


by_runtime.shape


# In[384]:


top_by_runtime = by_runtime.groupby('release_year').head().reset_index(drop=True)


# In[385]:


top_by_runtime.shape


# In[386]:


top_by_runtime.head()


# In[387]:


runtime_release_year = pd.pivot_table(top_by_runtime, index = 'release_year', values = 'runtime' )


# In[388]:


runtime_release_year.shape


# In[389]:


runtime_release_year.head()


# In[390]:


runtime_release_year.describe()


# In[391]:


# Plotting Function

def plotting(DATA, KIND , X_LABEL, Y_LABEL, TITLE): 
    
    DATA.plot(kind = KIND, figsize =(15, 7))
    plt.xlabel(X_LABEL , size =(20))
    plt.ylabel(Y_LABEL , size =(20))
    plt.title(TITLE , size =(25))
    plt.legend ()
    
       


# In[392]:


plotting(runtime_release_year, 'bar', 'Release Year', 'Runtime', 'Runtime Along Years')


# In[393]:


plotting(runtime_release_year, 'line', 'Release Year', 'Runtime', 'Runtime Along Years')


# ###### NOTE:  From the ghraph we can figure out that watching movies were popular between the year of 1960 to the eyear of 195, and then there was a reluctance of watching movies strating from 1967 up to 1987 and then started to populate again and kept growing positively.

# In[394]:


top_10_runtime = top_by_runtime.nlargest(10, 'runtime')
top_10_runtime 


# In[395]:


# Top 10 Movies by Runtime

top_10_runtime = pd.pivot_table(top_10_runtime, index = 'original_title', values = 'runtime')


# In[396]:


top_10_runtime.shape


# In[397]:


top_10_runtime.head()


# In[398]:


top_10_runtime


# In[399]:


top_10_runtime = top_10_runtime.runtime.sort_values( ascending=False)


# In[400]:


plotting(top_10_runtime, 'bar', 'Title of Movie', 'Runtime', 'Top 10 Movies by Runtime')


# ##### Note:  From the ghraph we can figur out that the highest runtime movie is Carlos.

# In[401]:


runtime_by_vote_average = pd.pivot_table(by_runtime, index = 'release_year', values = ['runtime', 'vote_average'])


# In[402]:


# Plotting 2d Function

def plotting2d (DATA, KIND, X_LABEL, Y_LABEL, TITLE):
    
    DATA.plot( x = X_LABEL , y = Y_LABEL , kind = KIND, figsize =(16 , 7), label = [X_LABEL , Y_LABEL] )
    plt.xlabel ( X_LABEL , size =(20) )
    plt.ylabel ( Y_LABEL , size =(20) )
    plt.title ( TITLE , size =(25) )
    plt.legend()


# In[403]:


plotting2d (runtime_by_vote_average, 'scatter' , 'runtime' , 'vote_average' , 'Runtime Vs Voting Average' )


# ##### NOTE: From the ghraph we can figure out that there is a moderate postive non-liner correlation, and there is anout lier valye.

# ## Research Question 02 - Top 10 Highest Revenue Movies

# In[404]:


by_revenue = tmdb_clean_columns.sort_values(['release_year','revenue_adj'], ascending=[True, False])


# In[405]:


by_revenue.shape


# In[406]:


by_revenue.describe()


# In[407]:


top_by_revenue = by_revenue.groupby('release_year').head().reset_index(drop=True)


# In[408]:


top_by_revenue.shape


# In[409]:


top_by_revenue.head()


# In[410]:


revenue_release_year = pd.pivot_table(top_by_revenue, index = 'release_year', values = 'revenue_adj')


# In[411]:


runtime_release_year.shape


# In[412]:


revenue_release_year.head()


# In[413]:


plotting (revenue_release_year, 'bar', 'release_year', 'revenue_adj', 'Revenue Along Release Years')


# In[414]:


plotting (revenue_release_year, 'line', 'release_year', 'revenue_adj', 'Revenue Along Release Years')


# ##### NOTE: From the graph we can figure out that there is a moderate linear positive association between revenue and release years as the revenue kept growth.

# In[415]:


# Top 10 Movies by Revenue

top_10_revenue = top_by_revenue.nlargest(10, 'revenue_adj')
top_10_revenue 


# In[416]:


top_10_revenue = pd.pivot_table(top_10_revenue, index = 'original_title', values = 'revenue_adj')


# In[417]:


top_10_revenue.shape


# In[418]:


top_10_revenue.head()


# In[419]:


top_10_revenue = top_10_revenue.revenue_adj.sort_values(ascending = False)


# In[420]:


top_10_revenue


# In[421]:


plotting (top_10_revenue, 'bar', 'original_title', 'revenue_adj', 'Top 10 Movies by Revenue')


# ##### NOTE:  From the grapgh we can figure out that the top revenue movie ws Avatar.  

# 
# 
# 
# 
# ## Research Question 03 - Top 10 Highest Budget Movies

# In[422]:


by_budget = tmdb_clean_columns.sort_values(['release_year','budget_adj'], ascending=[True, False])


# In[423]:


by_budget.shape


# In[424]:


by_budget.describe()


# In[425]:


top_by_budget  = by_budget.groupby('release_year').head().reset_index(drop=True)


# In[426]:


top_by_budget.shape


# In[427]:


top_by_budget.head()


# In[428]:


budget_release_year = pd.pivot_table(top_by_budget, index = 'release_year', values = 'budget_adj')


# In[429]:


budget_release_year.shape


# In[430]:


budget_release_year.head()


# In[431]:


plotting (budget_release_year, 'bar', 'release_year', 'budget_adj', 'Budget Along Release Years')


# In[432]:


plotting (budget_release_year, 'line', 'release_year', 'budget_adj', 'Budget Along Release Years')


# ##### NOTE: From the graph we can figure out that budgeting of movies were increasing by the years and the highest budgeting were in the recent years after the year 2000.

# In[433]:


# Top 10 Movies by Budget

top_10_budget = top_by_budget.nlargest(10, 'budget_adj')
top_10_budget 


# In[434]:


top_10_budget = pd.pivot_table(top_10_budget, index = 'original_title', values = 'budget_adj')


# In[435]:


top_10_budget.shape


# In[436]:


top_10_budget.head()


# In[437]:


top_10_budget = top_10_budget.budget_adj.sort_values(ascending = False)


# In[438]:


top_10_budget


# In[439]:


plotting (top_10_budget, 'bar', 'original_title', 'budget_adj', 'Top 10 Movies by  Budgeting')


# ##### NOTE: From the graph we can figur out that the highest budget of movie was The Warrior's Way.

# ## Research Question 04 - Top 10 Highest Rating Movies

# In[440]:


by_vote = tmdb_clean_columns.sort_values(['release_year','vote_average'], ascending=[True, False])


# In[441]:


by_vote.shape


# In[442]:


top_by_vote  = by_vote.groupby('release_year').head().reset_index(drop=True)


# In[443]:


top_by_vote.shape


# In[444]:


top_by_vote.head()


# In[445]:


vote_release_year = pd.pivot_table(top_by_vote, index = 'release_year', values = 'vote_average')


# In[446]:


vote_release_year.shape


# In[447]:


vote_release_year.head()


# In[448]:


plotting (vote_release_year, 'bar', 'release_year', 'vote_average', 'Voting Along Release Years')


# In[449]:


plotting (vote_release_year, 'line', 'release_year', 'vote_average', 'Voting Along Release Years')


# ##### NOTES: From the graph we can figure out that voting was moderat increasing along release years except the sharp decrease between the year 1964 and the year 1967.

# In[450]:


# Top Movies by Rating

top_10_vote = top_by_vote.nlargest(10, 'vote_average')
top_10_vote 


# In[451]:


top_10_vote  = pd.pivot_table(top_10_vote, index = 'original_title', values = 'vote_average')


# In[452]:


top_10_vote.shape


# In[453]:


top_10_vote.head()


# In[454]:


top_10_vote = top_10_vote.vote_average.sort_values(ascending = False)


# In[455]:


top_10_vote


# In[456]:


plotting (top_10_vote, 'bar', 'original_title', 'vote_average', 'The Highest Vote of Movie')


# ##### NOTE: from the graph we can figur out that highest vote of movie was The Shawshank Redemption.

# ## Research Question 05 - Top 10 Highest Net Profit Movies

# In[457]:


by_net_profit = tmdb_clean_columns.sort_values(['release_year','net_profit'], ascending=[True, False])


# In[458]:


by_net_profit.shape


# In[459]:


top_by_net_profit = by_net_profit.groupby('release_year').head().reset_index(drop=True)


# In[460]:


top_by_net_profit.shape


# In[461]:


top_by_net_profit.head()


# In[462]:


net_profit_release_year = pd.pivot_table(top_by_net_profit, index = 'release_year', values = 'net_profit')


# In[463]:


net_profit_release_year.shape


# In[464]:


net_profit_release_year.head()


# In[465]:


plotting (net_profit_release_year, 'bar', 'release_year', 'net_profit', 'Net Profit Along Release Years')


# In[466]:


plotting (net_profit_release_year, 'line', 'release_year', 'net_profit', 'Net Profit Along Release Years')


# ##### NOTE:  From the graph we can figure out that the net profit was increasing along release years and we can notice also there was a sharp increase in some years such as 1973, 1978, 2010 and also the movies after the year 2014. 

# In[467]:


# Top Movies by net_profit

top_10_net_profit = top_by_vote.nlargest(10, 'net_profit')
top_10_net_profit 


# In[468]:


top_10_net_profit  = pd.pivot_table(top_10_net_profit, index = 'original_title', values = 'net_profit')


# In[469]:


top_10_net_profit.shape


# In[470]:


top_10_net_profit.head()


# In[471]:


top_10_net_profit = top_10_net_profit.net_profit.sort_values(ascending = False)


# In[472]:


top_10_net_profit


# In[473]:


plotting (top_10_net_profit, 'bar', 'original_title', 'net_profit', 'The Highest Net Profit Movie')


# ##### NOTE: From the graph we can figure out that the highest net profit movie was Star Wars.

# In[474]:


by_net_profit.head()


# In[475]:


by_net_profit.shape


# In[476]:


by_net_profit.describe()


# In[477]:


top_10_genres = by_net_profit.nlargest(10, 'net_profit')


# In[478]:


top_10_genres.head()


# In[479]:


Geners_Net_Profit = pd.pivot_table(top_10_genres, index = 'genres', values = 'net_profit')


# In[480]:


Geners_Net_Profit 


# In[481]:


Geners_Net_Profit = Geners_Net_Profit.net_profit.sort_values(ascending = False)


# In[482]:


Geners_Net_Profit 


# In[483]:


plotting(Geners_Net_Profit, 'bar', 'genres','net_profit' , 'The Highest Genres of Movies')


# ##### NOTE: From the graph we can figure out the highest genres of movies were Action, Adventure, Fantasy and Science Fiction.

# In[484]:


explorations  = pd.pivot_table(by_net_profit, index = 'release_year', values = ['net_profit', 'runtime', 'budget_adj', 'revenue_adj', 'vote_average', 'popularity', 'genres', 'cast'])


# In[485]:


explorations.describe()


# In[486]:


explorations.shape


# In[487]:


# Explorations array was created to explore some associations
explorations.head()


# In[488]:


# Top explore the corelation between the Runtime and the Net Profit

plotting2d (explorations, 'scatter', 'runtime', 'net_profit', 'Runtime Vs Net Profit')


# ##### NOTE:  From graph we can figure out that there is a moderate positive changing correlation between runtime and net profit, and there are some outlier values.

# In[489]:


# Top explore the corelation between the Budget and the Revenue

plotting2d (explorations, 'scatter', 'budget_adj', 'revenue_adj', 'Budget Vs Revenue')


# ##### NOTE: From the graph we can figure out that there is a week negative changing correlation between budget and revenue and there are some outlier values.

# <a id='conclusions'></a>
# # Conclusions
# 
# 
# > **From this investigation we can summarize the findings as follow:**
# >
# > - There is a moderate postive non-liner correlation, and there is anout lier valye.
# >
# > - there is a moderate positive changing correlation between runtime and net profit, and there are some outlier values.
# >
# > - there is a week negative changing correlation between budget and revenue and there are some outlier values.
# 
# 
# 
# > **The creteria of the successful movies and could genrate average revenue about 137 million dollar as follow:**
# >1. The average runtime is 151 minutes
# >2. The average budget is 44 million dollar.
# >3. The genrs should be adventure, action and science fiction.
# 
# 
# >**Limitations**
# >
# >This report was done depending on the provided dataset which which has a missing information, also we don't know the information acuracy  included in this dataset or if it is up to date or no.  So, droping the rows with missing information may have affetted the analysis results in this report. On the other hand the remaining complete information if we asum they are acurate we can consider the analysis result positively whic can we depend on. 
# 
# 
# ##### Answer_Q1:
# > The highest runtimes movies were from the year 2008 to 2010.
# >
# > The top 10 highest runtimes movies startimg from the minimum runtimes are:  [ The Greatest Story Ever Told, The Godfather: Part II, The Lord of the Rings: The Return of the King, Malcolm X, Jodhaa Akbar, Gods and Generals, Lawrence of Arabia, Heaven's Gate, Cleopatra and Carlos. ]
# >
# >
# ##### Answer_Q2:
# >The highest revenues movies were from the year 2009 to 2015.
# >
# >The top 10 highest  revenue movies startimg from the minimum revenues are:  [ The Avengers ,  One Hundred and One Dalmatians, The Net, E.T. the Extra-Terrestrial, Star Wars: The Force Awakens, Jaws, The Exorcist, Titanic, Star Wars, Avatar. ]   
# >
# >
# ##### Answer_Q3:
# >The highest budgets movies were from the year 2006 to 2013.
# >
# >The top 10 highest  budget movies startimg from the minimum budgets are:  [ Waterworld, Harry Potter and the Half-Blood Prince, Avengers: Age of Ultron, Tangled, Spider-Man 3, Titanic, Superman, Returns, Pirates of the Caribbean: At World's End, Pirates of the Caribbean: On Stranger Tides, The Warrior's Way. ] 
# >
# >
# ##### Answer_Q4:
# >The highest rating movies were from the year 2013 to 2015.
# >
# >The top 10 highest  rating movies startimg from the minimum rating are:  [ original_title, Fight Club, Forrest Gump, Pulp Fiction, Schindler's List, The Dark Knight,  Godfather: Part II, Whiplash, The Godfather, Stop Making Sense and the The Shawshank Redemption. ]                   
# >
# >
# ##### Answer_Q5:
# >The highest net profit movies were at the year 1973, 1977 and from the year 2009 to 2015 except the year 2014.
# >
# >The top 10 highest  net profit movies startimg from the minimum net profit are:  [ The Avengers, The Godfather, Jurassic Park, The Empire Strikes Back, One Hundred and One Dalmatians, E.T. the Extra-Terrestrial, Jaws, The Exorcist, Titanic and the Star Wars. ]    
# >
# >

# #### End of the investigation. 
