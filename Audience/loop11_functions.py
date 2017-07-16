
# coding: utf-8

# #### Import packages

# In[1]:

import pandas as pd
from IPython.display import display, HTML

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
get_ipython().magic('matplotlib inline')


# ## Graph plotting functions

# In[2]:

# Function for basic bar graph
def grapher(df, cols, title, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=False, color=colours, edgecolor = "none", rot = 0)
    ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    plt.savefig(location+title+".svg")


# In[3]:

# Function for bar graph with error bars
def grapher_errorbar(df, cols, title, cis, labels, location):
    plot_df = df[cols]
    colours=['#9d9d9c','#074f5f','#42bac7']
    ax = plot_df.plot(kind='bar', legend=False, color=colours, edgecolor = "none", rot = 0, yerr=cis)
    ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    if not(not labels):
        ax.legend(labels,loc='best')
    plt.savefig(location+title+".svg")


# In[4]:

# Function for horizontal graph
def grapher_horizontal(df, cols, title, location):
    plot_df = df[cols]
    colours=['#9d9d9c','#074f5f','#42bac7']
    ax = plot_df.plot(kind='barh', legend=False, color=colours, edgecolor = "none", rot = 0)
    ax.set_yticklabels(list(plot_df[cols].index))
    ax.set_xlim([0,1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    plt.savefig(location+title+".svg")


# In[5]:

# Function for bar graph with stacked data sets
def grapher_stacked(df, cols, title, labels, cis, location):
    plot_df = df[cols].unstack(level=1).sort_index(ascending=False)
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0, yerr=cis)
    ax.set_xticklabels(list(plot_df.index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    ax.legend(labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[6]:

def grapher_errorbar_stacked(df, cols, title, cis, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0, yerr = cis)
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[7]:

# Function for bar graph with stacked data sets but no CIs
def grapher_no_error(df, cols, title, location):
    plot_df = df[cols].unstack(level=1).sort_index(ascending=False)
    colours=['#42bac7','#074f5f','#9d9d9c', '#d8b2d8', '#f47920']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0)
    ax.set_xticklabels(list(plot_df.index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[8]:

def grapher_stacked_CIS(df, cols, title, labels, location):
   plot_df = df[cols]
   colours=['#42bac7','#074f5f','#9d9d9c']
   ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0)
   #ax.set_xticklabels(list(plot_df[cols].index))
   ax.set_ylim([0,1])
   ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
   ax.set_title(title)
   plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   plt.savefig(location+title+".svg")


# In[9]:

# Function for graphs with stacked data sets
def grapher_stacked_N(df, cols, title, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0)
    ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[10]:

# Function for graphs with stacked data sets
def grapher_stacked_horizontal(df, cols, title, location):
    plot_df = df[cols]#.unstack(level=1)
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='barh', legend=False, color=colours, edgecolor = "none", rot = 0)
    ax.set_yticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.savefig(location+title+".svg")


# In[11]:

# Function for graphs with stacked data sets
def grapher_stacked_labels(df, cols, labels, title, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c', '#d8b2d8', '#f47920']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0)
    #ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2,  borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[12]:

# Function for graphs with stacked data sets
def grapher_stacked_labels_horizontal(df, cols, labels, title, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c', '#d8b2d8', '#f47920']
    ax = plot_df.plot(kind='barh', legend=True, color=colours, edgecolor = "none", rot = 0)
    #ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2,  borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[13]:

# Function for bar graph with stacked data sets
def grapher_demo_breakdown(df, cols, title, labels, cis, location):
    #plot_df = df[cols].unstack(level=1).sort_index(ascending=False)
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c', '#ffffff', '#eeeeee']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0, yerr=cis)
    ax.set_xticklabels(list(plot_df.index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    ax.legend(labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# ## Likert questions

# In[14]:

def likert_results(df, questions, graph_type, file_location):
    
    results_compact = []
    results_numerical = []
    
    # Convert to numerical data for calculation of mean and std dev.
    # 1 = SD, 2 = D, 3 = N A/D, 4 = A, 5 = SA
    for q in questions:
        
        temp = df[q].replace('Strongly agree', 5, regex=True)
        temp = temp.replace('Agree', 4, regex=True)
        temp = temp.replace('Neither agree nor disagree', 3, regex=True)
        temp = temp.replace('Disagree', 2, regex=True)
        temp = temp.replace('Strongly disagree', 1, regex=True)
        temp = temp.replace('NaN', 3, regex=True)
        results_numerical.append(pd.DataFrame(temp))
    
    # Concatenate the list into a DataFrame and use describe() to get the mean and std dev.
    results_numerical = pd.concat(results_numerical, axis=1)
    print(results_numerical.describe())
    
    # Compact standard Likert answers to three-point scale.
    # Create a list with question value counts (both number counts and as a percentage of the total).
    for q in questions:
        temp = df[q].replace('Strongly agree', 'Agree', regex=True)
        temp = temp.replace('Strongly disagree', 'Disagree', regex=True)
        temp = temp.replace('NaN', 3, regex=True)
        results_compact.append(pd.DataFrame(temp.value_counts()))
        results_compact.append(pd.DataFrame(temp.value_counts(normalize=True)))
    
    # Concatenate the list DataFrames into a single DataFrame
    results_compact = pd.concat(results_compact, axis=1)
    
    # Create column titles list ('N' and '&' column for each question).
    column_titles = []
    for q in questions:
        column_titles.append(q + ' (N)')
        column_titles.append(q + ' (%)')
    
    # Rename columns in the DataFrame.
    results_compact.columns = column_titles
    
    # Sort dataframe rows
    results_compact.sort_index(axis=0, ascending=True, inplace=True)
       
    # Print % total is for quality control. The total should be 1.0 if question was required.
    for q in questions:    
        total_count = results_compact[q + ' (%)'].sum(axis=0)
        print('TOTAL %: '  + str(total_count) + ' (' + q + ')')
    
        title = q.replace('"', '')
        graph_type(results_compact, [q + ' (%)'], title, file_location)
        
    #return final DataFrame for use in grapher functions.
    return results_compact


# ## Multiple choice - single answer

# In[25]:

def single_answer(df, questions, graph_type, file_location):
    
    results = {}
    # Create a list with question value counts (both number counts and as a percentage of the total).
    for q in questions:
        
        results[q] = []

        results[q].append(pd.DataFrame(df[q].value_counts(dropna=True)))
        results[q].append(pd.DataFrame(df[q].value_counts(normalize=True, dropna=True)))

        # Concatenate the list DataFrames into a single DataFrame
        results[q] = pd.concat(results[q], axis=1)
    
        # Create column titles list ('N' and '&' column for each question).
        column_titles = []
        column_titles.append(q + ' (N)')
        column_titles.append(q + ' (%)')
    
        # Rename columns in the DataFrame.
        results[q].columns = column_titles
        
             
        # Print sample size, % total and graph for each question
        # Note: % total is for quality control. The total should be 1.0 if question was required.
        sample_size = results[q][q + ' (N)'].sum(axis=0)
        print('SAMPLE SIZE: ' + str(sample_size) + ' (' + q + ')')
        total_count = results[q][q + ' (%)'].sum(axis=0)
        print('TOTAL %: '  + str(total_count) + ' (' + q + ')')
        
        # Sort dataframe rows
        results[q].sort_index(axis=0, ascending=True, inplace=True)
        
        # Display the results DataFrame
        display(results[q])
        
        # Graph the results
        title = q.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        graph_type(results[q], [q + ' (%)'], title, file_location)
        
    #return final DataFrames.
    #return results


# In[ ]:

# Function for analysing single answer questions where the results need to be grouped.
# Parameters :
#      df - dataframe, e.g. clean_df
#      questions - The question to be analysed. Takes a string in a list.
#      graph_type: The graph to be used.
#      group_scale: The scale to group the data by. Takes a list of integers.
#      group_labels - list of strings, labels of the columns for the output of the stacked data. 
#      file_location - string, folder location for saving the graph.

def single_answer_grouped(df, questions, graph_type, group_scale, group_labels, file_location):
    
    results = {}

    # Create a list with question value counts (both number counts and as a percentage of the total).
    for q in questions:
               
        results[q] = []
        
        # Group the data by the scale.
        temp = pd.cut(df[q], bins=group_scale, right=False, labels = group_labels)
        
        results[q].append(pd.DataFrame(temp.value_counts(dropna=True)))
        results[q].append(pd.DataFrame(temp.value_counts(normalize=True, dropna=True)))

        # Concatenate the list DataFrames into a single DataFrame
        results[q] = pd.concat(results[q], axis=1)
    
        # Create column titles list ('N' and '&' column for each question).
        column_titles = []
        column_titles.append(q + ' (N)')
        column_titles.append(q + ' (%)')
    
        # Rename columns in the DataFrame.
        results[q].columns = column_titles
        
        # Sort dataframe rows
        results[q].sort_index(axis=0, ascending=True, inplace=True)
        
        # Display the results DataFrame
        display(df[q].describe())
        display(results[q])
            
        # Graph the results
        title = q.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        graph_type(results[q], [q + ' (%)'], title, file_location)
        
    #return final DataFrames.
    #return results


# ## Multiple choice - multiple answer

# In[27]:

def multiple_answer(df, questions, graph_type, file_location):
    
    results = {}
    
    for q in questions:
        # Get sample size before the answers are split up.
        sample_size = len(df[q])
        
        # Replace NaN with 'Other'
        # Remove rows with 'Other'. NOT an accurate reflection of how many participants selected other.
        # Due to Loop11 data format.
        temp = df[q].fillna('Other')
        temp = temp[temp.str.contains('Other') == False]
    
        # Break up the strings with ',' as the delimiter
        broken_down = [sub.split(",") for sub in temp]
        reshaped_list = [item.strip() for sublist in broken_down  for item in sublist]
        set(reshaped_list)
        temp = pd.DataFrame(reshaped_list)
    
        # Tally the results in a list of DataFrames
        results[q] = []
    
        results[q].append(pd.DataFrame(temp[0].value_counts()))
        results[q].append(pd.DataFrame(temp[0].value_counts()/len(df)))
    
        # Concatenate the results list into one DataFrame
        results[q] = pd.concat(results[q], axis=1)
        results[q].columns = [q + ' (N)', q + ' (%)']
    
        # print sample size, total number of answers received and graph of the results
        print('SAMPLE SIZE: ' + str(sample_size))
        total_answers = results[q][q + ' (N)'].sum(axis=0)
        print('NO. ANSWERS: ' + str(total_answers))
        
        # Sort dataframe rows
        results[q].sort_index(axis=0, ascending=False, inplace=True)
    
        # Display the results DataFrame
        display(results[q])
        
        # Graph the results
        title = q.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        graph_type(results[q], [q + ' (%)'], title, file_location)


# ## Task Results

# In[28]:

# NOTE THIS FUNCTION NEEDS TO BE CORRECTED SO THE CI CALCULATION WORKS
def task_completion_rates_N(df, tasks, N, file_location):
 
    results = {}

    for t in tasks:
        results[t] = []
        results[t].append(pd.DataFrame(df[t].value_counts()))
        results[t].append(pd.DataFrame(df[t].value_counts(normalize=True)))

        results[t] = pd.concat(results[t], axis=1)
        results[t].index.names = ['Task Result']
   
        # Create column titles list ('N' and '&' column for each question).
        column_titles = []    
        column_titles.append(t + ' (N)')
        column_titles.append(t + ' (%)')
        
        # Rename columns in the DataFrame.
        results[t].columns = column_titles
       
        results[t] = results[t].reindex(['success', 'fail', 'abandon'])
        
        # CALCULATE CONFIDENCE INTERVALS
        Task_p = (results[t][t + ' (N)']['success'])/N

        # CI_95 = 1.96 * SQRT((adjusted_p * (1 - adjusted_p))/adjusted_n)
        Task_CI = 1.96 * np.sqrt(Task_p * (1 - Task_p)/N)

        # Print confidence intervals for reference.
        print('CONFIDENCE INTERVAL: ' ,t , Task_CI)
    
        # Display the results DataFrame
        display(results[t])
    
        # Graph the results
        title = t.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        grapher_errorbar(results[t], [t + ' (%)'], title, [Task_CI, Task_CI, Task_CI], [], file_location)

    #return results


# In[29]:

def task_completion_rates_adjusted_N(df, tasks, N, file_location):
    # Create task results dataframe for all tasks.

    results = {}

    for t in tasks:
        results[t] = []
        results[t].append(pd.DataFrame(df[t].value_counts()))
        results[t].append(pd.DataFrame(df[t].value_counts(normalize=True)))

        results[t] = pd.concat(results[t], axis=1)
        results[t].index.names = ['Task Result']
   
        # Create column titles list ('N' and '&' column for each question).
        column_titles = []    
        column_titles.append(t + ' (N)')
        column_titles.append(t + ' (%)')
        
        # Rename columns in the DataFrame.
        results[t].columns = column_titles
       
        results[t] = results[t].reindex(['success', 'fail', 'abandon'])
        
        # CALCULATE CONFIDENCE INTERVALS
        # For small sample size adjustment
        # Adjust the sample size, N
        def adjust_n(N):
            return N+(1.96**2)

        # Adjust the proportion of successes, p
        def adjust_p(p, N):
              return (p+1.92)/(N+3.84)

        Task_p = results[t][t + ' (N)']['success']

        # CI_95 = 1.96 * SQRT((adjusted_p * (1 - adjusted_p))/adjusted_n)
        Task_CI = 1.96 * np.sqrt((adjust_p(Task_p,N) * (1 - adjust_p(Task_p,N)))/adjust_n(N))

        # Print confidence intervals for reference.
        print('CONFIDENCE INTERVAL: ' ,t , Task_CI)
    
        # Display the results DataFrame
        display(results[t])
    
        # Graph the results
        title = t.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        grapher_errorbar(results[t], [t + ' (%)'], title, [Task_CI, Task_CI, Task_CI], [], file_location)

    #return results


# ## Comparison - Task vs multiple choice - MULTIPLE answer

# In[30]:

def compare_task_VS_multiple_adjusted_N(df, group_q, group_names, task, file_location):
    
    # Create dataframes of required groups and store in a dict. 
    answer_groups = {}
    
    for item in group_names:
        answer_groups[item] = df[df[group_q].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    results = {}
    
    for item in answer_groups:
        # Create a list with task value counts.
        results[item] = []
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts()))
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts(normalize=True)))

        results[item] = pd.concat(results[item], axis=1)

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1) 
    
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
    
    graph_columns = []
    for i in column_titles2:
        if "(N)" not in i: 
            graph_columns.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = task

    
    display(combined_results)
        
    # CALCULATE CONFIDENCE INTERVALS
    def adjust_n(N):
        return N+(1.96**2)
    # Adjust the proportion of successes, p
    def adjust_p(p, N):
        return (p+1.92)/(N+3.84)
    
    # List to store CI in
    CIs = [] 
    for item in group_names:
        group_N = len(answer_groups[item])
        Task_p = combined_results[item + ' (N)']['success']

        # CI_95 = 1.96 * SQRT((adjusted_p * (1 - adjusted_p))/adjusted_n)
        Task_CI = 1.96 * np.sqrt((adjust_p(Task_p, group_N) * (1 - adjust_p(Task_p, group_N)))/adjust_n(group_N))

        # Save CI in list.
        CIs.append([Task_CI,Task_CI,Task_CI])
        
        
        # Print confidence intervals for reference.
        print('CONFIDENCE INTERVAL: ' ,item , Task_CI)
      
    grapher_stacked_CIS(combined_results, graph_columns, task, [], file_location)


# In[31]:

def compare_task_VS_multiple_N(df, group_q, group_names, task, file_location):
    
    # Create dataframes of required groups and store in a dict. 
    answer_groups = {}
    
    for item in group_names:
        answer_groups[item] = df[df[group_q].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    results = {}
    
    for item in answer_groups:
        # Create a list with task value counts.
        results[item] = []
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts()))
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts(normalize=True)))

        results[item] = pd.concat(results[item], axis=1)

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1) 
    
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
    
    graph_columns = []
    for i in column_titles2:
        if "(N)" not in i: 
            graph_columns.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = task

    
    display(combined_results)
        
    # CALCULATE CONFIDENCE INTERVALS
    
    # List to store CI in
    CIs = [] 
    for item in group_names:
        group_N = len(answer_groups[item])
        Task_p = combined_results[item + ' (N)']['success']

        # CI_95 = 1.96 * SQRT((adjusted_p * (1 - adjusted_p))/adjusted_n)
        Task_CI = 1.96 * np.sqrt(Task_p * (1 - Task_p)/group_N)
        
        # CALCULATE CONFIDENCE INTERVALS
        Task_p = (results[t][t + ' (N)']['success'])/N


        # Save CI in list.
        CIs.append([Task_CI,Task_CI,Task_CI])
        
        
        # Print confidence intervals for reference.
        print('CONFIDENCE INTERVAL: ' ,item , Task_CI)
      
    grapher_stacked_CIS(combined_results, graph_columns, task, [], file_location)


# # Compare success rates between 2 or groups

# In[32]:

# Create resutls for ATO versus non-ATO per task
def breakdown_task_rates(df, task, breakdown, col_names, title, labels, location):
    
    groups = df[breakdown].unique()
    print(groups)
    results = []
    
    results.append(pd.DataFrame(df[task].groupby(df[breakdown]).value_counts()))
    results.append(pd.DataFrame(df[task].groupby(df[breakdown]).value_counts(normalize=True)))
    results = pd.concat(results, axis=1)
    results.index.names = col_names
    results.columns = [task+' (N)', task+' (%)']
    results = results.sort_index(ascending=False)
    
    # print
    display(results)
    
    # Calculate 95% CIfor each task results per group
    # CI = p_hat +- 1.96 * (sqrt((p_hat*(1-p_hat))/n))
    cis=[]
    for i in groups:
        n = results[task + ' (N)'][i].sum()
        p_hat = (results[task + ' (N)'][i,'success'])/n
        cis.append(1.96 * (np.sqrt((p_hat * (1-p_hat))/n)))
    
    grapher_stacked(results, task+' (%)', title, labels, cis, location)


# # Compare likert results between 2 or groups

# In[33]:

# break down likert scale questions by dividing up group 
def breakdown_likert_rates(df, task, breakdown, col_names, title, labels, location):
    # Compact standard Likert answers to three-point scale.
    cleaned = df[task].replace({'Very dissatisfied': 'Dissatisfied', 'Very satisfied': 'Satisfied',
                'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)

    temp = pd.concat([cleaned, df[breakdown]], axis=1)

    results = []
    results.append(pd.DataFrame(temp[task].groupby(temp[breakdown]).value_counts()))
    results.append(pd.DataFrame(temp[task].groupby(temp[breakdown]).value_counts(normalize=True)))
    results = pd.concat(results, axis=1)
    results.index.names = col_names
    results.columns = [task+' (N)', task+' (%)']
    results = results.sort_index(ascending=False)

    display(results)

    grapher_no_error(results, task+' (%)', title, labels, location)


# ## Comparison - Task vs likert

# In[34]:

'''
def compare_task_VS_likert(df, task, question, labels, location):
    
    likert_compact = []
    
    # Compact standard Likert answers to three-point scale and save in a list.
    # Create a list with task value counts.
    likert_compact.append(pd.DataFrame(df[question].replace(
                {'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)))
    likert_compact.append(pd.DataFrame(df[task]))

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(likert_compact , axis=1) 

    # Create DataFrame with the results grouped by task completion rate.
    combined_results_df = []
    
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts()))
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts(normalize=True)))
    
    combined_results_df = pd.concat(combined_results_df, axis=1)
    combined_results_df.index.names = [task + ' : Result', question]
    combined_results_df.columns = ['(N)', '(%)']
    combined_results_df = combined_results_df.sort_index(ascending=False)
    
    display(combined_results_df)
    
    # Graph the results
    title = question.replace('"', '')
    title = title.replace('?', '')
    title = (title[:100] + '...') if len(title) > 100 else title
    
    # Graph the results
    grapher_no_error(combined_results_df, '(%)', title, labels, location)
    '''


# In[35]:

# Function for comparison of task success to likert scale questions
# Parameters :
#      df - dataframe, e.g. clean_df
#      task - string, name of the column in the dataframe with the task data
#      question task - string, name of the column in the dataframe with the likert question data
#      labels - list of strings, labels of the columns for the output of the stacked data, 
#      location - string, folder location for saving the graph to
#      scale - dictionary, the likert scale used

def compare_task_VS_likert(df, task, question, location, scale):
    
    likert_compact = []
    
    # Compact standard Likert answers to three-point scale and save in a list.
    # Create a list with task value counts.
    likert_compact.append(pd.DataFrame(df[question].replace(
                {'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)))
    likert_compact.append(pd.DataFrame(df[task]))

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(likert_compact , axis=1) 

    # Create DataFrame with the results grouped by task completion rate.
    combined_results_df = []
    
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts()))
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts(normalize=True)))
    
    combined_results_df = pd.concat(combined_results_df, axis=1)
    combined_results_df.index.names = [task + ' : Result', question]
    combined_results_df.columns = ['(N)', '(%)']
    combined_results_df = combined_results_df.sort_index(ascending=False)
    
    display(combined_results_df)
    
    # Graph the results
    t = task.replace('"', '')
    q = question.replace('"', '')
    q = q.replace('* ', '')
    title = t + ' vs. ' + q
    
    # Graph the results
    grapher_no_error(combined_results_df, '(%)', title, location)


# ## Comparison - Task vs multiple choice - SINGLE answer

# In[36]:

'''
def compare_task_VS_single(df, task, question, graph_type, file_location):
    
    likert_compact = []
    
    # Compact standard Likert answers to three-point scale and save in a list.
    # Create a list with task value counts.
    likert_compact.append(pd.DataFrame(df[question]))
    likert_compact.append(pd.DataFrame(df[task]))

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(likert_compact , axis=1) 

    # Create DataFrame with the results grouped by task completion rate.
    combined_results_df = []
    
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts()))
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts(normalize=True)))
    
    combined_results_df = pd.concat(combined_results_df, axis=1)
    combined_results_df.index.names = [task + ' : Result', question]
    combined_results_df.columns = ['(N)', '(%)']
    combined_results_df = combined_results_df.sort_index(ascending=False)
    
    display(combined_results_df)
'''


# In[37]:

# Function for comparison of task success to single answer questions
# Parameters :
#      df - dataframe, e.g. clean_df
#      task - string, name of the column in the dataframe with the task data
#      question task - string, name of the column in the dataframe with the likert question data
#      location - string, folder location for saving the graph to
def compare_task_VS_single(df, task, question, file_location):
    
    results = []
    
    # Create a list with task value counts.
    results.append(pd.DataFrame(df[question]))
    results.append(pd.DataFrame(df[task]))

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results , axis=1) 

    # Create DataFrame with the results grouped by task completion rate.
    combined_results_df = []
    
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts()))
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[task]).value_counts(normalize=True)))
    
    combined_results_df = pd.concat(combined_results_df, axis=1)
    combined_results_df.index.names = [task + ' : Result', question]
    combined_results_df.columns = ['(N)', '(%)']
    combined_results_df = combined_results_df.sort_index(ascending=False)
    
    display(combined_results_df)
    
    # Graph the results
    t = task.replace('"', '')
    q = question.replace('"', '')
    q = q.replace('* ', '')
    title = t + ' vs. ' + q
    title = (title[:100] + '...') if len(title) > 100 else title
    
    # Graph the results
    grapher_no_error(combined_results_df, '(%)', title, file_location)


# ## Comparison - Likert vs multiple choice - SINGLE answer

# In[38]:

def compare_likert_VS_single(df, likert, question, graph_type, file_location):
    
    results_compact = []
    
    # Compact standard Likert answers to three-point scale and save in a list.
    # Create a list with task value counts.
    results_compact.append(pd.DataFrame(df[likert].replace({'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)))
    results_compact.append(pd.DataFrame(df[question]))

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results_compact , axis=1) 

    # Add grouped results dataframes to a list.
    combined_results_df = []
    
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[likert]).value_counts()))
    combined_results_df.append(pd.DataFrame(combined_results[question].groupby(combined_results[likert]).value_counts(normalize=True)))
    
    # Convert list of grouped results dataframes into one DataFrame
    combined_results_df = pd.concat(combined_results_df, axis=1)
    combined_results_df.index.names = [likert, question]
    combined_results_df.columns = ['(N)', '(%)']
    combined_results_df = combined_results_df.sort_index(ascending=False)
    
    display(combined_results_df)


# ## Likert vs multiple choice - MULTIPLE answer

# In[39]:

def compare_likert_VS_multiple(df, group_q, group_names, task, file_location):
    
    temp = []
    temp.append(pd.DataFrame(df[task].replace({'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)))
    temp.append(pd.DataFrame(df[group_q]))
    
    temp_df = pd.concat(temp, axis=1)
    
    # Create dataframes of required groups and store in a dict. 
    answer_groups = {}
    
    for item in group_names:
        answer_groups[item] = temp_df[temp_df[group_q].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    results = {}   
    
    for item in answer_groups:
        
        # Create a list with task value counts.
        results[item] = []
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts()))
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts(normalize=True)))

        results[item] = pd.concat(results[item], axis=1)

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1) 
    
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = task
      
    display(combined_results)


# ## Multiple choice: SINGLE answer vs MULTIPLE answer

# In[29]:

def compare_multiple_VS_single(df, group_q, group_names, single_q, file_location):
    
    # Create dataframes of required groups and store in a dict. 
    answer_groups = {}
    
    for item in group_names:
        answer_groups[item] = df[df[group_q].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    results = {}
    column_titles = []
    graph_columns = []
    
    for item in answer_groups:
        # Create a list with single_q value counts.
        results[item] = []
        results[item].append(pd.DataFrame(answer_groups[item][single_q].value_counts()))
        results[item].append(pd.DataFrame(answer_groups[item][single_q].value_counts(normalize=True)))

        results[item] = pd.concat(results[item], axis=1)
    
    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1) 
    
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
            
    graph_columns = []
    for i in column_titles2:
        if "(N)" not in i: 
            graph_columns.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = single_q
      
    display(combined_results)
    
    grapher_stacked_N(combined_results, graph_columns, single_q, file_location)


# In[ ]:




# ## Display other results (for relevant questions)

# In[30]:

def other_responses(df, question):
    responses = []

    responses.append(pd.DataFrame(df[question]))
    responses = pd.concat(responses, axis=1)
    responses = responses[responses[question].notnull()]

    print('NUMBER OF RESPONSES: ' , len(responses))
    
    for item in responses[question]:
        display(item)
        


# ## Create a list of dataframes for dividing multiple answer questions

# In[31]:

# Create dataframes of required groups.
def define_groups(df, question, groups):
    
    answer_groups = {}
    
    for item in groups:
        answer_groups[item] = df[df[question].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    return answer_groups


# # Functions to compare one group with everyone else

# ## Pull out one group.

# In[32]:

# Function to pull out the required group.
# Everyone else is placed in a 'General' group.
# This function is used in the COMPARE...TWO_GROUPS functions.

def pull_out_group(df, group_q, group_names):
    answer_groups = {}
    
    for item in group_names:
        answer_groups[item] = df[df[group_q].str.contains('(?i)'+item) == True]
        answer_groups['General'] = df[df[group_q].str.contains('(?i)'+item) == False]
        answer_groups[item].name = item
             
    sample_total = 0
    for item in answer_groups:
        print(item, 'sample size:', len(answer_groups[item]))
        sample_total += len(answer_groups[item])
        
    print('TOTAL:', sample_total)
    
    return answer_groups


# ## Compare Task results for one group vs everyone else

# In[33]:

# Function to compare TASK RESULTS for ONE demographic group with everyone else.

# !!!!!!!!!!!!!!!!!!
# THIS FUNCTION IS PRINTING THE CONFIDENCE INTERVALS FOR EACH GROUP BUT NOT PRINTING THEM ON THE GRAPH CORRECTLY.
# FIX THIS! 
# THE GRAPH CURRENTLY WANTS 3 CONFIDENCE INTERVALS (for success, fail, abandon) BUT THE CIS ARE CALCULATED BY GROUP.
# !!!!!!!!!!!!!!!!!!

# PARAMETERS
# answer_groups: a list of the groups being compared. This can be created using the pull_out_group() function.
# task: The task. Takes a string which is the column name.
# file_location: relative pathway where the graph will be saved.


def COMPARE_task_TWO_GROUPS(df, group_q, group_names, task, file_location):
       
    # Create dataframes for each group and place in a dict.
    answer_groups = pull_out_group(df, group_q, group_names)
    
    results = {}
    
    for item in answer_groups:
        # Create a list with task value counts.
        results[item] = []
        
        results[item].append(pd.DataFrame(answer_groups[item][task].value_counts()))
        results[item].append(pd.DataFrame((answer_groups[item][task].value_counts(normalize=True))))

        results[item] = pd.concat(results[item], axis=1)

        
    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1) 
    
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
   
    graph_columns = []
    for i in column_titles2:
        if "(N)" not in i: 
            graph_columns.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = task

    display(combined_results)
    
    # Calculate 95% CI for each task results per group
    # CI = p_hat +- 1.96 * (sqrt((p_hat*(1-p_hat))/n))
    cis = []
    cis_labels={}
    
    headers  = ['General'] + group_names
    for i in headers:
        n = combined_results[i + ' (N)'].sum()
        p_hat = (combined_results[i + ' (N)']['success'])/n
        cis_labels[i] =  1.96 * (np.sqrt((p_hat * (1-p_hat))/n))
        cis.append(1.96 * (np.sqrt((p_hat * (1-p_hat))/n)))
        
    print('Confidence intervals:', cis)
    
    grapher_errorbar_stacked(combined_results, graph_columns, task, cis[0], file_location)


# ## Compare single answer question for one group vs everyone else.

# In[34]:

# Function to compare a SINGLE ANSWER QUESTION for ONE demographic group with everyone else.

# PARAMETERS
# ----------------------
# answer_groups: a list of the groups being compared. This can be created using the pull_out_group() function.
# task: The task. Takes a string which is the column name.
# graph-type: Takes a beta.grapher function. Options are grapher_stacked_N or grapher_stacked_horizontal.
# file_location: relative pathway where the graph will be saved.
# ----------------------


def COMPARE_single_answer_TWO_GROUPS(df, group_q, group_names, question, graph_type, file_location):
       
    # Create dataframes for each group and place in a dict.
    answer_groups = pull_out_group(df, group_q, group_names)
    
    results = {}
    
    for item in answer_groups:
        # Create a list with question value counts.
        results[item] = []
        results[item].append(pd.DataFrame(answer_groups[item][question].value_counts()))
        results[item].append(pd.DataFrame((answer_groups[item][question].value_counts(normalize=True))*100))

        results[item] = pd.concat(results[item], axis=1)

    # convert the flat list into dataframe columns 
    combined_results = pd.concat(results, axis=1)
    
   
    # Get column names and add 'N' or '%'
    column_titles = []
    for item in combined_results.columns.get_level_values(0):
        column_titles.insert(0, item + ' (N)')
        column_titles.insert(0, item + ' (%)')
    
    # remove duplicate column names from the list
    column_titles2 = []
    for i in column_titles:
        if i not in column_titles2:
            column_titles2.insert(0, i)
   
    graph_columns = []
    for i in column_titles2:
        if "(N)" not in i: 
            graph_columns.insert(0, i)
    
    # Rename columns in the DataFrame.
    combined_results.columns = column_titles2
    combined_results.index.name = question

    
    display(combined_results)
       
    graph_type(combined_results, graph_columns, question, file_location)
   


# ## Compare likert question for one group vs everyone else.

# In[35]:

# COMPARE the answers of two groups in a Likert scale question.

# Parameters:
# ----------------------
# df: the dataframe.
# group_q: the question the specific group will be pulled from.
# group_names: The name of the group. Takes a list with a string (the group name).
# question: The likert question to be analysed.
# ----------------------


def COMPARE_likert_TWO_GROUPS(df, group_q, group_names, question, file_location):
    
    # Create dataframes for each group and place in a dict.
    answer_groups = pull_out_group(df, group_q, group_names)
    
    # Convert to pandas dataframes for each group and place in a results dict.
    results = {}
    
    for group, item in answer_groups.items():
        results[group] = []
        results[group].append(pd.DataFrame(item))
        results[group] = pd.concat(results[group], axis=1)
    
    # Variables for the final dataframe and graph.
    results_compact = []
    column_titles = []
    graph_columns = []
    graph_labels = []
    
    # Compact the likert scale questions and create data for graphing.
    for group, item in results.items():
        item[question] = item[question].replace(to_replace='Strongly agree', value='Agree', regex=True)
        item[question] = item[question].replace(to_replace='Strongly disagree', value='Disagree', regex=True)

        temp = item[question]
        
        results_compact.append(pd.DataFrame(temp.value_counts()))
        results_compact.append(pd.DataFrame((temp.value_counts(normalize=True))*100))
        
        column_titles.append(group + ' ' + question + ' (N)')
        column_titles.append(group + ' ' + question + ' (%)')
        graph_columns.append(group + ' ' + question + ' (%)')
        graph_labels.append(group + ' (%)')

    # Concatenate the list DataFrames into a single DataFrame
    results_compact = pd.concat(results_compact, axis=1)
    
    # Rename columns in the DataFrame.
    results_compact.columns = column_titles
    display(results_compact)
    
    # Sort dataframe rows
    results_compact.sort_index(axis=0, ascending=True, inplace=True)
    
    grapher_stacked_labels(results_compact, graph_columns, graph_labels, question, file_location)
        
    #return answer_groups


# ## Compare multiple answer question for one group vs everyone else.

# In[36]:

# Function to compare a MULTIPLE ANSWER QUESTION for TWO GROUPS.

# Parameters:
# ----------------------
# df: the dataframe.
# group_q: the question the specific group will be pulled from.
# group_names: The name of the group. Takes a list with a string (the group name).
# question: The likert question to be analysed.
# ----------------------


def COMPARE_multiple_answer_TWO_GROUPS(df, group_q, group_names, question, file_location): 
    
    # Create dataframes for each group and place in a dict.
    answer_groups = pull_out_group(df, group_q, group_names)
    
    # Convert to pandas dataframes for each group and place in a results dict.
    results = {}
    
    for group, item in answer_groups.items():
        results[group] = []
        results[group].append(pd.DataFrame(item))
        results[group] = pd.concat(results[group], axis=1)
    
    # Variables for the final dataframe and graph.
    answers_split_out = {}
    column_titles = []
    graph_columns = []
    graph_labels = []
    
    # Compact the likert scale questions and create data for graphing.
    for group, item in results.items():
        
        # Replace NaN with 'Other'
        # Remove rows with 'Other'. NOT an accurate reflection of how many participants selected other.
        # Due to Loop11 data format.
        temp = item[question].fillna('Other')
        temp = temp[temp.str.contains('Other') == False]
        
        # Break up the strings with ',' as the delimiter
        broken_down = [sub.split(",") for sub in temp]
        reshaped_list = [item.strip() for sublist in broken_down for item in sublist]
        set(reshaped_list)
        temp = pd.DataFrame(reshaped_list)
        
        # Tally the results for each group in a dict of DataFrames
        answers_split_out[group] = []
        
        answers_split_out[group].append(pd.DataFrame(temp[0].value_counts()))
        answers_split_out[group].append((pd.DataFrame(temp[0].value_counts(normalize=True))*100))
        answers_split_out[group] = pd.concat(answers_split_out[group], axis=1)
        answers_split_out[group].columns = [group + question + ' (N)', group + question + ' (%)']
        
        #column_titles.append(group + question + ' (N)')
        #column_titles.append(group + question + ' (%)')
        graph_columns.append(group + question + ' (%)')
        graph_labels.append(group + ' (%)')
    
    # Concatenate into one dataframe
    temp_df = []
    for group, item in answers_split_out.items():
        temp_df.append(item)
    
    results_final = pd.concat(temp_df, axis=1) 
    
    display(results_final)
    
    grapher_stacked_labels_horizontal(results_final, graph_columns, graph_labels, question, file_location)

