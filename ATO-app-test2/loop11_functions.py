
# coding: utf-8

# #### Import packages

# In[2]:

import pandas as pd
from IPython.display import display, HTML

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
get_ipython().magic('matplotlib inline')


# ## Graph plotting functions

# In[3]:

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


# In[4]:

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


# In[5]:

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


# In[6]:

# Function for bar graph with stacked data sets
def grapher_stacked(df, cols, title, labels, cis, location):
    plot_df = df[cols].unstack(level=0).sort_index(ascending=False)
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0, yerr=cis)
    ax.set_xticklabels(list(plot_df.index))
    ax.set_ylim([0,1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_title(title)
    ax.legend(labels)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[1]:

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


# In[1]:

# Function for graphs with stacked data sets
def grapher_stacked_N(df, cols, title, location):
    plot_df = df[cols]
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='bar', legend=True, color=colours, edgecolor = "none", rot = 0)
    ax.set_xticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(location+title+".svg")


# In[2]:

# Function for graphs with stacked data sets
def grapher_stacked_horizontal(df, cols, title, location):
    plot_df = df[cols]#.unstack(level=1)
    colours=['#42bac7','#074f5f','#9d9d9c']
    ax = plot_df.plot(kind='barh', legend=False, color=colours, edgecolor = "none", rot = 0)
    ax.set_yticklabels(list(plot_df[cols].index))
    ax.set_title(title)
    plt.savefig(location+title+".svg")


# ## Likert questions

# In[9]:

def likert_results(df, questions, graph_type, file_location):
    
    results_compact = []
    
    # Compact standard Likert answers to three-point scale.
    # Create a list with question value counts (both number counts and as a percentage of the total).
    for q in questions:
        temp = df[q].replace('Strongly agree', 'Agree', regex=True)
        temp = temp.replace('Strongly disagree', 'Disagree', regex=True)
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
       
    # Print sample size, % total and graph for each question
    # Note: % total is for quality control. The total should be 1.0 if question was required.
    for q in questions:
        sample_size = results_compact[q + ' (N)'].sum(axis=0)
        print('SAMPLE SIZE: ' + str(sample_size) + ' (' + q + ')')
    
        total_count = results_compact[q + ' (%)'].sum(axis=0)
        print('TOTAL %: '  + str(total_count) + ' (' + q + ')')
    
        title = q.replace('"', '')
        graph_type(results_compact, [q + ' (%)'], title, file_location)
        
    #return final DataFrame for use in grapher functions.
    return results_compact
    


# ## Multiple choice - single answer

# In[3]:

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
    
        #Rename columns in the DataFrame.
        results[q].columns = column_titles
       
        # Print sample size, % total and graph for each question
        # Note: % total is for quality control. The total should be 1.0 if question was required.
        sample_size = results[q][q + ' (N)'].sum(axis=0)
        print('SAMPLE SIZE: ' + str(sample_size) + ' (' + q + ')')
        total_count = results[q][q + ' (%)'].sum(axis=0)
        print('TOTAL %: '  + str(total_count) + ' (' + q + ')')
        
        # Display the results DataFrame
        display(results[q])
        
        # Graph the results
        title = q.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        graph_type(results[q], [q + ' (%)'], title, file_location)
        
    #return final DataFrames.
    #return results


# ## Multiple choice - multiple answer

# In[11]:

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
    
        # Display the results DataFrame
        display(results[q])
        
        # Graph the results
        title = q.replace('"', '')
        title = title.replace('?', '')
        title = (title[:100] + '...') if len(title) > 100 else title
        graph_type(results[q], [q + ' (%)'], title, file_location)


# ## Task Results

# In[3]:

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


# In[4]:

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

# In[1]:

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


# In[2]:

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


# ## Comparison - Task vs likert

# In[14]:

def compare_task_VS_likert(df, task, question, graph_type, file_location):
    
    likert_compact = []
    
    # Compact standard Likert answers to three-point scale and save in a list.
    # Create a list with task value counts.
    likert_compact.append(pd.DataFrame(df[question].replace({'Strongly disagree': 'Disagree', 'Strongly agree': 'Agree'}, regex=True)))
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


# ## Comparison - Task vs multiple choice - SINGLE answer

# In[17]:

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


# ## Comparison - Likert vs multiple choice - SINGLE answer

# In[18]:

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

# In[ ]:

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

# In[ ]:

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

# In[ ]:

def other_responses(df, question):
    responses = []

    responses.append(pd.DataFrame(df[question]))
    responses = pd.concat(responses, axis=1)
    responses = responses[responses[question].notnull()]

    print('NUMBER OF RESPONSES: ' , len(responses))
    
    for item in responses[question]:
        display(item)
        


# ## Create a list of dataframes for dividing multiple answer questions

# In[19]:

# Create dataframes of required groups.
def define_groups(df, question, groups):
    
    answer_groups = {}
    
    for item in groups:
        answer_groups[item] = df[df[question].str.contains('(?i)'+item) == True]
        answer_groups[item].name = item
    
    return answer_groups


# In[ ]:



