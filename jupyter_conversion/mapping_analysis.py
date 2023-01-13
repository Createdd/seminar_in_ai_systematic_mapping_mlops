# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: thesis_analysis
#     language: python
#     name: thesis_analysis
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Library-imports-and-configs" data-toc-modified-id="Library-imports-and-configs-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Library imports and configs</a></span></li><li><span><a href="#Setup" data-toc-modified-id="Setup-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Setup</a></span></li><li><span><a href="#Load-bibliographical-data" data-toc-modified-id="Load-bibliographical-data-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Load bibliographical data</a></span><ul class="toc-item"><li><span><a href="#Prepare-data-for-further-analysis" data-toc-modified-id="Prepare-data-for-further-analysis-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Prepare data for further analysis</a></span></li><li><span><a href="#Remov-unuseful-columns" data-toc-modified-id="Remov-unuseful-columns-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Remov unuseful columns</a></span></li><li><span><a href="#Add-columns-for-mapping-study-facets" data-toc-modified-id="Add-columns-for-mapping-study-facets-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Add columns for mapping study facets</a></span></li></ul></li><li><span><a href="#Final-DF" data-toc-modified-id="Final-DF-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Final DF</a></span><ul class="toc-item"><li><span><a href="#Table-describing-the-columns" data-toc-modified-id="Table-describing-the-columns-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Table describing the columns</a></span></li></ul></li><li><span><a href="#Explore-notes" data-toc-modified-id="Explore-notes-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Explore notes</a></span><ul class="toc-item"><li><span><a href="#Extract-search-terms-from-notes" data-toc-modified-id="Extract-search-terms-from-notes-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Extract search terms from notes</a></span></li></ul></li><li><span><a href="#Explore-facets" data-toc-modified-id="Explore-facets-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Explore facets</a></span><ul class="toc-item"><li><span><a href="#Research-Facet" data-toc-modified-id="Research-Facet-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Research Facet</a></span></li><li><span><a href="#Cont-Facet" data-toc-modified-id="Cont-Facet-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Cont Facet</a></span></li><li><span><a href="#Domain-Facet" data-toc-modified-id="Domain-Facet-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Domain Facet</a></span></li></ul></li><li><span><a href="#Explore-interaction-of-research-and-contribution-facets" data-toc-modified-id="Explore-interaction-of-research-and-contribution-facets-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Explore interaction of research and contribution facets</a></span><ul class="toc-item"><li><span><a href="#Correlations" data-toc-modified-id="Correlations-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Correlations</a></span></li><li><span><a href="#Crosstab-between-research-and-contribution" data-toc-modified-id="Crosstab-between-research-and-contribution-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Crosstab between research and contribution</a></span><ul class="toc-item"><li><span><a href="#Absolute-numbers" data-toc-modified-id="Absolute-numbers-8.2.1"><span class="toc-item-num">8.2.1&nbsp;&nbsp;</span>Absolute numbers</a></span><ul class="toc-item"><li><span><a href="#Visualizations" data-toc-modified-id="Visualizations-8.2.1.1"><span class="toc-item-num">8.2.1.1&nbsp;&nbsp;</span>Visualizations</a></span></li></ul></li><li><span><a href="#Percent-of-Totals" data-toc-modified-id="Percent-of-Totals-8.2.2"><span class="toc-item-num">8.2.2&nbsp;&nbsp;</span>Percent of Totals</a></span><ul class="toc-item"><li><span><a href="#Visualizations" data-toc-modified-id="Visualizations-8.2.2.1"><span class="toc-item-num">8.2.2.1&nbsp;&nbsp;</span>Visualizations</a></span></li></ul></li></ul></li></ul></li><li><span><a href="#Explore-year" data-toc-modified-id="Explore-year-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Explore year</a></span></li><li><span><a href="#Explore-Authors" data-toc-modified-id="Explore-Authors-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Explore Authors</a></span></li><li><span><a href="#Explore-Keywords" data-toc-modified-id="Explore-Keywords-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Explore Keywords</a></span></li><li><span><a href="#Cluster-similar-words-from-keywords-and-identify-groups" data-toc-modified-id="Cluster-similar-words-from-keywords-and-identify-groups-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Cluster similar words from keywords and identify groups</a></span><ul class="toc-item"><li><span><a href="#Load-Model" data-toc-modified-id="Load-Model-12.1"><span class="toc-item-num">12.1&nbsp;&nbsp;</span>Load Model</a></span></li><li><span><a href="#Get-keywords-and-compare-to-vocab-in-pre-trained-model" data-toc-modified-id="Get-keywords-and-compare-to-vocab-in-pre-trained-model-12.2"><span class="toc-item-num">12.2&nbsp;&nbsp;</span>Get keywords and compare to vocab in pre-trained model</a></span></li><li><span><a href="#PCA" data-toc-modified-id="PCA-12.3"><span class="toc-item-num">12.3&nbsp;&nbsp;</span>PCA</a></span></li><li><span><a href="#Cluster" data-toc-modified-id="Cluster-12.4"><span class="toc-item-num">12.4&nbsp;&nbsp;</span>Cluster</a></span></li><li><span><a href="#Plot" data-toc-modified-id="Plot-12.5"><span class="toc-item-num">12.5&nbsp;&nbsp;</span>Plot</a></span></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-12.6"><span class="toc-item-num">12.6&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></li><li><span><a href="#Cluster-similar-words-from-domain-facet" data-toc-modified-id="Cluster-similar-words-from-domain-facet-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Cluster similar words from domain facet</a></span><ul class="toc-item"><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-13.1"><span class="toc-item-num">13.1&nbsp;&nbsp;</span>Conclusion</a></span></li></ul></li><li><span><a href="#Further-work" data-toc-modified-id="Further-work-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Further work</a></span></li></ul></div>
# -

# # Introduction
#
# This is the corresponding notebook to the class "seminar in AI" at the Master's study program at Johannes Kepler University in Austria.
#
# This notebook describes the mapping study conducted for my master thesis on MLOps. 
#
# The goal is to analyse various papers for the study field and provide a base for further research in presenting a reference architecture in MLOps. 

# # Library imports and configs
#
# Various used libraries are imported and certain imports configured.

# +
import pandas as pd
import bibtexparser
import numpy as np

from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn.utils import check_random_state

import plotly.express as px
import plotly.graph_objects as go
import re

# for displaying plotly inside jupyter notebook
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# # %matplotlib notebook
# -

# # Setup
#
# Here we define the 
# - project root
# - what columns from the bibliography are used
# - Random Seed for reproducability is set
#

# +
PROJECT_ROOT = '../'

# DATA_PATH = PROJECT_ROOT+'bibliography/zotero_collection_export.bib'
DATA_PATH = PROJECT_ROOT + 'bibliography/mapping_study/zotero_collection_ieee_export.bib'

# +
search_words = ['research', 'cont', 'domain', 'summary']

cont_cols = [
    'approach', 'casestudy', 'experiment', 'literature', 'metric', 'model',
    'nonempirical', 'process', 'tool'
]

research_cols = [
    'evaluation', 'experience', 'opinion', 'philosophical', 'solution',
    'validation'
]

# +
SEED = 0
RNG = np.random.RandomState(SEED)

random_state = check_random_state(RNG)
print(RNG)
print(RNG.permutation(10))


if not set(RNG.permutation(10)).issubset(set([2, 8, 4, 9, 1, 6, 7, 3, 0, 5])):
    raise ValueError('RandomState not working')

# -

# # Load bibliographical data 
#
#
# From the bibliography tool Zotero the entries are important and put into a dataframe for further usage.

# +
with open(DATA_PATH) as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)

df = pd.DataFrame(bib_database.entries)

# + code_folding=[]
if not len(df) >= 46:
    raise ValueError('Not enough content loaded')
# -

# ## Prepare data for further analysis

df

# ## Remov unuseful columns

df.columns

unuseful_cols = [
    'file', 'doi', 'issn', 'pages', 'booktitle', 'shorttitle', 'month',
    'volume', 'journal', 'isbn', 'address', 'publisher', 'series', 'ENTRYTYPE'
]

useful_columns = [col for col in df.columns if col not in unuseful_cols]
useful_columns

df = df[useful_columns]

df.head()

df.shape

# ## Add columns for mapping study facets

# + hide_input=false
# add new columns
only_notes_df = df[df.note.notna()].copy()

if len(only_notes_df) != len(df):
    raise ValueError('There are articles with missing notes')

only_notes_df = only_notes_df.reindex(only_notes_df.columns.tolist() +
                                      search_words,
                                      axis=1)

only_notes_df.shape  #.columns
# -

# # Final DF

df = only_notes_df.copy()
df.head()

df.shape

# +
# Export dataframe to excel
# df.to_excel("bibliography_dataframe.xlsx")
# -

# ## Table describing the columns
#
# In the following cell a table with corresponding column descriptions is created. This is necessary for the mapping study.

# +
descriptions = [
    'Those are the notes that are taken with Zotero. This column is used for extracting further information for the facets',
    'Automatically extracted keywords via Zotero tool. Keywords are categorizing the article to a degree.',
    'Abstract (summary) of the article. Extracted via Zotero tool.',
    'Year of publication. Extracted via Zotero tool.',
    'Author of publication. Extracted via Zotero tool.', 
    'Title of publication. Extracted via Zotero tool.',
    'ID of publication in this dataframe. Extracted via Zotero tool.',
    'Research facet according to mapping study.',
    'Contribution facet according to mapping study',
    'Domain facet according to domain study',
    'Short summary notes'
    
]

explain_cols = pd.DataFrame(df.columns, columns=['Data Item'])
explain_cols['Description']= descriptions
explain_cols['Relevant RQ']= None

explain_cols.to_excel("dataframe_explanations.xlsx")

explain_cols


# -

# # Explore notes

# ## Extract search terms from notes

# +
def extract_search_terms(search_term, note, row):
    cols = {}
    res = None

    cleaned_note = note.replace("\\par", "")
    cleaned_note = cleaned_note.split('\n')

    for word in search_words:
        for content in cleaned_note:
            if word in content:
                try:
                    key, val = re.split(':', content)
                    cols[key] = val.replace(" ", "")

                except ValueError as e:
                    print(f'{word} not in {content} -> skip: ', row.title, e)
    #                     print(cleaned_note)

    try:
        res = cols[search_term]
    except KeyError as e:
        print(f'no {search_term} skip: ', row.title)

    return res


# print(extract_search_terms(df[df.note.notna()].note.iloc[0], df[df.note.notna()].iloc[0])['research'])

# + hide_input=false
for term in search_words:
    df[term] = df.apply(
        lambda row: extract_search_terms(term, row['note'], row), axis=1)

df.head()
# -

df.shape

# # Explore facets

# ## Research Facet

df.research.value_counts()

# +
val_counts = df.research.str.split(',').explode().value_counts()

if len(val_counts) > 6:
    raise ValueError('Error with splitting. Too many values to for Research facet', len(val_counts))

print(f'''Sum of research facet units

{val_counts}

in total: {val_counts.sum()}
''')

# +
fig = px.bar(
    df.research.str.split(',').explode().value_counts(),
    y='research',
    title='Research facet distribution')

fig.update_layout(
    xaxis=dict(tickangle=45, title='Research facet categories'),
    yaxis=dict(title='Counts')
)

fig.show()
# -

df[df.research == 'solution'].title.head()

# ## Cont Facet

# +
val_counts = df.cont.str.split(',').explode().value_counts()

if len(val_counts) > 9:
    raise ValueError('Error with splitting. Too many values to for Contribution facet')

print(f'''Sum of contribution facet units

{val_counts}

in total: {val_counts.sum()}
''')

# +
fig = px.bar(
    df.cont.str.split(',').explode().value_counts(),
    y='cont',
    title='Contribution facet distribution')

fig.update_layout(
    xaxis=dict(tickangle=45, title='Contribution facet categories'),
    yaxis=dict(title='Counts')
)

fig.show()
# -

temp = df[df.cont.str.contains('model','literature')]##.title#.head()
temp[temp.research == 'philosophical'].title

# +
# DF of counted units per facet


# res_cont = df.cont.str.split(',').explode().value_counts().to_frame(
#     'count').rename_axis('cont').reset_index()

# res_research = df.research.str.split(',').explode().value_counts().to_frame(
#     'count').rename_axis('research').reset_index()

# res = pd.concat([res_research, res_cont])
# res
# -

# ## Domain Facet

# +
val_counts = df.domain.str.strip().str.split(',').explode().value_counts()
val_counts


for val in val_counts.keys():
    if val == '':
        raise ValueError('Error with splitting. There are empty parts', val)

print(f'''Sum of domain facet units

{val_counts}

in total: {val_counts.sum()}
''')
# -

top_20_domain_words = df.domain.str.split(',').explode().value_counts()[:20]

# +
fig = px.bar(
    top_20_domain_words,
    y='domain',
    title='Domain facet distribution')

fig.update_layout(
    xaxis=dict(tickangle=45, title='Domain facet categories'),
    yaxis=dict(title='Counts')
)

fig.show()
# -

# # Explore interaction of research and contribution facets

research_cols

# +
applied_research_cols = df['research'].value_counts().keys()

if len(research_cols) > len(applied_research_cols):
    research_cols = applied_research_cols.to_list()
    print(f'''
Only found the following research facet cols:
{research_cols}
''')
# -

# ## Correlations 
#
# Sanity check of facets by correlation with crosstab with exploded content (1-hot encoded)

# +
exploded_cont = df['cont'].str.get_dummies(sep=',')
exploded_research = df['research'].str.get_dummies(sep=',')
exploded_cont

exploded = pd.concat([df, exploded_cont, exploded_research], axis=1)



exploded_facets = exploded[cont_cols + research_cols]
print(f'''
Overall sums:
{exploded_facets.sum()}

and total nr of facet units: {exploded_facets.sum().sum()}
''')

# +
corr = exploded_facets.corr()
corr

corr = corr[abs(corr) >= 0.4]

fig = px.imshow(corr, text_auto=True, aspect='auto')
# fig.update_layout(
#     autosize=True,
# #     width=800,
# #     height=800,
# )
fig.show()
# -

# ## Crosstab between research and contribution
#
# By copying 
# As the research facet is only assigned once per article, we would need to copy the multilabel column  "cont" to create a crosstab.
#
# This also implies that the **absolute numbers** for the research facet are not the ground truth. 

# +
test = df.assign(cont=df.cont.str.split(',')).explode('cont')
test

cta = pd.crosstab(test.cont, test.research, margins=True)
cta

# +
# cta = pd.crosstab(
#     index=[
#         df.assign(research=df.research.str.split(',')).explode(
#             'research').reset_index().research
#     ],
#     columns=[
#         df.assign(
#             cont=df.cont.str.split(',')).explode('cont').reset_index().cont
#     ],
#     margins=True)

# cta
# -

# ### Absolute numbers

ct = cta[cta.columns[:-1]].iloc[:-1]
ct

# #### Visualizations

fig = px.imshow(ct, text_auto=True, aspect='auto', title='Contribution without totals')
fig.show()

# +
#replaced_all_cell_for_heatmap = cta.replace(to_replace = cta.iloc[-1,-1], value = 'All', inplace=False)

fig = px.imshow(cta.iloc[:-1], text_auto=True, aspect='auto', title='Contribution with totals')
# fig = px.imshow(cta, text_auto=True)
fig.show()
# -

fig = px.bar(ct)  #, barmode='group')
fig.show()

# +
data = []
#use for loop on every zoo name to create bar data
for x in ct.columns:
    data.append(go.Bar(name=str(x), x=ct.index, y=ct[x]))

figure = go.Figure(data)
# figure.update_layout(barmode = 'stack')

#For you to take a look at the result use
figure.show()
# -

# ### Percent of Totals

cta.pipe(
    lambda x: x.div(x['All'], axis='index')
).applymap('{:.0%}'.format).iloc[:-1]

cta.T.pipe(lambda x: x.div(x['All'], axis='index')).applymap('{:.0%}'.format).iloc[:-1]

cta.describe()
cta.T.describe()


# #### Visualizations

# +
temp = cta.pipe(
    lambda x: x.div(x['All'], axis='index')
).applymap('{:.0%}'.format).iloc[:-1]#, :-1]
test = temp.copy()

for col in test.columns:
    test[col] = test[col].str.rstrip('%').astype('float') / 100.0

fig = px.imshow(test, text_auto=True, aspect='auto', title='Contribution facet in percentages')
fig.show()

# +
temp = cta.T.pipe(lambda x: x.div(x['All'], axis='index')).applymap(
    '{:.0%}'.format).iloc[:-1]  #, :-1]
test = temp.copy()

for col in test.columns:
    test[col] = test[col].str.rstrip('%').astype('float') / 100.0

fig = px.imshow(test,
                text_auto=True,
                aspect='auto',
                title='Research facet in percentages')
fig.show()
# -

# # Explore year

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="count", x=df.year))

# # Explore Authors
#

author_counts = df.author.str.split(',').explode().value_counts()
most_frequent_authors = author_counts[author_counts > 1]

# +
fig = px.bar(
    most_frequent_authors,
    y='author',
    title='Frequent authors')

fig.update_layout(
    xaxis=dict(tickangle=45, title='Author'),
    yaxis=dict(title='Word counts')
)

fig.show()
# -

# # Explore Keywords
#

vc  = df.keywords.str.split(',').explode().value_counts()
top_20 = vc[:20] #(vc[vc > 2])

# +
fig = px.bar(
    top_20,
    y='keywords',
    title='20 most frequent keywords')

fig.update_layout(
    xaxis=dict(tickangle=45, title='Keywords'),
    yaxis=dict(title='Word counts')
)

fig.show()
# -

df.keywords.str.split(',').explode().value_counts()

# # Cluster similar words from keywords and identify groups

# ## Load Model
#

# +
try:
    from gensim.models import KeyedVectors
    model = KeyedVectors.load('models/word2vec-google-news-300.model')
    
except:
    print('Couldnt find a saved model')
    import gensim.downloader as api
    
    model = api.load('word2vec-google-news-300')
    model.save('models/word2vec-google-news-300.model')
# -

# ## Get keywords and compare to vocab in pre-trained model

# +
keywords_expl = df.keywords.str.split(',').explode()
listed_words = keywords_expl 

print(keywords_expl)

# +
words = set(listed_words) & set(list(model.key_to_index.keys()))
vectors = list([model.get_vector(word) for word in words])

len(words), len(vectors)
# -

# ## PCA
#
# --- 
#
# TSNE
#
# https://stats.stackexchange.com/questions/263539/clustering-on-the-output-of-t-sne/264647#264647
#
#
# only reproducable with high perplexity!
#
# discouraged to be used with clustering
#
#
# t-SNE is also a method to reduce the dimension. One of the most major differences between PCA and t-SNE is it preserves only local similarities whereas PA preserves large pairwise distance maximize variance.
#
# - https://medium.com/analytics-vidhya/pca-vs-t-sne-17bcd882bf3d#:~:text=One%20of%20the%20most%20major,large%20pairwise%20distance%20maximize%20variance.&text=It%20takes%20a%20set%20of,it%20into%20low%20dimensional%20data.
#
#

# +
pca = PCA(n_components=2, random_state=RNG)
pca_transformed = pca.fit_transform(vectors)
X_pca = pca_transformed

words = pd.DataFrame(words)
pca_df = pd.DataFrame(pca_transformed)
pca_df = pd.merge(words, pca_df, left_index=True, right_index=True)
pca_df.columns = ['words', 'x', 'y']

pca_df
# -

pca_df[pca_df.words =='Tools']

print(pca_df)

# ## Cluster

# +
NUM_CLUSTERS = 5

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS,
                        random_state=RNG,
                        n_init=1000,
                        max_iter=1000)
kmeans.fit(X_pca)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# print("Cluster id labels for inputted data")
# print(labels)
# print("Centroids data")
# print(centroids)

# print(
#     "Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):"
# )
# print(kmeans.score(X))

# silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

# print("Silhouette_score: ")
# print(silhouette_score)

# +
pca_df['cluster'] = labels
# to make clusters categorical for plotting
pca_df.cluster = pca_df.cluster.astype(str)

pca_df.sort_values(by=['cluster'])
# -

pca_df.cluster.value_counts()

# ## Plot

# +
display_size = 12

fig = px.scatter(
    pca_df,
    x="x",
    y="y",
    color="cluster",
    #                  size='petal_length',
    hover_data=['words'],
    text=pca_df['words'],
    width=1200,
    height=1600)

# fig.update_layout(height=1600, width=1600, title_text='Vector Clusters')
fig.update_traces(textposition='bottom center',
                  textfont_size=display_size,
                  marker={'size': display_size},
                 )

fig.update_layout(
    legend=dict(
        x=0.9,
        y=0.98,
        font=dict(
            size=display_size,)
    )
)

fig.show()
# -

# ## Conclusion
#
# To lay base for the further work on the domain research, I chose 5 clusters and wanted to see if those clusters align with my assumptions on my initially designed domain tables. 
#
# We can see the following, and already associate umbrella terms for the clusters:
#
# ````
# Cluster 0 - BUSINESS:
# ['Companies' 'Industries' 'Business' 'Uncertainty' 'Conferences' 'Ethics'
#  'Buildings' 'Sociology' 'Surgery' 'Interviews']
# Cluster 1 - OPS:
# ['DevOps' 'robustness' 'Autonomic' 'Matlab' 'implementation' 'Azure'
#  'bots' 'SDLC' 'agile' 'deploying']
# Cluster 2 - DATA:
# ['Training' 'Logistics' 'Codes' 'Publishing' 'Timing' 'Pipelines'
#  'Forecasting' 'Measurement' 'Fasteners' 'Organizations']
# Cluster 3 - ML:
# ['MCDA' 'AI' 'Docker' 'management' 'monitoring' 'important' 'LSTM' 'SLR'
#  'MOO' 'fairness']
# Cluster 4 - DEV:
# ['Automation' 'Collaboration' 'Databases' 'Orchestration' 'Framework'
#  'Optimization' 'Middleware' 'Servers' 'Deployment' 'Regression']
#  
# ````
#
#
#
# This indicates that the developed intuition of the first research iteration in regards to designing a reference architecture is promising.
#

NR_SAMPLES = 10
print(f'''
Cluster 0 - BUSINESS:\n{pca_df[pca_df.cluster == '0'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 1 - OPS:\n{pca_df[pca_df.cluster == '1'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 2 - DATA:\n{pca_df[pca_df.cluster == '2'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 3 - ML:\n{pca_df[pca_df.cluster == '3'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 4 - DEV:\n{pca_df[pca_df.cluster == '4'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
''')

# # Cluster similar words from domain facet

# +
keywords_expl = df.domain.str.split(',').explode()
listed_words = keywords_expl 

# print(keywords_expl)

words = set(listed_words) & set(list(model.key_to_index.keys()))
vectors = list([model.get_vector(word) for word in words])

print(len(words), len(vectors), 'words')

# pca = PCA(n_components=2, random_state=RNG)
pca_transformed = pca.fit_transform(vectors)
X_pca = pca_transformed

words = pd.DataFrame(words)
pca_df = pd.DataFrame(pca_transformed)
pca_df = pd.merge(words, pca_df, left_index=True, right_index=True)
pca_df.columns = ['words', 'x', 'y']

# pca_df

NUM_CLUSTERS = 5

kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS,
                        random_state=RNG,
                        n_init=1000,
                        max_iter=1000)
kmeans.fit(X_pca)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

pca_df['cluster'] = labels
# to make clusters categorical for plotting
pca_df.cluster = pca_df.cluster.astype(str)

pca_df.cluster.value_counts()

# +
fig = px.scatter(
    pca_df,
    x="x",
    y="y",
    color="cluster",
    hover_data=['words'],
    text=pca_df['words'],
    width=1200,
    height=1600)

# fig.update_layout(height=1600, width=1600, title_text='Vector Clusters')
fig.update_traces(textposition='bottom center',
                  textfont_size=display_size,
                  marker={'size': display_size},
                 )

fig.update_layout(
    legend=dict(
        x=0.9,
        y=0.98,
        font=dict(
            size=display_size,)
    )
)

fig.show()
# -

# ## Conclusion

NR_SAMPLES = 8
print(f'''
Cluster 0 :\n{pca_df[pca_df.cluster == '0'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 1 :\n{pca_df[pca_df.cluster == '1'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 2 :\n{pca_df[pca_df.cluster == '2'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 3 :\n{pca_df[pca_df.cluster == '3'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
Cluster 4 :\n{pca_df[pca_df.cluster == '4'].sample(n=NR_SAMPLES, random_state=RNG).words.values}
''')

# # Further work
#
# There are many ways on how this base can be used for further work. Consider the following ideas:
#
# - Cluster corpus of abstracts
# - Plot interaction between domain and other facets
# - Built domain model (will be done in master thesis)


