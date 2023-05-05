import sys
import re
import random
import json
import requests
import pandas as pd
import hashlib
import numpy as np
import seaborn as sns
from tqdm import tqdm
from bs4 import BeautifulSoup
from itertools import combinations
import matplotlib.pyplot as plt
import visualise_spacy_tree
from copy import deepcopy
from collections import defaultdict, Counter

from entity_options import get_entity_options
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import streamlit as st
# from spacy.matcher import Matcher
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

nltk.download('punkt')
nltk.download('stopwords')
from utilss import *
import spacy
from spacy.tokens import Token
from spacy import displacy
# from transformers import AutoTokenizer
from conf.change_config import addtional_config_chamge

sys.path.insert(0,
                '/Users/olawumiolasunkanmi/Library/CloudStorage/OneDrive-UniversityofNorthCarolinaatChapelHill/FALL2022/BACKUPS/GraphProjects/COVIDHere/')

# Step 1: Preprocessing the text
stop_words = set(stopwords.words('english'))
EN = spacy.load('en_ner_bionlp13cg_md')
sw_spacy = EN.Defaults.stop_words
stop_words.update(sw_spacy)
stop_words.update({'Table', 'Figure', 'fig', 'sev', 'et.', 'et', 'al.', 'al', 'i.e.', 'ie.,', 'Title', 'etc.'})


def show_named_entities(paper, styl):
    if styl == 'dep':
        text = ' '.join(paper.split('. ')[:1])
        doc = EN(text)
        st.divider()
        # ', '.join(get_relation(paper)[:4])
        st.markdown(f"SAMPLE SENTENCE:  {text}... ")
        st.divider()
        # display(Image(png))
        # dep_svg = displacy.render(doc, style=styl, jupyter=False)

        from spacy.tokens import Token
        Token.set_extension('plot', default={}, force=True)  # Create a token underscore extension
        for it, token in enumerate(doc):
            if token.text not in stop_words:
                if token.dep_ not in ['punct']:
                    node_label = '{0} [{1}]'.format(token.orth_, token.i)
                    token._.plot['label'] = node_label
                    if token.dep_ in ['ROOT', 'acl']:
                        token._.plot['color'] = 'dodgerblue'
                    if token.dep_ in ['nsubj', 'dobj']:
                        token._.plot['color'] = 'blue'
                    if token.dep_ in ['nsubj'] and (
                            doc[it + 1].dep_ in ['compound'] or doc[it - 1].dep_ in ['compound']):
                        token._.plot['color'] = 'blue'

        png = visualise_spacy_tree.create_png(doc)
        st.image(png, caption='Dependency tree', use_column_width='never')
    elif styl == 'ent':
        doc = EN(paper)
        ent_html = displacy.render(doc, style=styl, options=get_entity_options(random_colors=True), jupyter=False)
        st.markdown(ent_html, unsafe_allow_html=True)


def randomize_list(input_list):
    """
    Returns a randomized version of the input list.
    """
    randomized_list = input_list.copy()
    random.shuffle(randomized_list)
    return randomized_list


@st.cache_data
def search_get_pmc(query, retmax):
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    param query: the search keyword eg 'covid genes'
    param retmax: the maximum response required, though 70% is returned
    param path: location to save the PMCIDs
    param summary_mode: 'dataframe' or 'dict'
    return: Dataframe or dictionary structure of the search results

    """
    db = 'pmc'
    results = defaultdict(list)
    count = 0
    wc = []
    rets = retmax + 10
    retmode = 'json'
    search_url = 'https://www.ncbi.nlm.nih.gov/'
    domain = search_url + 'entrez/eutils'
    query_link_search = f'{domain}/esearch.fcgi?db={db}&retmax={rets}&retmode={retmode}&term={query}'
    response = requests.get(query_link_search)
    pubmed_json = response.json()

    tags = ['xref', 'title', 'table', 'fig', 'back']
    if pubmed_json['esearchresult'].get('idlist'):
        # Downloading Document Summaries
        randomized_pid_list = randomize_list(pubmed_json["esearchresult"]["idlist"])
        for paperId in randomized_pid_list:
            query_link_summary = f'{domain}/esummary.fcgi?db={db}&id={paperId}&retmode={retmode}'
            if count < retmax:
                summary = requests.get(query_link_summary).json()['result'][paperId]
                fulltext_url = search_url + 'pmc/oai/oai.cgi'
                query_full_text = f'{fulltext_url}?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{paperId}&metadataPrefix={db}'
                fulltext = requests.get(query_full_text)
                soup = BeautifulSoup(fulltext.text, features='xml')
                try:
                    meta = soup.GetRecord.record.metadata
                    for tg in tags:
                        for t in meta(tg):
                            t.decompose()
                    kwd = ', '.join(kw.text for kw in meta('kwd-group') if kw)
                    wc.extend(list(filter(None, kwd.strip().split('\n'))))

                    body = meta.body
                    body_fulltexts = body.getText().encode().decode('ascii', 'ignore')
                    body_fulltext = re.sub(r"\b(?!(?:Although|Also)\b)(?:[A-Z][A-Za-z'`-]+)(?:,? (?:(?:and |& )?" +
                                           "(?:[A-Z][A-Za-z'`-]+)|(?:et al.?)))*(?:, *(?:19|20)[0-9][0-9]" +
                                           "(?:, p\.? [0-9]+)?| *\((?:19|20)[0-9][0-9](?:, p\.? [0-9]+)?\))", '',
                                           body_fulltexts)
                    body_fulltext = re.sub(r"(\[[^][]*]|\d+( $))", '', body_fulltext)
                    body_fulltext = re.sub(r"[A-Z]\w*(?: et al)", '', body_fulltext)
                    body_fulltext = re.sub(r"^[\d\-.]+", '', body_fulltext, flags=re.MULTILINE)
                    body_fulltext = ' '.join([x for x in body_fulltext.split(' ') if x not in stop_words])
                    body_fulltext = ' '.join(
                        [x.strip() for x in body_fulltext.split(' ') if
                         x not in [i['name'] for i in summary['authors']]])

                    body_fulltext = re.sub(r'\(\s*\)', '', body_fulltext)

                    results['PMC' + paperId].append({'paperId': 'PMC' + paperId, 'pubdate': summary['pubdate'],
                                                     'authors': [i['name'] for i in summary['authors']],
                                                     'title': summary['title'],
                                                     'fulljournalname': summary['fulljournalname'],
                                                     'fulltext': ''.join([k.strip('\n') for k in body_fulltext]),
                                                     'No of words': len(body_fulltext.split())})
                    count += 1
                except AttributeError:
                    pass

        ls = [list(rs[0].values()) for rs in results.values()]
        df = pd.DataFrame(ls, columns=['paperId', 'pubdate', 'authors', 'title', 'fulljournalname', 'fulltext',
                                       'No of words'])
        return wc, df, json.dumps(results, indent=4)
    else:
        st.error(pubmed_json)


#
# @st.cache
# def generate_data(searchquery, retmax):
#     return search_get_pmc(searchquery, retmax)

# @st.cache
def plot_stats(col):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    sns.set(rc={'figure.figsize': (20, 12)})
    sns.set_style("whitegrid")
    colors = plt.cm.PuBu_r(np.linspace(0, 0.5, len(col)))
    ax.barh(col.index, col, align="center", color=colors, tick_label=col.index)
    plt.xlabel('Count')
    plt.ylabel('Journals')
    st.pyplot(fig)


@st.cache_data
def preprocess_text(text, n_gram):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuations
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words and len(token) > 1]
    if n_gram == 1:
        return tokens
    else:
        bigrams = list(ngrams(tokens, n_gram))
        # Combine unigrams and bigrams
        tokens = ['_'.join(bigram) for bigram in bigrams]
        return tokens


@st.cache_data
def calculate_word_frequency(text, n_gram):
    tokens = preprocess_text(text, n_gram)
    freq_dict = Counter(tokens)
    return freq_dict


@st.cache_data
def plot_word_frequency(freq_dict, title, source=None):
    if isinstance(freq_dict, dict):
        df = pd.DataFrame(freq_dict.items(), columns=['Word', 'Frequency'])
    else:
        df = freq_dict
    df['Frequency'] = df['Frequency'].astype(int)
    df.sort_values('Frequency', ascending=False, inplace=True)
    source = ColumnDataSource(df) if source is None else source
    p = figure(x_range=df['Word'], height=350, width=600, title=title)
    p.vbar(x='Word', top='Frequency', width=0.9, source=source)
    p.xaxis.major_label_orientation = "vertical"
    hover = HoverTool(tooltips=[('Word', '@Word'), ('Frequency', '@Frequency')])
    p.add_tools(hover)
    # show(p)
    # Convert the Bokeh plot to a Streamlit figure using st.bokeh_chart
    st.bokeh_chart(p, use_container_width=True)


# # Plot the top 10 frequent words in each article
# @st.cache_data
# def plot_singlepaper(text, n_gram, title=None):
#     freq_dict = calculate_word_frequency(text, n_gram)
#     plot_word_frequency(freq_dict, title)

# Plot the top 10 frequent words across all articles
@st.cache_data
def plot_multiplepaper(articles, n_gram):
    all_words = Counter()
    if isinstance(articles, list):
        for text in articles:
            freq_dict = calculate_word_frequency(text, n_gram)
            all_words += freq_dict
    elif isinstance(articles, str):
        freq_dict = calculate_word_frequency(articles, n_gram)
        all_words += freq_dict
    try:
        all_words
        if all_words:
            df = pd.DataFrame(list(all_words.items()), columns=['Word', 'Frequency'])
            df['Frequency'] = df['Frequency'].astype(int)  # convert to int
            df = df.sort_values('Frequency', ascending=False)
            df = df.head(30)
            word_freq = pd.pivot_table(df, values='Frequency', index=['Word'], aggfunc=np.sum)
            word_freq = word_freq.reset_index()
            if n_gram == 1:
                head = 'Top 30 frequent words across article(s)'
            else:
                word_freq['Word'] = word_freq['Word'].str.replace('_', ' ')
                head = f'Top 30 frequent {n_gram}-gram words across article(s)'
            plot_word_frequency(word_freq, head)
    except ValueError as e:
        print(e)


def view_entity_df(parsed_entities):
    dataset = make_df(parsed_entities)
    dataset.drop_duplicates(subset='entity', ignore_index=True)
    return dataset


# def read_url(url):
# '''Read URL CONTENTS'''
# if url.endswith('pdf'):
#     pdf = requests.get(url,  timeout = 30)
#     doc = pdf2image.convert_from_bytes(pdf.content)

#     # Get the article text
#     article = []
#     for i, data in enumerate(doc):
#         txt = pytesseract.image_to_string(data).encode("utf-8")
#         article.append(txt.decode("utf-8"))
#     print('Total Pages:', i)
#     article_txt = " ".join(article)
# else:
#     res = requests.get(url)
#     doc = res.text
#     soup = BeautifulSoup(doc, 'html5lib')
#     for script in soup(["script", "style", 'aside']):
#         script.extract()
#     blog_txt = " ".join(re.split(r'[\n\t]+', soup.get_text()))
#     article_txt = nlp(blog_txt)
#     print('Entity lenghth:', len(article_txt.ents))
# return article_txt


def getcase(base_strg, new_strg):
    if base_strg.islower():
        new_strg = new_strg.lower()
    elif base_strg.isupper():
        new_strg = new_strg.upper()
    elif base_strg.capitalize():
        new_strg = new_strg.capitalize()
    return new_strg


def query_raw(text, url="http://bern2.korea.ac.kr/plain"):
    """BERN Biomedical entity linking API"""
    return requests.post(url, json={'text': text}, verify=False).json()


def make_df(parsed_entities):
    """Turn the parsed entity to a pandas frame
    from BERN2 Json format"""
    df = pd.json_normalize(parsed_entities)
    d = {k: [] for k in list(df['entities'][0][0].keys())}
    d['text'], d['timestamp'], d['text_sha256'] = [], [], []
    for rowid, entities in enumerate(df['entities']):
        if type(entities) == list:
            text = df['text'][rowid]
            timestamp = df['timestamp'][rowid]
            text_sha256 = df['text_sha256'][rowid]
            for entity in entities:
                for k in list(entity.keys()):
                    d[k].append(entity[k])
                d['text'].append(text)
                d['timestamp'].append(timestamp)
                d['text_sha256'].append(text_sha256)
    dataset = pd.DataFrame(d)
    return dataset

import matplotlib.pyplot as plt
from pyvis.network import Network
import pandas as pd
import streamlit as st

# @st.cache_data
def plotnetwork(data, physics=None):

    # create a new pyvis network
    net = Network(height='750px', directed=True, width='100%', notebook=False)

    # loop through each dictionary in the data list
    for d in data:

        if 'entities' not in d:
            continue

        entities = d['entities']
        triples = d['triples']

        # add nodes to the network
        for entity in entities:
            entity_id = entity['entity_id']
            entity_label = entity['entity_label']
            entity_type = entity['entity_type']
            equivalent_identifiers = entity["equivalent_identifiers"]
            equivalent_labels = entity["equivalent_labels"]
            entity_proba = entity["entity_proba"]

            net.add_node(entity_id, label=entity_label, title=entity_type,
                         equivalent_identifiers=equivalent_identifiers, equivalent_labels=equivalent_labels,
                         entity_proba=entity_proba)

        # add edges to the network
        for triple in triples:
            # print(triple['subject'], list(net.nodes)[0])
            subject = [node['id'] for node in net.nodes if node.get('id') == triple['subject']]
            obj = [node['id'] for node in net.nodes if node.get('id') == triple['object']]
            #
            # subject = net.nodes[triple['subject']]
            predicate = triple['predicate']
            # obj = net.nodes[triple['object']]
            net.add_edge(subject[0], obj[0], label=predicate)

    # color nodes based on their entity_type
    entity_type_colors = {
        'biolink:PopulationOfIndividualOrganisms': '#00FF00',
        'biolink:OrganismalEntity': '#9370DB',
        'biolink:Entity': '#FF00FF',
        'biolink:SubjectOfInvestigation': '#C0C0C0',
        'biolink:ThingWithTaxon': '#800080',
        'biolink:Cohort': '#008000',
        'biolink:NamedThing': '#808080',
        'biolink:StudyPopulation': '#0000FF',
        'biolink:Disease': '#FF0000',
        'biolink:BiologicalEntity': '#FFC0CB',
        'biolink:DiseaseOrPhenotypicFeature': '#FF0001',
        'biolink:ChemicalEntity': '#008000',
        'biolink:ChemicalMixture': '#3CB371',
        'biolink:MolecularEntity': '#00FF00',
        'biolink:Gene': '#00FF12',
        'gene': '#00FF12',
        'biolink:Drug': '#006400',
        'drug': '#007700',
        'biolink:Polypeptide': '#FFA500',
        'biolink:AnatomicalEntity': '#4B0082',
        'biolink:GenomicEntity': '#9371DB',
        'biolink:BiologicalProcess': '#4B0084',
        'species': '#C2F0C2',
        'DNA': '#90EE90',
        'disease': '#FF1111',
        'RNA': '#00FF22',
        'biolink:PlanetaryEntity': '#FF4444',
        'biolink:CellLine': '#FF7777',
        'cell_line': '#FFAA00',
        'cell_type': '#FFCC00',
        'biolink:Mutation': '#00FF13',
        'mutation': '#00FF44',
        'biolink:InformationContentEntity': '#008000',
        'biolink:AdministrativeEntity': '#008001'}

    for node in net.nodes:
        node_type = node['title']
        if isinstance(node_type, list):
            node_col = node_type[0]
        else:
            node_col = node_type
        color = entity_type_colors.get(node_col, '#7f7f7f')  # use gray as the default color
        node['color'] = color
    if physics:
        net.show_buttons(filter_=['physics'])
    # show the network
    net.save_graph('my_network.html')
    return net


def plot_node_type_relationship(network, search1, search2, title):
    # Create a subgraph with nodes of the given types
    subgraph = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=False)
    if 'biolink' in search1 and 'biolink' in search2:
        key = 'label'
    else:
        key = 'title'
    for node in network.nodes:
        if node[key] == search1 or node[key] == search2:
            subgraph.add_node(node['id'], label=node['label'], title=node['title'], color=node['color'])
    # print(node)
    for edge in network.edges:
        source_node = subgraph.get_node(edge['from'])
        target_node = subgraph.get_node(edge['to'])
        if source_node is not None and target_node is not None:
            subgraph.add_edge(edge['from'], edge['to'], label=edge['label'])

    # Set title and save to file
    subgraph.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 20
                }
            }
        }
    """)
    subgraph.show_buttons(filter_=['physics'])
    subgraph.set_options("""
        var options = {
            "interaction": {
                "hover": true
            },
            "manipulation": {
                "enabled": true
            },
            "physics": {
                "barnesHut": {
                    "springConstant": 0.04,
                    "avoidOverlap": 1
                },
                "minVelocity": 0.75
            },
            "layout": {
                "improvedLayout": true
            },
            "height": "600px",
            "width": "100%"
        }
    """)
    subgraph.save_graph(title)

# import torch
import spacy
from spacy.matcher import Matcher
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from neo4j_conn import Neo4jConn
# from transformers import pipeline
from itertools import islice
import time
from config import Config

lemmatizer = WordNetLemmatizer()
EN = spacy.load('en_ner_bionlp13cg_md')
URI = "bolt://44.211.229.233:7687"
USER = "neo4j"
PWD = "polls-introduction-distance"
neo4j_utils = Neo4jConn(uri=URI, user=USER, pwd=PWD)

NODENORM_URL = 'https://nodenormalization-sri.renci.org/get_normalized_nodes'
NAME_RESOLUTION_URL = 'https://name-resolution-sri.renci.org/lookup'
PLAIN_URL = 'http://bern2.korea.ac.kr/plain'

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # Load the zero-shot classification pipeline
# zero_shot_classifier = pipeline("zero-shot-classification", device=device)


def normalize(il):
    input = {'curies': il}
    results = requests.post(NODENORM_URL, json=input)
    return results.json()


def rearrange_dict(input_dict):
    output_dict = {}
    id_dict = input_dict[next(iter(input_dict))]
    output_dict['entity_id'] = id_dict['id']['identifier']
    output_dict['entity_label'] = id_dict['id'].get('label', '')
    equivalent_identifiers = set(id.get('identifier', '') for id in id_dict['equivalent_identifiers'])
    equivalent_labels = set(id.get('label', '') for id in id_dict['equivalent_identifiers'] if id.get('label', ''))
    output_dict['equivalent_identifiers'] = [id_dict['id']['identifier']] + list(
        equivalent_identifiers - {id_dict['id']['identifier']})
    output_dict['equivalent_labels'] = list(equivalent_labels - {output_dict['entity_label']})
    output_dict['entity_type'] = id_dict['type']
    return output_dict


def resolve_name(query):
    res = requests.post(f"{NAME_RESOLUTION_URL}?string={query}&offset=0&limit=1000")
    if res.status_code == 200:
        response = res.json()
        if any(query.lower() in s.lower() for v in response.values() for s in v):
            keys = (k for k, v in response.items() if any(query.lower() == s.lower() for s in v))
            keys = list(islice(keys, 10))  # limit the number of items returned to 10
            if keys:
                return rearrange_dict(normalize(keys))
        else:
            return None
    else:
        print(res.status_code)
        return None


def send_post_request(payload, timeout=30, retries=1):
    """
    Sends a POST request to the specified URL with the given payload.
    If the request times out, retries up to the specified number of times
    with an exponential backoff strategy.

    Args:
        url (str): The URL to send the POST request to.
        payload (str): The payload to include in the POST request (must be less than 5000 characters).
        timeout (int): The timeout limit for the request (in seconds). Default is 30 seconds.
        retries (int): The maximum number of retries to attempt. Default is 3 retries.

    Returns:
        The server response, or None if all retries failed.
    """
    responses = []
    if len(payload) > 5000:
        # print("Payload is too long. Splitting into smaller chunks...")
        chunks = [payload[i:i + 5000] for i in range(0, len(payload), 5000)]
    else:
        chunks = [payload]

    for i, chunk in enumerate(chunks):
        for j in range(retries):
            try:
                response = requests.post(PLAIN_URL, json={"text": chunk}, timeout=timeout, verify=False)
                if response.ok:
                    responses.append(response.json())
                    break
                else:
                    print(f"Request failed with status code {response.status_code}. Retrying...")
            except requests.exceptions.Timeout:
                if j == retries - 1:
                    print("All retries failed. Request timed out.")
                    return None
                else:
                    print(f"Request timed out. Retrying in {2 ** j} seconds...")
                    time.sleep(2 ** j)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                return None

        if i < len(chunks) - 1:
            time.sleep(1)  # Add a delay between chunk requests to avoid overwhelming the server
    return responses


@st.cache_data
def parse_post_request(payload):
    if isinstance(payload, str):
        responses = send_post_request(payload, timeout=30, retries=1)
    if isinstance(payload, list):
        responses = []
        for payld in payload:
            responses.extend(send_post_request(payld, timeout=30, retries=1))

    resp = json.dumps(responses, indent=4)
    with open('responses.json', "w+") as f:
        f.write(resp)
    print('send_post_request done!')
    return parse_entities(responses)


def parse_entities(responses):
    id_name_mapping = {}
    parsed_entities = []
    for ix, entities in enumerate(responses):
        text = entities['text']
        e = []
        if not entities.get('annotations'):
            # parsed_entities.append({'text': text, 'timestamp': entities['timestamp'],
            #                         'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
            continue
        for entity in entities['annotations']:
            entity_type = entity['obj']
            entity_name = entity['mention']
            entity_proba = entity['prob']
            try:
                entid = entity['id']
                if len(entid) < 1:
                    entity_id = entity_name
                elif len(entid) == 1:
                    entity_id = entid[0]
                else:
                    entity_id = ''.join(entid)
                id_name_mapping[entity_id] = entity_name
            except IndexError:
                entity_id = entity_name
            if entity_name not in id_name_mapping.values():
                # print(entity_name)
                new_entity_name = id_name_mapping.get(entity_id)
                text = text.replace(entity_name, new_entity_name)
                # print('new text: ', text)
                entity_name = new_entity_name
            else:
                entity_name = entity_name

            temp = resolve_name(entity_name)
            if temp:
                if entity_id not in temp['equivalent_identifiers']:
                    temp['equivalent_identifiers'].append(entity_id)
                    temp['entity_type'].append(entity_type)
                    temp.update({'entity_proba': entity_proba})
                else:
                    temp['equivalent_identifiers'].append(entity_id)
                    temp['entity_type'].append(entity_type)
                    temp.update({'entity_proba': entity_proba})
                e.append(temp)
        parsed_entities.append({'entities': e, 'text': text, 'triples': get_triples(e, text),
                                'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
        print(f'response: {ix} of {len(responses)} done')
    qry = json.dumps(parsed_entities, indent=4)
    with open('parsed_entity.json', "w+") as f:
        f.write(qry)
    return parsed_entities


def get_parsed():
    payload = list(Config.DF.fulltext)
    parsed = parse_post_request(payload[:2])
    addtional_config_chamge(parsed)
    # gg = open('parsed_entity1.json')
    # parsed = json.load(gg)


def get_p_relation(paper):
    doc = EN(paper)
    # Matcher class object
    matcher = Matcher(doc.vocab)
    spans = []
    # define the rule
    rule = [{'DEP': 'ROOT'},
            {'DEP': 'prep', 'OP': "?"},
            {'DEP': 'agent', 'OP': "?"},
            {'POS': 'VERB', 'OP': "?"},
            {'POS': 'VERB', 'OP': "?"},
            {'DEP': 'prt', 'OP': "?"},
            {'DEP': 'prep', 'OP': "?"},
            {'POS': 'ADV', 'OP': "?"},
            {'POS': 'PART', 'OP': "?"}]

    matcher.add("matching", [rule], on_match=None)
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end]  # The matched span
        spans.append(span.text)
    return spans


def get_triples(entities, text):
    id_list = [entity['entity_id'] for entity in entities]
    label_list = [entity['entity_label'] for entity in entities]

    id_pairs = [[id_list[i], id_list[i + 1]] for i in range(len(id_list) - 1)]
    label_pairs = [[label_list[i], label_list[i + 1]] for i in range(len(label_list) - 1)]

    triples = []
    for k, item in enumerate(label_pairs):
        # texts = sent_and_ents(item, textss)
        # if texts:
        item1 = item[0]
        item2 = item[1]
        start_idx = text.find(item1)
        end_idx = text.find(item2) + len(item2)
        if start_idx != -1 and end_idx != -1:
            subsentence = text[start_idx:end_idx].strip()
            rels = get_p_relation(subsentence)
            if not rels:
                triples.append({'subject': id_pairs[k][0], 'predicate': 'rel', 'object': id_pairs[k][1]})
            else:
                candidate_rel = [
                    lemmatizer.lemmatize(rel, pos=wordnet.VERB) if wordnet.synsets(rel, pos=wordnet.VERB) else 'rel' for
                    rel in rels]
                if len(candidate_rel) > 1:
                    for rel in candidate_rel:
                        triples.append({'subject': id_pairs[k][0], 'predicate': rel, 'object': id_pairs[k][1]})
                    # print('rel>1,', item, candidate_rel)
                    # sequence = [f"{text} {label}" for label in candidate_rel]
                    # # Classify the sequence of candidate labels
                    # results = zero_shot_classifier(sequence, candidate_rel)
                    # # Extract the top relation label` `
                    # predicted_label = results[0]
                    # triples.append({'subject':id_pairs[k][0], 'predicate': predicted_label['labels'], 'object': id_pairs[k][1]})
                else:
                    triples.append({'subject': id_pairs[k][0], 'predicate': candidate_rel[0], 'object': id_pairs[k][1]})

    return triples

def get_node_labels():
    label_ls = []
    label_type_query = """CALL db.labels()"""
    result = neo4j_utils.query(label_type_query)
    for el in result:
        label_ls.append(el[0])
    return label_ls
