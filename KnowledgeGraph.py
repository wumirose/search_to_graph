import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import utilss
import streamlit as st
import streamlit.components.v1 as components


def get_distinct_node(network, crit):
    labels = set()
    if crit == 'NodeName':
        key = 'label'
    else:
        key = 'title'
    for node in network.nodes:
        if key == 'title':
            labels.add(node[key][0])
        else:
            labels.add(node[key])
    return list(labels)


def get_rel_types():
    rel_ls = []
    rel_type_query = """CALL db.relationshipTypes()"""
    result = utilss.neo4j_utils.query(rel_type_query)
    for el in result:
        rel_ls.append(el[0])
    return rel_ls


def get_edge_types_for_node(network, node_id=None):
    edge_types = set()
    for edge in network.edges:
        if node_id:
            if edge['from'] == node_id or edge['to'] == node_id:
                edge_types.add(edge['label'])
        else:
            edge_types.add(edge['label'])
    return list(edge_types)


def view_entity_plot():
    parsed = Config.PARSED
    df = pd.DataFrame([(''.join(
        [entity['entity_type'][0] if isinstance(entity['entity_type'], list) else entity['entity_type']]),
                        entity['entity_label'].capitalize())
        for entitie in parsed
        for entity in entitie.get('entities', [])],
        columns=['entity_type', 'entity_label'])

    plt.figure(figsize=(20, 5))
    x_order = df['entity_label'].value_counts().sort_values(ascending=False).index[:50]  # Pick top 50
    ax = sns.countplot(x="entity_label", hue="entity_type", dodge=False,
                       order=x_order,
                       data=df, palette="Set2")
    for p in ax.patches:
        ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.01))
    plt.xticks(rotation=90)
    plt.legend(loc='upper right', ncol=2)  # Set ncol=2 to break legend into two columns
    plt.xlabel('Entities')
    plt.ylabel('Counts')
    st.pyplot()


def main_kg():
    tab1, tab2, tab3 = st.tabs(["Entities", "Triples Table", 'Graph'])
    utilss.get_parsed()
    with tab1:
        view_entity_plot()
    with tab2:
        df1 = pd.DataFrame(
            [(entity['subject'].capitalize(), entity['predicate'], entity['object'].capitalize())
             for entitie in Config.PARSED
             if 'triples' in entitie
             for entity in entitie['triples']],
            columns=['subject', 'predicate', 'object'])
        st.dataframe(df1)
    with tab3:
        tab1, tab2 = st.tabs(['Visualize_Subgraph', "Visualize_Graph"])
        with tab1:
            plotnetworkx()
        with tab2:
            utilss.plotnetwork(Config.PARSED, shownetwork=True, physics=True)
            HtmlFile = open("my_network.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=1200, width=1000)
            # # get_rel_types()
            # st.write(get_edge_types_for_node(network, node_id=None))


def plotnetworkx():
    network = utilss.plotnetwork(Config.PARSED, physics=True)
    search = st.selectbox('search by:', ('NodeName', 'NodeType'))
    if search:
        lists = get_distinct_node(network, search)
        selected = st.multiselect('Select subject/Object to visualize', lists)
        if len(selected) == 0:
            st.warning('Choose at least 2 search item to start')
        elif len(selected) == 2:
            search1, search2 = selected
            title = f'{search1} - {search1} Relationship'
            utilss.plot_node_type_relationship(network, search1, search2, title)
            HtmlFile = open(f"{title}.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=1200, width=1000)
