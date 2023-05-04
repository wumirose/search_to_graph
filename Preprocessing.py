from wordcloud import WordCloud
from utilss import *
from config import Config
import streamlit as st

searchquery = Config.searchquery
retmax = Config.retmax
WC = Config.WC
DF = Config.DF
JSONDOC = Config.JSONDOC
DESCRIPTION = 'Preprocessing'
st.session_state.title = None


def run_process():
    if WC and len(DF) > 1:
        word_could_dict = Counter(WC)
        wcd = WordCloud().fit_words(dict(word_could_dict))
        journals = DF.fulljournalname.value_counts()

        # tab1, tab2, tab3 = st.tabs(["Keywords WordCloud", "Frequent N-gram CountsPlot", "Journal Statistics"])
        col1, col2, col3 = st.columns(3)#(["Keywords WordCloud", "Frequent N-gram CountsPlot", "Journal Statistics"])

        with col1:
            col1.subheader("Keywords WordCloud")
            st.image(wcd.to_array(), caption='Keywords WordCloud')

        with col2:
            col2.subheader(f"Frequent Bi-gram words")
            # n = st.selectbox('Select N', (1, 2, 3))  # st.('Select N', (1, 2, 3))
            # if n:
            plot_multiplepaper(list(DF.fulltext), 2)
            # else:
            #     st.error('Select N')
        with col3:
            col3.subheader("Journal Statistics")
            plot_stats(journals)
        st.dataframe(DF)
        filename = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./>?]', searchquery)
        tab1, tab2 = st.tabs(['Download Dataframe', 'Visualize a Paper'])
        with tab1:
            col1, col2 = st.columns([.5, 1])
            with col1:
                try:
                    st.download_button(
                        label="Download JSON",
                        file_name=f"{filename[0]}.json",
                        mime="application/json",
                        data=JSONDOC,
                    )
                except PermissionError as e:
                    st.error(e)
            with col2:
                try:
                    st.download_button(
                        label="Download data as CSV",
                        data=DF.to_csv(index=False).encode("utf-8"),
                        file_name=f'{filename[0]}.csv',
                        mime='text/csv',
                    )
                except UnicodeEncodeError as e:
                    st.error(e)
        with tab2:
            titles = list(DF["title"].drop_duplicates())
            titles.insert(0, '<Select Title>')
            # models = ('<Select Title>', 'en_core_sci_lg', 'en_ner_bionlp13cg_md', 'en_ner_bc5cdr_md')
            title = st.selectbox("Named Entities (NE)", titles)
            if title != '<Select Title>':
                st.session_state.title = title
                if st.session_state.title is not None:
                    # print('st.session_state.title', st.session_state.title)
                    nerplots()
            else:
                st.error('WARNING: Make Selection')


def nerplots():
    index = DF.index[DF['title'] == st.session_state.title].tolist()[0]
    doc_ = DF["fulltext"][index]
    tab21, tab22, tab23 = st.tabs(["DependencyTree", "Entity visualizer", "N-gram visualizer"])
    with tab21:
        show_named_entities(doc_, 'dep')
    with tab22:
        show_named_entities(doc_, 'ent')
    with tab23:
        plot_multiplepaper(doc_, 2)
