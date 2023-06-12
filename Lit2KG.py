from utilss import *
from conf.change_config import change_config
import streamlit as st
# from KnowledgeGraph import parse_post_request

DB = 'PubMed Central'


def main():
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'expanded'
    st.set_page_config(page_title="Literature to KG!", page_icon="📚", layout="wide")#,
                       # initial_sidebar_state=st.session_state.sidebar_state)
    st.header('Build Knowledge Graph 🔯 from Literature Articles 📚')

    # Sidebar for search query and retmax
    st.sidebar.header(f'Literatures-based KGs Generators\nDatabase: {DB}')
    st.sidebar.markdown("""____""")
    searchquery = st.sidebar.text_input('Enter Your search query', placeholder='eg. COVID-19')
    if searchquery == "":
        st.sidebar.error('No search term added, enter a search term')
    else:
        st.sidebar.markdown(
            f"""
                    You are searching for: {searchquery}
                    """
        )
    retmax = st.sidebar.number_input('The maximum docs to be retrieved?', min_value=0, max_value=100, value=0,
                                     step=5)
    if st.sidebar.button("Submit"):
        WC, DF, JSONDOC = search_get_pmc(searchquery, retmax)
        if JSONDOC:
            change_config(searchquery, retmax, WC, DF, JSONDOC)
        # st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'

            tab1, tab2 = st.tabs(["ExploratoryAnalysis", "Knowledge Graph"])
            with tab1:
                from Preprocessing import run_process
                run_process()
            with tab2:
                from KnowledgeGraph import main_kg
                main_kg()
        else:
            st.write('No search result!')
    else:
        st.sidebar.error('WARNING: refine the search term')


if __name__ == '__main__':
    main()
