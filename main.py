import streamlit as st
from streamlit_searchbox import st_searchbox
from autocomplete import FiltersAutocomplete


def render():
    st.title('Search for filter')

    if 'selected_filters' not in st.session_state:
        st.session_state['selected_filters'] = set()

    if 'autocomplete_render' not in st.session_state:
        st.session_state['autocomplete_render'] = FiltersAutocomplete()

    search = st_searchbox(
        st.session_state['autocomplete_render'].autocomplete,
        clear_on_submit=True,
        clearable=True,
        key="search filter"
    )

    if search:
        st.session_state['selected_filters'].add(search)
        for elem in st.session_state['selected_filters']:
            st.checkbox(elem, value=True)


if __name__ == "__main__":
    render()
