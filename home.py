import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def do_stuff_on_page_load():
    st.set_page_config(layout="wide")

do_stuff_on_page_load()

st.markdown(f'''
<style>
.appview-container .main .block-container{{
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;    }}
</style>
''', unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('# For each of the NLTK syntactic tags, define your own syntactic tag:')
    nltkTags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','.',',','$']
    corpusTags = ['Con','Q','D','Ex','Fw','P','A','A','A','Ls','Mod','N','N','N','N','PDT','Pos','Pro','Pro','Adv','Adv','Adv','RP','To','Uh','V','V','V','V','V','V','Wh','Wh','Wh','Wh','.',',','$']
    map = {}
    for i in range(len(nltkTags)):
        map[nltkTags[i]] = st.text_input(nltkTags[i],corpusTags[i])

st.title('HASPNeL Syntactic Tagger')
st.markdown('***')

st.markdown("The objective of this web app is to transform utterances in english as a list of strings into strings with the structure: <word>|<categoty> for each word of each utterance. The categorization is based on the Python's library, NLTK.")
st.markdown('### Step 1. Open the sidebar on the left to define your own syntactic categories.')
col1, col2 = st.columns([6,2])
with col1:
    col1.markdown('### Step 2. Define your utterances or upload a .csv file with the following format ->')
with col2:
    with open('data/utterances.csv') as f:
        col2.download_button('Download CSV', f, 'utterances.csv')

option = st.selectbox(
    '',
    ('Define', 'Upload'))

if option == 'Define':
    if 'data' not in st.session_state:
        data = pd.DataFrame({'utterance':[]})
        st.session_state.data = data

    data = st.session_state.data

    st.dataframe(data)

    def add_dfForm():
        row = pd.DataFrame({'utterance':[st.session_state.input_colA]})
        st.session_state.data = pd.concat([st.session_state.data, row],ignore_index=True)


    dfForm = st.form(key='dfForm')
    with dfForm:
        dfColumns = st.columns(1)
        with dfColumns[0]:
            st.text_input('Enter utterances to add them in the dataframe. Reload page to reset.', key='input_colA')
        st.form_submit_button(on_click=add_dfForm)
else:
    uploaded_file = st.file_uploader("Choose a file")
    try:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
    except:
        pass

st.markdown('### Step 3. Push the button to see and download results.')

if st.button('Process'):
    utt = data.iloc[:,0].values
    taggedUtt = []
    for u in utt:
        ut = ''
        text = word_tokenize(u)
        tags = nltk.pos_tag(text)
        for p in tags:
            ut += p[0] + '|' + map[p[1]] + ' '
        taggedUtt.append(ut)
    dft = pd.DataFrame({'utterance':utt,'tagged':taggedUtt})
    st.dataframe(dft)

    dft.to_csv('data/utterancesTagged.csv',index=False)

    with open('data/utterancesTagged.csv') as ff:
        st.download_button('Download CSV', ff, 'utterancesTagged.csv')
    

