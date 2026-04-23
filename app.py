import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import io

# 1. Forceer Black Theme & App Look via CSS
st.set_page_config(page_title="SEO Link Matrix Pro", layout="wide")

st.markdown("""
    <style>
    /* Achtergrond en tekst */
    .stApp { background-color: #000000; color: #ffffff; }
    header, .stSidebar { background-color: #000000 !important; }
    
    /* Matrix styling */
    .stDataFrame { border: 1px solid #333; border-radius: 10px; }
    
    /* Headers en titels */
    h1, h2, h3 { color: #0080ff !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Knoppen */
    .stButton>button { 
        background-color: #0080ff; color: white; border-radius: 5px; 
        width: 100%; border: none; font-weight: bold;
    }
    
    /* Verstop Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.title("🔗 SEO Link Intelligence Matrix")

# ========================================================
# 2. SIDEBAR (CONFIG)
# ========================================================
with st.sidebar:
    st.header("⚙️ App Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")
    min_score = st.slider("Min. Score (%)", 50, 95, 80) / 100
    max_links = st.slider("Links per URL", 1, 10, 5)

# ========================================================
# 3. HELPERS
# ========================================================
def clean_url_for_text(url):
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    return re.sub(r'[-_/]', ' ', path)

def get_embeddings(text_list, api_key):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=text_list, model='text-embedding-3-small')
    return np.array([data.embedding for data in response.data])

def extract_topical_category(text):
    words = re.findall(r'\w{4,}', str(text).lower())
    stop_words = {'deze', 'voor', 'naar', 'met', 'door', 'geen', 'over'}
    filtered = [w for w in words if w not in stop_words]
    unique = list(dict.fromkeys(filtered))
    return " / ".join(unique[:2]).upper() if unique else "ALGEMEEN"

# ========================================================
# 4. INPUT SECTIE
# ========================================================
col_input1, col_input2 = st.columns([1, 1])
with col_input1:
    uploaded_file = st.file_uploader("Upload Website Export (CSV)", type=['csv'])
with col_input2:
    focus_urls_input = st.text_area("Focus URL's", placeholder="Eén URL per regel...")

# ========================================================
# 5. CORE LOGICA
# ========================================================
if st.button("🚀 GENEREER MATRIX & KANSEN"):
    if not (openai_key and uploaded_file and focus_urls_input):
        st.error("Vul alle velden in.")
    else:
        df = pd.read_csv(uploaded_file)
        url_col = df.columns[0]
        focus_urls = [x.strip() for x in focus_urls_input.split('\n') if x.strip()]
        
        # Voorbereiden data
        df = df[df[url_col].str.strip() != ""].copy()
        df['combined_text'] = df[url_col].apply(clean_url_for_text) + " " + df.iloc[:, 1].astype(str)
        df['Category'] = df['combined_text'].apply(extract_topical_category)
        url_to_cat = dict(zip(df[url_col], df['Category']))

        # Embeddings & Similarity
        vectors = get_embeddings(df['combined_text'].tolist(), openai_key)
        sim_matrix = cosine_similarity(vectors)

        results = []
        for focus_url in focus_urls:
            if focus_url not in df[url_col].values: continue
            idx_source = df.index[df[url_col] == focus_url].tolist()[0]
            source_topic = df.iloc[idx_source]['Category']
            
            similarities = sim_matrix[idx_source]
            sorted_indices = np.argsort(similarities)[::-1]
            
            count = 0
            for idx_target in sorted_indices:
                target_url = df.iloc[idx_target][url_col]
                score = float(similarities[idx_target])
                if focus_url != target_url and score >= min_score:
                    results.append({
                        'From': source_topic,
                        'Focus URL': focus_url,
                        'Target URL': target_url,
                        'To': url_to_cat.get(target_url, "ALGEMEEN"),
                        'Score': round(score * 100)
                    })
                    count += 1
                    if count >= max_links: break

        st.session_state['link_results'] = pd.DataFrame(results)

# ========================================================
# 6. INTERACTIEVE MATRIX & OUTPUT
# ========================================================
if 'link_results' in st.session_state:
    res_df = st.session_state['link_results']
    
    # Matrix opbouwen
    matrix = pd.crosstab(res_df['From'], res_df['To'])
    all_cats = sorted(list(set(res_df['From']).union(set(res_df['To']))))
    matrix = matrix.reindex(index=all_cats, columns=all_cats, fill_value=0)

    st.subheader("📊 Cross-Linking Matrix")
    st.info("💡 Klik op een cel in de matrix om de specifieke URL-kansen hieronder te bekijken.")

    # De Matrix tonen met selectie-mogelijkheid
    # De kolommen (Received Categories) staan standaard bovenaan
    selection = st.dataframe(
        matrix.style.background_gradient(cmap='Blues', axis=None),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single_cell"
    )

    # Als er een cel geklikt wordt
    if selection and selection.selection.cells:
        selected_cell = selection.selection.cells[0]
        row_idx = selected_cell['row']
        col_idx = selected_cell['column']
        
        from_cat = matrix.index[row_idx]
        to_cat = matrix.columns[col_idx]
        
        st.divider()
        st.subheader(f"🔗 Kansrijke links: {from_cat} ➔ {to_cat}")
        
        # Filter de resultaten op basis van de klik
        filtered_df = res_df[(res_df['From'] == from_cat) & (res_df['To'] == to_cat)]
        
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['Focus URL', 'Target URL', 'Score']].sort_values('Score', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write("Geen directe links gevonden voor deze selectie.")
    else:
        # Als er niets geklikt is, toon dan de Topic Hubs onder elkaar
        st.divider()
        st.subheader("🏗️ Topic Hubs (Alle resultaten)")
        for cat in sorted(res_df['From'].unique()):
            with st.expander(f"📁 Hub: {cat}"):
                st.dataframe(res_df[res_df['From'] == cat][['Focus URL', 'Target URL', 'Score']], use_container_width=True)
