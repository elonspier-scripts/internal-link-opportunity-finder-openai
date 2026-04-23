import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import io

# Pagina instellingen
st.set_page_config(page_title="Semantic Link Finder (OpenAI)", layout="wide")

st.title("🔗 Semantic Link Finder")
st.markdown("Vind relevante interne link kansen op basis van semantische relevantie via OpenAI.")

# ========================================================
# 1. SIDEBAR CONFIGURATIE
# ========================================================
st.sidebar.header("🔑 API Instellingen")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_KEY", ""))

st.sidebar.divider()
st.sidebar.header("⚙️ Analyse Instellingen")
min_score = st.sidebar.slider("Minimale Similarity Score (%)", 50, 95, 80) / 100
max_links = st.sidebar.slider("Max targets per URL", 1, 10, 5)

# ========================================================
# 2. HELPERS
# ========================================================
def clean_url_for_text(url):
    """Maakt van een URL-slug leesbare woorden voor de AI."""
    path = url.split('/')[-1] if not url.strip().endswith('/') else url.split('/')[-2]
    clean_path = re.sub(r'[-_/]', ' ', path)
    return clean_path

def get_embeddings(text_list, api_key):
    """Genereert embeddings via OpenAI text-embedding-3-small."""
    client = OpenAI(api_key=api_key)
    model_name = 'text-embedding-3-small' 
    
    embeddings = []
    progress_bar = st.progress(0)
    total_items = len(text_list)
    
    batch_size = 100
    
    for i in range(0, total_items, batch_size):
        batch = text_list[i:i+batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=model_name
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            if "429" in str(e):
                st.error("⚠️ Rate Limit bereikt bij OpenAI. Controleer je quota of wacht even.")
            else:
                st.error(f"OpenAI API Fout: {e}")
            st.stop()
            
        progress_bar.progress(min((i + batch_size) / total_items, 1.0))
    
    progress_bar.empty()
    return np.array(embeddings)

def extract_topical_category(text):
    """Extraheert hoofdonderwerpen uit de tekst voor de 'Category' kolom."""
    words = re.findall(r'\w{4,}', str(text).lower())
    stop_words = {'deze', 'voor', 'naar', 'met', 'door', 'geen', 'over', 'mijn', 'niet', 'eenvoudig'}
    filtered = [w for w in words if w not in stop_words]
    unique_words = list(dict.fromkeys(filtered))
    return " / ".join(unique_words[:2]).upper() if unique_words else "ALGEMEEN"

# ========================================================
# 3. HOOFDSCHERM INPUT
# ========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Stap 1: Website Data")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

with col2:
    st.subheader("Stap 2: Focus URL's")
    focus_urls_input = st.text_area("Focus URL's (één per regel)", height=150, placeholder="https://jouwsite.nl/doelpagina")

# ========================================================
# 4. DE ANALYSE RUNNER
# ========================================================
if st.button("🚀 Start Analyse", type="primary"):
    if not openai_key:
        st.error("⚠️ Voer een OpenAI API Key in.")
    elif not uploaded_file or not focus_urls_input:
        st.error("⚠️ Upload een CSV en voer Focus URL's in.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            url_col = df.columns[0]
            content_cols = df.columns[1:]
            focus_urls = [x.strip() for x in focus_urls_input.split('\n') if x.strip()]
            
            df = df[df[url_col].str.strip() != ""].copy()
            df['url_text'] = df[url_col].apply(clean_url_for_text)
            df['combined_text'] = df.apply(
                lambda row: str(row['url_text']) + ' ' + ' '.join(row[content_cols].values.astype(str)), 
                axis=1
            )
            df['Topical_Category'] = df['combined_text'].apply(extract_topical_category)

            st.info(f"⚙️ Bezig met genereren van embeddings...")
            vectors = get_embeddings(df['combined_text'].tolist(), openai_key)
            sim_matrix = cosine_similarity(vectors)

            results = []
            for focus_url in focus_urls:
                if focus_url not in df[url_col].values:
                    continue

                idx_source = df.index[df[url_col] == focus_url].tolist()[0]
                source_topic = df.iloc[idx_source]['Topical_Category']
                similarities = sim_matrix[idx_source]
                sorted_indices = np.argsort(similarities)[::-1]
                
                count = 0
                for idx_target in sorted_indices:
                    target_url = df.iloc[idx_target][url_col]
                    score = float(similarities[idx_target])

                    if focus_url != target_url and score >= min_score:
                        results.append({
                            'Category': source_topic, # Altijd vullen voor tab-filtering
                            'Focus URL': focus_url,   # Altijd vullen voor tab-filtering
                            'Interne Link Kans': f"{target_url}",
                            'Score (%)': round(score * 100)
                        })
                        count += 1
                        if count >= max_links: break

            if results:
                all_results_df = pd.DataFrame(results)
                st.success(f"✅ Analyse voltooid!")

                # --- SECTIE 1: HOOFDTABEL ---
                st.subheader("📊 Volledig Overzicht")
                
                # Schoon de DataFrame op voor weergave (verberg herhalingen)
                display_df = all_results_df.copy()
                display_df.loc[display_df.duplicated(['Focus URL']), ['Category', 'Focus URL']] = ""

                def apply_color(val):
                    if isinstance(val, (int, float, np.integer)):
                        if val <= 60: return 'color: #dc3545; font-weight: bold;'
                        elif val <= 84: return 'color: #ffc107; font-weight: bold;'
                        else: return 'color: #28a745; font-weight: bold;'
                    return ''

                st.dataframe(
                    display_df.style.map(apply_color, subset=['Score (%)']),
                    use_container_width=True,
                    column_config={"Score (%)": st.column_config.NumberColumn(format="%d%%")}
                )

                st.divider()

                # --- SECTIE 2: CATEGORIE MATRIX (TABS) ---
                st.subheader("📁 Analyse per Categorie Matrix")
                unique_categories = sorted(all_results_df['Category'].unique())
                
                if unique_categories:
                    tabs = st.tabs(unique_categories)
                    for i, cat in enumerate(unique_categories):
                        with tabs[i]:
                            cat_df = all_results_df[all_results_df['Category'] == cat].copy()
                            # Schoonmaken voor tabel-weergave binnen tab
                            cat_display = cat_df.copy()
                            cat_display.loc[cat_display.duplicated('Focus URL'), 'Focus URL'] = ""
                            
                            st.markdown(f"**Relevante links binnen de categorie: `{cat}`**")
                            st.table(cat_display[['Focus URL', 'Interne Link Kans', 'Score (%)']])

                # --- SECTIE 3: DOWNLOAD ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    all_results_df.to_excel(writer, index=False, sheet_name='Link_Kansen')
                    workbook = writer.book
                    worksheet = writer.sheets['Link_Kansen']
                    wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
                    worksheet.set_column('A:A', 20)
                    worksheet.set_column('B:C', 50, wrap_format)
                    worksheet.set_column('D:D', 15)
                    
                    grn_txt = workbook.add_format({'font_color': '#28a745', 'bold': True})
                    yel_txt = workbook.add_format({'font_color': '#ffc107', 'bold': True})
                    red_txt = workbook.add_format({'font_color': '#dc3545', 'bold': True})

                    worksheet.conditional_format(1, 3, len(all_results_df), 3, {'type': 'cell', 'criteria': '>=', 'value': 85, 'format': grn_txt})
                    worksheet.conditional_format(1, 3, len(all_results_df), 3, {'type': 'cell', 'criteria': 'between', 'minimum': 61, 'maximum': 84, 'format': yel_txt})
                    worksheet.conditional_format(1, 3, len(all_results_df), 3, {'type': 'cell', 'criteria': '<=', 'value': 60, 'format': red_txt})
                
                st.download_button(
                    label="📥 Download Resultaten (Excel)",
                    data=output.getvalue(),
                    file_name="interne_link_kansen_openai.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Geen relevante matches gevonden.")
        except Exception as e:
            st.error(f"Fout tijdens de verwerking: {str(e)}")
