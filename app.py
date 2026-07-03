import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier

# =====================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="Dashboard AI - Churn BPJS TK",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 2. SISTEM CACHING DATA & MODEL (Optimasi Kecepatan)
# =====================================================================
@st.cache_data
def load_data():
    """Memuat data bersih hasil ekspor dari train_model.py"""
    df = pd.read_excel("df_final.xlsx")

    # FIX #1: Dataset tidak memiliki kolom 'ID'.
    # Buat ID sintetis berbasis index agar fitur identifikasi peserta tetap berjalan.
    if 'ID' not in df.columns:
        df.insert(0, 'ID', [f"BPJSTK-{i+1:05d}" for i in range(len(df))])
    return df

@st.cache_resource
def load_model():
    """Memuat model CatBoost yang sudah dilatih (format .cbm)"""
    model = CatBoostClassifier()
    model.load_model("model_churn_bpjs.cbm")
    return model

# Eksekusi muat data
df_final = load_data()
model = load_model()

# =====================================================================
# 3. PROSES INFERENSI (PREDIKSI) REAL-TIME
# =====================================================================
# FIX #2: Ambil urutan & nama fitur LANGSUNG dari model agar selalu sinkron
# dengan saat training. Ini mencegah error jika urutan kolom dataset berubah.
expected_features = model.feature_names_

df_infer = df_final.copy()
# 'ID' dijadikan index supaya bukan ikut sebagai fitur prediksi
df_infer = df_infer.set_index('ID')

# Pilih hanya fitur yang diharapkan model (ini juga otomatis membuang 'Target_Churn')
X_infer = df_infer[expected_features].copy()

# FIX #3: Ambil daftar kolom kategorikal LANGSUNG dari model (bukan hard-code).
# CatBoost menyimpan indeks fitur kategorikal di get_cat_feature_indices().
cat_feature_indices = model.get_cat_feature_indices()
cat_features = [expected_features[i] for i in cat_feature_indices]

# Bersihkan NaN pada kolom kategorikal & paksa ke string
# (kolom Kanal_bayar punya 227 NaN di dataset, ini wajib di-handle)
for col in cat_features:
    X_infer[col] = X_infer[col].fillna('Unknown').astype(str)

# Dapatkan probabilitas churn (kolom indeks ke-1 untuk probabilitas kelas 1)
probabilitas = model.predict_proba(X_infer)[:, 1]

# Gabungkan hasil probabilitas kembali ke dataframe utama untuk visualisasi
df_display = df_final.copy()
df_display['Probabilitas_Churn'] = probabilitas

# =====================================================================
# 4. KENDALI OPERASIONAL (SIDEBAR)
# =====================================================================
st.sidebar.title("⚙️ Kendali Operasional")
st.sidebar.markdown("Sesuaikan sensitivitas peringatan AI terhadap risiko peserta.")

# Slider dinamis untuk mengatur ambang batas probabilitas
threshold = st.sidebar.slider(
    "Ambang Batas Probabilitas (Threshold)",
    min_value=0.0, max_value=1.0, value=0.50, step=0.05,
    help="Semakin rendah angka ini, AI akan semakin agresif mengeluarkan peringatan Churn."
)

st.sidebar.divider()
st.sidebar.info("💡 **Mode Advisor MIT:** Intervensi data bersifat real-time berdasarkan threshold.")

# Tentukan label Prediksi berdasarkan Threshold dinamis dari sidebar
df_display['Prediksi_Status'] = np.where(
    df_display['Probabilitas_Churn'] >= threshold,
    '🔴 Akan Churn',
    '🟢 Aman (Aktif)'
)

# =====================================================================
# 5. ANTARMUKA TAB UTAMA
# =====================================================================
st.title("🛡️ Dashboard Prediksi Churn BPJS Ketenagakerjaan")
tab_ai, tab_eda = st.tabs(["🚀 Operational AI (Intervensi CS)", "📊 Strategic Insights (EDA Makro)"])

# ---------------------------------------------------------------------
# TAB 1: OPERATIONAL AI (MATRIX & DEEP DIVE)
# ---------------------------------------------------------------------
with tab_ai:
    st.header("Matriks Risiko Churn Peserta")

    # Perhitungan KPI Dinamis
    total_peserta = len(df_display)
    df_churn = df_display[df_display['Prediksi_Status'] == '🔴 Akan Churn']
    jumlah_churn = len(df_churn)
    churn_rate = (jumlah_churn / total_peserta) * 100 if total_peserta > 0 else 0

    # Menampilkan Tiga Kartu KPI
    kpi1, kpi2= st.columns(2)
    kpi1.metric("Total Peserta Dianalisis", f"{total_peserta:,}")
    kpi2.metric("Prediksi Churn (Risiko)", f"{jumlah_churn:,} ({churn_rate:.1f}%)", delta_color="inverse")

    st.divider()

    # Tabel Matriks Risiko (Diurutkan dari risiko tertinggi)
    st.subheader("📋 Daftar Prioritas Intervensi Cabang")

    # FIX #4: Karena df_display sudah punya kolom 'ID' sejak load_data(),
    # tidak perlu reset_index lagi. Cukup sort biasa.
    df_tabel = df_display.sort_values(by='Probabilitas_Churn', ascending=False)

    kolom_tampil = ['ID', 'Prediksi_Status', 'Probabilitas_Churn', 'Total_Produk_Aktif',
                    'Friksi_Pembayaran', 'Segmen_Generasi', 'Kota_kabupaten']

    st.dataframe(
        df_tabel[kolom_tampil].style.format({'Probabilitas_Churn': "{:.1%}"})
                            .background_gradient(subset=['Probabilitas_Churn'], cmap='Reds'),
        use_container_width=True,
        height=300
    )

    st.divider()

    # Fitur Penjelasan AI (Deep Dive)
    st.subheader("🔍 AI Deep Dive (Analisis Individu)")

    # FIX #5: Urutkan list peserta berisiko berdasarkan probabilitas tertinggi
    # agar staf CS bisa langsung memprioritaskan kasus paling kritikal.
    df_churn_sorted = df_churn.sort_values(by='Probabilitas_Churn', ascending=False)
    list_risiko = df_churn_sorted['ID'].head(100).tolist()

    if len(list_risiko) > 0:
        # ===== ENHANCEMENT: Search Bar & Filter Peserta =====
        search_col, filter_col = st.columns([2, 1])

        with search_col:
            # Search Bar: cari ID peserta dengan auto-complete
            search_input = st.text_input(
                "🔍 Cari ID Peserta",
                placeholder="Ketik ID (misal: BPJSTK-02438) atau filter manual",
                help="Cari berdasarkan ID peserta untuk akses cepat ke laporan diagnosis"
            )

        with filter_col:
            # Quick Access: tombol untuk top 5 risiko tertinggi
            if st.button("⚡ Top 5 Risiko", help="Tampilkan 5 peserta dengan risiko churn tertinggi"):
                search_input = list_risiko[0]  # Ambil top 1

        # ===== Logic pencarian: support auto-complete & partial match =====
        if search_input:
            # Filter: cari ID yang cocok (case-insensitive, partial match)
            filtered_list = [
                id_val for id_val in list_risiko
                if search_input.upper() in id_val.upper()
            ]

            if filtered_list:
                # Jika hasil pencarian hanya 1, auto-select. Jika lebih, tampilkan dropdown
                if len(filtered_list) == 1:
                    pilih_id = filtered_list[0]
                else:
                    pilih_id = st.selectbox(
                        f"📌 Hasil pencarian ({len(filtered_list)} ID ditemukan):",
                        filtered_list,
                        format_func=lambda x: f"{x} - {df_display[df_display['ID']==x]['Probabilitas_Churn'].iloc[0]*100:.1f}% risk"
                    )
            else:
                st.warning(f"❌ Tidak ada peserta berisiko dengan ID mengandung '{search_input}'")
                pilih_id = None
        else:
            # Jika tidak ada pencarian, tampilkan selectbox normal dengan format lebih informatif
            pilih_id = st.selectbox(
                "📋 Atau pilih dari daftar 100 peserta berisiko tertinggi:",
                list_risiko,
                format_func=lambda x: f"{x} - {df_display[df_display['ID']==x]['Probabilitas_Churn'].iloc[0]*100:.1f}% risk"
            )

        # ===== Tampilkan laporan diagnosis untuk ID yang dipilih =====
        if pilih_id:
            data_individu = df_tabel[df_tabel['ID'] == pilih_id].iloc[0]
            prob = data_individu['Probabilitas_Churn'] * 100

            # Styling berdasarkan risk level (warna-kode otomatis)
            if prob >= 80:
                box_type = "error"  # Merah: kritikal
                risk_label = "🔴 KRITICAL (Risiko Sangat Tinggi)"
            elif prob >= 60:
                box_type = "warning"  # Oranye: tinggi
                risk_label = "🟠 HIGH (Risiko Tinggi)"
            else:
                box_type = "info"  # Biru: medium
                risk_label = "🟡 MEDIUM (Risiko Sedang)"

            if box_type == "error":
                st.error(
                    f"**{risk_label} — {pilih_id}**\n\n"
                    f"Probabilitas Churn: **{prob:.1f}%** (Status: {data_individu['Prediksi_Status']})\n\n"
                    f"**Faktor Pendorong:** Peserta berusia {data_individu['Usia']} tahun, termasuk segmen **{data_individu['Segmen_Generasi']}** "
                    f"di wilayah **{data_individu['Kota_kabupaten']}**. Tingkat risiko ini diperkuat oleh penggunaan kanal pembayaran "
                    f"**{data_individu['Friksi_Pembayaran']}** ({data_individu['Kanal_bayar']}). Saat ini peserta terdaftar pada "
                    f"**{data_individu['Total_Produk_Aktif']} program jaminan** (JHT: {data_individu['JHT']}, JKK: {data_individu['JKK']}, JKM: {data_individu['JKM']}).\n\n"
                    f"**🚨 Rekomendasi URGENT:** Alokasikan staf representatif HARI INI untuk menghubungi peserta secara prioritas dan tawarkan "
                    f"migrasi ke kanal autodebet digital atau program retensi khusus."
                )
            elif box_type == "warning":
                st.warning(
                    f"**{risk_label} — {pilih_id}**\n\n"
                    f"Probabilitas Churn: **{prob:.1f}%** (Status: {data_individu['Prediksi_Status']})\n\n"
                    f"**Faktor Pendorong:** Peserta berusia {data_individu['Usia']} tahun, termasuk segmen **{data_individu['Segmen_Generasi']}** "
                    f"di wilayah **{data_individu['Kota_kabupaten']}**. Tingkat risiko ini diperkuat oleh penggunaan kanal pembayaran "
                    f"**{data_individu['Friksi_Pembayaran']}** ({data_individu['Kanal_bayar']}). Saat ini peserta terdaftar pada "
                    f"**{data_individu['Total_Produk_Aktif']} program jaminan** (JHT: {data_individu['JHT']}, JKK: {data_individu['JKK']}, JKM: {data_individu['JKM']}).\n\n"
                    f"**📞 Rekomendasi:** Hubungi peserta dalam 3-5 hari kerja untuk follow-up dan tawarkan migrasi ke kanal autodebet digital."
                )
            else:
                st.info(
                    f"**{risk_label} — {pilih_id}**\n\n"
                    f"Probabilitas Churn: **{prob:.1f}%** (Status: {data_individu['Prediksi_Status']})\n\n"
                    f"**Faktor Pendorong:** Peserta berusia {data_individu['Usia']} tahun, termasuk segmen **{data_individu['Segmen_Generasi']}** "
                    f"di wilayah **{data_individu['Kota_kabupaten']}**. Penggunaan kanal pembayaran **{data_individu['Friksi_Pembayaran']}** "
                    f"({data_individu['Kanal_bayar']}). Peserta terdaftar pada **{data_individu['Total_Produk_Aktif']} program jaminan** "
                    f"(JHT: {data_individu['JHT']}, JKK: {data_individu['JKK']}, JKM: {data_individu['JKM']}).\n\n"
                    f"**💡 Rekomendasi:** Monitor peserta dan pertahankan engagement melalui komunikasi berkala."
                )

            # Tambahan: Perbandingan dengan rata-rata segmen
            st.divider()
            avg_prob_segmen = df_display[df_display['Segmen_Generasi'] == data_individu['Segmen_Generasi']]['Probabilitas_Churn'].mean()
            avg_prob_kota = df_display[df_display['Kota_kabupaten'] == data_individu['Kota_kabupaten']]['Probabilitas_Churn'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Probabilitas vs Rata-rata Segmen",
                    f"{prob:.1f}%",
                    delta=f"{(prob - avg_prob_segmen*100):.1f}pp",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Probabilitas vs Rata-rata Kota",
                    f"{prob:.1f}%",
                    delta=f"{(prob - avg_prob_kota*100):.1f}pp",
                    delta_color="inverse"
                )

    else:
        # FIX #6: Perbaiki typo "Terdikte" -> "Terdeteksi"
        st.success("Terdeteksi 0 peserta berisiko tinggi berdasarkan ambang batas saat ini. Coba turunkan threshold di sidebar.")

# ---------------------------------------------------------------------
# TAB 2: STRATEGIC INSIGHTS (EDA MAKRO)
# ---------------------------------------------------------------------
with tab_eda:
    st.header("Eksplorasi Pola Bisnis Makro")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Keterikatan Produk (Switching Cost)")
        # Hitung rasio churn vs produk
        df_produk = df_final.groupby('Total_Produk_Aktif')['Target_Churn'].value_counts(normalize=True).unstack().reset_index()
        # Pengaman jika tidak ada kelas tertentu
        for c in [0, 1]:
            if c not in df_produk.columns:
                df_produk[c] = 0.0

        df_produk.columns = ['Total Produk', 'Aktif', 'Churn']
        df_produk['Churn (%)'] = df_produk['Churn'] * 100

        fig_produk = px.bar(df_produk, x='Total Produk', y='Churn (%)', text='Churn (%)',
                            color_discrete_sequence=['#FF4B4B'])
        fig_produk.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_produk.update_layout(yaxis_title="Persentase Churn")
        st.plotly_chart(fig_produk, use_container_width=True)

    with col2:
        st.subheader("Siklus Waktu Rentan (Tenor)")
        # Hitung rata-rata churn berdasarkan tenor
        df_tenor = df_final.groupby('Jendela_Risiko_Tenor', observed=False)['Target_Churn'].mean().reset_index()
        df_tenor['Churn Rate (%)'] = df_tenor['Target_Churn'] * 100

        # FIX #7: Urutkan tenor secara logis (bukan alfabetis) agar plot garis
        # tampak menceritakan progresi waktu (New -> Mid -> Loyal, dst.)
        urutan_tenor = ['New_Member', 'Early_Stage', 'Mid_Term', 'Long_Term', 'Loyal']
        urutan_ada = [t for t in urutan_tenor if t in df_tenor['Jendela_Risiko_Tenor'].values]
        if urutan_ada:
            df_tenor['Jendela_Risiko_Tenor'] = pd.Categorical(
                df_tenor['Jendela_Risiko_Tenor'], categories=urutan_ada, ordered=True
            )
            df_tenor = df_tenor.sort_values('Jendela_Risiko_Tenor')

        # Plot garis
        fig_tenor = px.line(df_tenor, x='Jendela_Risiko_Tenor', y='Churn Rate (%)',
                            markers=True, color_discrete_sequence=['#0068C9'])
        fig_tenor.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_tenor, use_container_width=True)

    st.divider()

    st.subheader("Peta Gesekan Kanal Pembayaran (Heatmap)")
    # Hitung rata-rata target churn per kota dan kanal pembayaran
    pivot_kanal_kota = pd.pivot_table(df_final, values='Target_Churn',
                                      index='Kota_kabupaten', columns='Friksi_Pembayaran',
                                      aggfunc='mean').reset_index()

    # Ambil 15 Kota Terbanyak
    top_kota = df_final['Kota_kabupaten'].value_counts().head(15).index
    pivot_top = pivot_kanal_kota[pivot_kanal_kota['Kota_kabupaten'].isin(top_kota)].set_index('Kota_kabupaten')

    fig_heat = px.imshow(
        pivot_top,
        text_auto=".1%",
        aspect="auto",
        color_continuous_scale='Reds',
        labels=dict(color="Rasio Churn", x="Jenis Kanal Bayar", y="Kota/Kabupaten")
    )
    st.plotly_chart(fig_heat, use_container_width=True)
