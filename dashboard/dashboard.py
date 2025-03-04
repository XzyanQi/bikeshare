import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Judul & Deskripsi
st.title("🚲 Bike Sharing Analysis Dashboard")
st.markdown("""
### Proyek Analisis Data: Bike Sharing Dataset
**Oleh:** Faqih Muhammad Ihsan
            
Dashboard ini menyajikan analisis terhadap pola penggunaan sepeda selama periode 2011-2012,
dengan fokus pada pengaruh musim, cuaca, dan pola waktu terhadap jumlah peminjaman sepeda.
""")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    day_df = pd.read_csv('data/day.csv')
    hour_df = pd.read_csv('data/hour.csv')
    
    # Konversi Tanggal
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    # Pemetaan Musim, Cuaca, dan Hari
    season_mapping = {1: 'Semi', 2: 'Panas', 3: 'Gugur', 4: 'Dingin'}
    weathersit_mapping = {1: 'Cerah', 2: 'Berkabut', 3: 'Salju/Hujan Ringan', 4: 'Hujan Lebat'}
    weekday_mapping = {0: 'Minggu', 1: 'Senin', 2: 'Selasa', 3: 'Rabu', 4: 'Kamis', 5: 'Jumat', 6: 'Sabtu'}
    
    # Menerapkan pemetaan
    day_df['season_label'] = day_df['season'].map(season_mapping)
    day_df['weathersit_label'] = day_df['weathersit'].map(weathersit_mapping)
    day_df['weekday_label'] = day_df['weekday'].map(weekday_mapping)
    
    hour_df['season_label'] = hour_df['season'].map(season_mapping)
    hour_df['weathersit_label'] = hour_df['weathersit'].map(weathersit_mapping)
    hour_df['weekday_label'] = hour_df['weekday'].map(weekday_mapping)
    
    # Membuat Kolom Label
    day_df['holiday_label'] = day_df['holiday'].apply(lambda x: 'Hari Libur' if x == 1 else 'Hari Kerja')
    hour_df['holiday_label'] = hour_df['holiday'].apply(lambda x: 'Hari Libur' if x == 1 else 'Hari Kerja')
    
    day_df['workingday_label'] = day_df['workingday'].apply(lambda x: 'Akhir Pekan/Libur' if x == 0 else 'Hari Kerja')
    hour_df['workingday_label'] = hour_df['workingday'].apply(lambda x: 'Akhir Pekan/Libur' if x == 0 else 'Hari Kerja')
    
    # Membuat Kolom Tauhn
    day_df['month'] = day_df['dteday'].dt.month
    day_df['year'] = day_df['dteday'].dt.year
    hour_df['month'] = hour_df['dteday'].dt.month
    hour_df['year'] = hour_df['dteday'].dt.year
    
    # Menerapkan Pengelompokkan
    features = day_df[['temp', 'hum', 'windspeed', 'cnt']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    day_df['cluster'] = clusters
    cluster_mapping = {
        0: 'Penggunaan Rendah',
        1: 'Penggunaan Sedang',
        2: 'Penggunaan Tinggi'
    }
    day_df['usage_level'] = day_df['cluster'].map(cluster_mapping)
    
    return day_df, hour_df

# Memuat Data
try:
    day_df, hour_df = load_data()
    st.success("Data berhasil dimuat!")
except Exception as e:
    st.error(f"Error saat memuat data: {e}")
    st.warning("Pastikan file day.csv dan hour.csv berada di direktori yang sama dengan aplikasi ini.")
    st.stop()

# Membuat Tombol Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["📊 Overview", 
     "🌦️ Pengaruh Musim & Cuaca", 
     "⏰ Pola Waktu Penggunaan",
     "🔍 Analisis Lanjutan"]
)

# Overview page
if page == "📊 Overview":
    st.header("📊 Overview Data Bike Sharing")
    
    # Menampilkan Ringkasan Data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(day_df[['casual', 'registered', 'cnt']].describe())
    
    with col2:
        st.subheader("Korelasi Antar Variabel")
        corr = day_df[['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tren Bulanan
    st.subheader("Tren Peminjaman Sepeda Bulanan (2011-2012)")
    monthly_data = day_df.groupby(['year', 'month'])['cnt'].sum().reset_index()
    monthly_data['date'] = monthly_data.apply(lambda x: datetime(int(x['year']), int(x['month']), 1), axis=1)
    monthly_data = monthly_data.sort_values('date')
    
    fig = px.line(monthly_data, x='date', y='cnt', markers=True)
    fig.update_layout(
        xaxis_title="Bulan",
        yaxis_title="Jumlah Peminjaman",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribusi Rental
    st.subheader("Distribusi Jumlah Peminjaman Sepeda Harian")
    fig = px.histogram(day_df, x='cnt', nbins=30, marginal='box')
    fig.update_layout(
        xaxis_title="Jumlah Peminjaman",
        yaxis_title="Frekuensi",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Musim dan Cuaca
elif page == "🌦️ Pengaruh Musim & Cuaca":
    st.header("🌦️ Pengaruh Musim & Cuaca terhadap Peminjaman Sepeda")
    
    # Analysis dari musim
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Peminjaman Berdasarkan Musim")
        season_avg = day_df.groupby('season_label')['cnt'].mean().reset_index()
        fig = px.bar(season_avg, x='season_label', y='cnt', color='season_label',
                    labels={'cnt': 'Rata-rata Peminjaman', 'season_label': 'Musim'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Peminjaman Berdasarkan Kondisi Cuaca")
        weather_avg = day_df.groupby('weathersit_label')['cnt'].mean().reset_index()
        fig = px.bar(weather_avg, x='weathersit_label', y='cnt', color='weathersit_label',
                    labels={'cnt': 'Rata-rata Peminjaman', 'weathersit_label': 'Kondisi Cuaca'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Peminjaman per Musim")
        fig = px.box(day_df, x='season_label', y='cnt',
                    labels={'cnt': 'Jumlah Peminjaman', 'season_label': 'Musim'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribusi Peminjaman per Kondisi Cuaca")
        fig = px.box(day_df, x='weathersit_label', y='cnt',
                    labels={'cnt': 'Jumlah Peminjaman', 'weathersit_label': 'Kondisi Cuaca'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Hubungan suhu dan jjumlah
    st.subheader("Hubungan antara Suhu dan Jumlah Peminjaman Sepeda")
    fig = px.scatter(day_df, x='temp', y='cnt', color='season_label', 
                   size='cnt', opacity=0.7,
                   labels={'temp': 'Suhu (Normalisasi)', 'cnt': 'Jumlah Peminjaman', 'season_label': 'Musim'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Evek Kelembapan dan kecepatan angin
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pengaruh Kelembaban terhadap Peminjaman")
        fig = px.scatter(day_df, x='hum', y='cnt', color='season_label', 
                       opacity=0.7, trendline='ols',
                       labels={'hum': 'Kelembaban (Normalisasi)', 'cnt': 'Jumlah Peminjaman', 'season_label': 'Musim'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pengaruh Kecepatan Angin terhadap Peminjaman")
        fig = px.scatter(day_df, x='windspeed', y='cnt', color='season_label', 
                       opacity=0.7, trendline='ols',
                       labels={'windspeed': 'Kecepatan Angin (Normalisasi)', 'cnt': 'Jumlah Peminjaman', 'season_label': 'Musim'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Pola Waktu
elif page == "⏰ Pola Waktu Penggunaan":
    st.header("⏰ Pola Waktu Penggunaan Sepeda")
    
    # Jam
    st.subheader("Pola Penggunaan Berdasarkan Jam")
    hourly_pattern = hour_df.groupby('hr')['cnt'].mean().reset_index()
    fig = px.line(hourly_pattern, x='hr', y='cnt', markers=True,
                labels={'hr': 'Jam', 'cnt': 'Rata-rata Peminjaman'})
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Hari kerja & Akhir Pekan
    st.subheader("Perbandingan Pola Jam: Hari Kerja vs Akhir Pekan/Libur")
    hourly_weekday = hour_df.groupby(['hr', 'workingday_label'])['cnt'].mean().reset_index()
    fig = px.line(hourly_weekday, x='hr', y='cnt', color='workingday_label', markers=True,
                labels={'hr': 'Jam', 'cnt': 'Rata-rata Peminjaman', 'workingday_label': 'Jenis Hari'})
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Dalam Seminggu
    st.subheader("Pola Penggunaan Berdasarkan Hari dalam Seminggu")
    weekday_order = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    weekday_pattern = day_df.groupby('weekday_label')['cnt'].mean().reset_index()
    weekday_pattern['weekday_label'] = pd.Categorical(weekday_pattern['weekday_label'], categories=weekday_order, ordered=True)
    weekday_pattern = weekday_pattern.sort_values('weekday_label')
    
    fig = px.bar(weekday_pattern, x='weekday_label', y='cnt', color='weekday_label',
               labels={'cnt': 'Rata-rata Peminjaman', 'weekday_label': 'Hari'})
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bandingkan dengan Hari Libur
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Perbandingan: Hari Libur vs Hari Kerja")
        fig = px.box(day_df, x='holiday_label', y='cnt', color='holiday_label',
                   labels={'cnt': 'Jumlah Peminjaman', 'holiday_label': 'Jenis Hari'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Rata-rata Peminjaman: Hari Libur vs Hari Kerja")
        holiday_avg = day_df.groupby('holiday_label')['cnt'].mean().reset_index()
        fig = px.bar(holiday_avg, x='holiday_label', y='cnt', color='holiday_label',
                   labels={'cnt': 'Rata-rata Peminjaman', 'holiday_label': 'Jenis Hari'})
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Perbandingan Tipe Pengguna
    st.subheader("Perbandingan Pengguna Casual vs Registered Berdasarkan Hari")
    user_weekday = day_df.groupby('weekday_label')[['casual', 'registered']].mean().reset_index()
    user_weekday['weekday_label'] = pd.Categorical(user_weekday['weekday_label'], categories=weekday_order, ordered=True)
    user_weekday = user_weekday.sort_values('weekday_label')
    
    user_weekday_melted = pd.melt(user_weekday, id_vars=['weekday_label'], value_vars=['casual', 'registered'],
                                  var_name='user_type', value_name='count')
    
    fig = px.bar(user_weekday_melted, x='weekday_label', y='count', color='user_type', barmode='group',
                labels={'count': 'Rata-rata Peminjaman', 'weekday_label': 'Hari', 'user_type': 'Tipe Pengguna'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Analiis Lanjutan
elif page == "🔍 Analisis Lanjutan":
    st.header("🔍 Analisis Lanjutan")
    
    # Hasil Pengelompokkan
    st.subheader("Hasil Clustering Penggunaan Sepeda")
    
    # Karakteristik
    cluster_analysis = day_df.groupby('usage_level')[['temp', 'hum', 'windspeed', 'cnt', 'casual', 'registered']].mean()
    st.dataframe(cluster_analysis)
    
    # Visualisasi
    st.subheader("Visualisasi Cluster Berdasarkan Suhu dan Jumlah Peminjaman")
    fig = px.scatter(day_df, x='temp', y='cnt', color='usage_level',
                   labels={'temp': 'Suhu (Normalisasi)', 'cnt': 'Jumlah Peminjaman', 'usage_level': 'Level Penggunaan'})
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Musim
    st.subheader("Distribusi Level Penggunaan Berdasarkan Musim")
    cluster_season = pd.crosstab(day_df['season_label'], day_df['usage_level'])
    cluster_season_melted = pd.melt(cluster_season.reset_index(), id_vars=['season_label'], 
                                   value_vars=cluster_season.columns, 
                                   var_name='usage_level', value_name='count')
    
    fig = px.bar(cluster_season_melted, x='season_label', y='count', color='usage_level', barmode='stack',
               labels={'count': 'Jumlah Hari', 'season_label': 'Musim', 'usage_level': 'Level Penggunaan'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Waktu
    st.subheader("Pola Temporal Cluster Penggunaan")
    cluster_time = day_df.groupby(['year', 'month', 'usage_level']).size().reset_index(name='count')
    cluster_time['yearmonth'] = cluster_time['year'].astype(str) + '-' + cluster_time['month'].astype(str).str.zfill(2)
    
    fig = px.line(cluster_time, x='yearmonth', y='count', color='usage_level',
                 labels={'count': 'Jumlah Hari', 'yearmonth': 'Bulan-Tahun', 'usage_level': 'Level Penggunaan'})
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar
    st.subheader("Karakteristik Cluster (Diagram Radar)")
    
    # Baguab Radar
    radar_df = cluster_analysis.copy()
    for col in radar_df.columns:
        radar_df[col] = (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
    
    categories = ['Suhu', 'Kelembaban', 'Kecepatan Angin', 'Total Peminjaman', 'Pengguna Casual', 'Pengguna Registered']
    
    fig = go.Figure()
    
    for idx, level in enumerate(radar_df.index):
        fig.add_trace(go.Scatterpolar(
            r=radar_df.iloc[idx].values,
            theta=categories,
            fill='toself',
            name=level
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard dibuat oleh Faqih Muhammad Ihsan | © 2025")