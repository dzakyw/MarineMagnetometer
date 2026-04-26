import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline
from scipy.stats import median_abs_deviation
from datetime import datetime
import geomag  # pip install geomag
from math import radians, sin, cos, sqrt, atan2

# ================== 1. UTILITY FUNCTIONS ==================

def clean_numeric_columns(df, columns):
    """Mengganti nilai non-numerik (seperti '*') dengan NaN pada kolom yang ditentukan"""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_data(uploaded_file):
    """Load file xlsx atau csv, bersihkan kolom numerik dari '*'"""
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Kolom yang mungkin mengandung '*' (semua numerik kecuali Tbase yang berisi waktu)
    numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 
                    'Altitude', 'Depth', 'Fbase']
    df = clean_numeric_columns(df, numeric_cols)
    return df

def parse_datetime(df):
    """Gabungkan Reading_Date dan Reading_Time menjadi kolom datetime"""
    try:
        df['datetime'] = pd.to_datetime(df['Reading_Date'].astype(str) + ' ' + df['Reading_Time'].astype(str))
    except Exception as e:
        raise ValueError(f"Tidak dapat menggabungkan Reading_Date dan Reading_Time: {e}")
    return df

def separate_base_and_survey(df):
    """Pisahkan baris data base (Tbase dan Fbase tidak null) dan data survei (Field tidak null)"""
    survey_df = df[df['Field'].notna()].copy()
    base_df = df[df['Tbase'].notna() & df['Fbase'].notna()].copy()
    if not base_df.empty:
        # Tbase bisa berupa string datetime, konversi ke datetime
        base_df['base_datetime'] = pd.to_datetime(base_df['Tbase'])
    return survey_df, base_df

def hampel_filter(series, window_size=5, n_sigmas=3.0):
    """Hampel filter untuk deteksi outlier, mengganti outlier dengan NaN"""
    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()
    mad = series.rolling(window=window_size, center=True, min_periods=1).apply(
        lambda x: median_abs_deviation(x, nan_policy='omit'), raw=True
    )
    mad = mad.fillna(mad.median())
    deviation = np.abs(series - rolling_median)
    outlier_mask = deviation > (n_sigmas * mad)
    cleaned = series.copy()
    cleaned[outlier_mask] = np.nan
    return cleaned, outlier_mask

def interpolate_nan(series, method='cubic'):
    """Interpolasi nilai NaN dengan spline cubic atau linear"""
    idx = series.index
    valid = ~np.isnan(series.values)
    if method == 'cubic' and np.sum(valid) > 3:
        cs = CubicSpline(idx[valid], series.values[valid])
        interpolated = cs(idx)
    else:
        interpolated = series.interpolate(method='linear', limit_direction='both')
    return pd.Series(interpolated, index=idx)

def moving_average(series, window=5):
    return series.rolling(window=window, center=True, min_periods=1).mean()

def butterworth_filter(series, cutoff=0.1, fs=1.0, order=4, btype='low'):
    """Butterworth lowpass filter"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    # Isi NaN sementara dengan median untuk menghindari error pada filtfilt
    if series.isna().any():
        series = series.fillna(series.median())
    return filtfilt(b, a, series)

def apply_filter(series, method, **params):
    """Apply filter berdasarkan pilihan user"""
    if method == 'Hampel (despiking)':
        cleaned, _ = hampel_filter(series, window_size=params.get('window', 5), n_sigmas=params.get('threshold', 3.0))
        result = interpolate_nan(cleaned, method='cubic')
    elif method == 'Moving Average':
        result = moving_average(series, window=params.get('window', 5))
    elif method == 'Savitzky-Golay':
        window = params.get('window', 11)
        if window % 2 == 0:
            window += 1
        # Isi NaN sementara dengan interpolasi lokal
        temp = series.interpolate(method='linear', limit_direction='both')
        result = savgol_filter(temp, window_length=window, polyorder=3)
        result = pd.Series(result, index=series.index)
    elif method == 'Butterworth Lowpass':
        result = butterworth_filter(series, cutoff=params.get('cutoff', 0.1), fs=1.0, order=4)
    else:
        result = series.copy()
    return result

def compute_igrf(lat, lon, alt_m, datetime_obj):
    """Hitung IGRF total field (F) pada titik survei"""
    alt_km = alt_m / 1000.0
    year = datetime_obj.year
    day_of_year = datetime_obj.timetuple().tm_yday
    year_decimal = year + (day_of_year - 1) / 365.25
    try:
        _, _, _, F, _, _ = geomag.mag_field(lat, lon, alt_km, year_decimal)
    except:
        F = np.nan
    return F

def compute_diurnal_correction(survey_df, base_df, reference_method='first'):
    """Koreksi diurnal dengan interpolasi linear Fbase"""
    base_df = base_df.sort_values('base_datetime')
    base_times = base_df['base_datetime'].values
    base_values = base_df['Fbase'].values
    
    survey_times = survey_df['datetime'].values
    interpolated = np.interp(
        [t.timestamp() for t in survey_times],
        [t.timestamp() for t in base_times],
        base_values
    )
    
    if reference_method == 'first':
        ref_value = base_values[0]
    elif reference_method == 'mean':
        ref_value = np.mean(base_values)
    else:
        ref_value = 0.0
    return interpolated - ref_value

def compute_distance_along_line(df):
    """Hitung jarak kumulatif (meter) menggunakan rumus Haversine"""
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371000
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lon2 - lon1)
        a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    distances = [0.0]
    for i in range(1, len(df)):
        d = haversine(df.iloc[i-1]['Longitude'], df.iloc[i-1]['Latitude'],
                      df.iloc[i]['Longitude'], df.iloc[i]['Latitude'])
        distances.append(distances[-1] + d)
    return np.array(distances)

# ================== 2. MAIN STREAMLIT APP ==================

st.set_page_config(page_title="Pengolahan Data Magnetik Kelautan", layout="wide")
st.title("🌊 Pengolahan Data Magnetik Kelautan + Peta Anomali")

uploaded_file = st.sidebar.file_uploader("📂 Upload file Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Load data
    df_raw = load_data(uploaded_file)
    st.subheader("📋 Data Awal (10 baris pertama)")
    st.dataframe(df_raw.head(10))
    
    # Parse datetime
    try:
        df_raw = parse_datetime(df_raw)
    except Exception as e:
        st.error(f"❌ Gagal parse datetime: {e}")
        st.stop()
    
    # Pisahkan data survei dan base
    survey_df, base_df = separate_base_and_survey(df_raw)
    if survey_df.empty:
        st.error("❌ Tidak ada data survei (kolom Field kosong)")
        st.stop()
    if base_df.empty:
        st.warning("⚠️ Tidak ada data base (Tbase/Fbase kosong). Koreksi diurnal tidak dilakukan.")
    
    # Sidebar: Parameter filter
    st.sidebar.header("🔧 Parameter Filtering")
    
    # Filter Field
    field_method = st.sidebar.selectbox(
        "Filter Field", 
        ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay", "Butterworth Lowpass"]
    )
    field_params = {}
    if field_method == "Hampel (despiking)":
        field_params['window'] = st.sidebar.slider("Window Hampel", 3, 21, 5, 2)
        field_params['threshold'] = st.sidebar.slider("Threshold sigma", 1.0, 5.0, 3.0, 0.5)
    elif field_method in ["Moving Average", "Savitzky-Golay"]:
        field_params['window'] = st.sidebar.slider("Window size", 3, 51, 11, 2)
    elif field_method == "Butterworth Lowpass":
        field_params['cutoff'] = st.sidebar.slider("Cutoff frequency (0-0.5)", 0.01, 0.5, 0.1, 0.01)
    
    # Filter Altitude
    alt_method = st.sidebar.selectbox(
        "Filter Altitude", 
        ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay"]
    )
    alt_params = {}
    if alt_method == "Hampel (despiking)":
        alt_params['window'] = st.sidebar.slider("Window Alt Hampel", 3, 21, 5, 2)
        alt_params['threshold'] = st.sidebar.slider("Threshold sigma Alt", 1.0, 5.0, 3.0, 0.5)
    elif alt_method in ["Moving Average", "Savitzky-Golay"]:
        alt_params['window'] = st.sidebar.slider("Window size Alt", 3, 51, 11, 2)
    
    # Tombol proses
    if st.button("🚀 Proses Filter & Koreksi"):
        with st.spinner("Memproses data..."):
            # Filter Field
            if field_method != "None":
                survey_df['Field_filtered'] = apply_filter(survey_df['Field'], field_method, **field_params)
            else:
                survey_df['Field_filtered'] = survey_df['Field']
            
            # Filter Altitude
            if alt_method != "None":
                survey_df['Altitude_filtered'] = apply_filter(survey_df['Altitude'], alt_method, **alt_params)
            else:
                survey_df['Altitude_filtered'] = survey_df['Altitude']
            
            # Koreksi diurnal
            if not base_df.empty:
                ref_method = st.selectbox("Metode referensi base", ['first', 'mean', 'none'], key='ref')
                diurnal_corr = compute_diurnal_correction(survey_df, base_df, 
                                                         reference_method=ref_method if ref_method != 'none' else 'constant')
                survey_df['Diurnal_Correction'] = diurnal_corr
            else:
                survey_df['Diurnal_Correction'] = 0.0
            
            # Hitung IGRF (dengan progress bar)
            progress_bar = st.progress(0)
            igrf_vals = []
            for i, row in survey_df.iterrows():
                igrf = compute_igrf(row['Latitude'], row['Longitude'], row['Altitude_filtered'], row['datetime'])
                igrf_vals.append(igrf)
                progress_bar.progress((i+1)/len(survey_df))
            survey_df['IGRF'] = igrf_vals
            
            # TMI = Field_filtered - IGRF - Diurnal_Correction
            survey_df['TMI'] = survey_df['Field_filtered'] - survey_df['IGRF'] - survey_df['Diurnal_Correction']
            
            st.session_state['survey_df'] = survey_df
            st.success("✅ Proses selesai!")
    
    # Jika data sudah diproses
    if 'survey_df' in st.session_state:
        survey_df = st.session_state['survey_df']
        
        # Tampilkan hasil numerik
        st.subheader("📊 Hasil setelah koreksi (10 baris pertama)")
        st.dataframe(survey_df[['datetime', 'Field_filtered', 'IGRF', 'Diurnal_Correction', 'TMI']].head(10))
        
        # ========== PLOT PETA DAN LINTASAN ==========
        st.header("🗺️ Visualisasi Peta dan Lintasan Anomali")
        
        # Pilih line name
        if 'Line_Name' in survey_df.columns:
            available_lines = survey_df['Line_Name'].dropna().unique()
            if len(available_lines) > 0:
                selected_lines = st.multiselect("Pilih Line Name untuk ditampilkan", available_lines, default=available_lines)
                plot_df = survey_df[survey_df['Line_Name'].isin(selected_lines)]
            else:
                plot_df = survey_df
        else:
            plot_df = survey_df
            st.info("Kolom Line_Name tidak ditemukan, semua data ditampilkan.")
        
        if not plot_df.empty:
            # 1. Peta interaktif (OpenStreetMap)
            st.subheader("📍 Peta Lintasan dengan Warna TMI")
            fig_map = px.scatter_mapbox(plot_df, lat="Latitude", lon="Longitude", 
                                        color="TMI", size=2,
                                        hover_name="Line_Name" if 'Line_Name' in plot_df else None,
                                        hover_data=["TMI", "Field_filtered"],
                                        color_continuous_scale="Viridis",
                                        title="Anomali TMI sepanjang lintasan")
            fig_map.update_layout(mapbox_style="open-street-map", mapbox_zoom=8)
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
            # 2. Plot 2D Longitude vs Latitude
            st.subheader("📈 Plot Lintasan 2D (Longitude vs Latitude)")
            fig_scatter = px.scatter(plot_df, x="Longitude", y="Latitude", color="TMI",
                                     hover_data=["Line_Name", "TMI"] if 'Line_Name' in plot_df else ["TMI"],
                                     color_continuous_scale="Viridis",
                                     title="Lintasan Survei diwarnai TMI")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 3. Profil anomali sepanjang jarak per line
            st.subheader("📉 Profil Anomali TMI Sepanjang Lintasan")
            if 'Line_Name' in plot_df.columns:
                for line in plot_df['Line_Name'].unique():
                    line_df = plot_df[plot_df['Line_Name'] == line].sort_values('datetime')
                    if len(line_df) > 1:
                        dist = compute_distance_along_line(line_df)
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(x=dist/1000, y=line_df['TMI'], 
                                                      mode='lines+markers', name=line,
                                                      marker=dict(size=3)))
                        fig_line.update_layout(title=f"Line: {line}",
                                               xaxis_title="Jarak (km)",
                                               yaxis_title="TMI (nT)")
                        st.plotly_chart(fig_line, use_container_width=True)
                    else:
                        st.write(f"Line {line} tidak cukup titik untuk plot profil.")
            else:
                # Jika tidak ada Line_Name, buat satu profil keseluruhan
                if len(plot_df) > 1:
                    dist = compute_distance_along_line(plot_df)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=dist/1000, y=plot_df['TMI'], 
                                                  mode='lines+markers', name='TMI',
                                                  marker=dict(size=3)))
                    fig_line.update_layout(title="Profil Anomali TMI (seluruh track)",
                                           xaxis_title="Jarak (km)",
                                           yaxis_title="TMI (nT)")
                    st.plotly_chart(fig_line, use_container_width=True)
            
            # 4. Perbandingan Field, IGRF, TMI (sampel 500 titik)
            st.subheader("📊 Perbandingan Field, IGRF, dan TMI")
            sample = plot_df.head(500)
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(y=sample['Field_filtered'], mode='lines', name='Field (filtered)'))
            fig_compare.add_trace(go.Scatter(y=sample['IGRF'], mode='lines', name='IGRF'))
            fig_compare.add_trace(go.Scatter(y=sample['TMI'], mode='lines', name='TMI'))
            fig_compare.update_layout(title="Kurva Field, IGRF, TMI", xaxis_title="Index", yaxis_title="nT")
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # 5. Plot sebelum/sesudah filter Field dan Altitude
            st.subheader("🔍 Verifikasi Filter: Field (Original vs Filtered)")
            fig_field = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                      subplot_titles=("Field Original", "Field Filtered"))
            fig_field.add_trace(go.Scatter(y=survey_df['Field'], mode='lines', name='Original'), row=1, col=1)
            fig_field.add_trace(go.Scatter(y=survey_df['Field_filtered'], mode='lines', name='Filtered', line=dict(color='red')), row=2, col=1)
            fig_field.update_layout(height=500)
            st.plotly_chart(fig_field, use_container_width=True)
            
            if 'Altitude' in survey_df.columns and 'Altitude_filtered' in survey_df.columns:
                st.subheader("🔍 Verifikasi Filter: Altitude (Original vs Filtered)")
                fig_alt = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        subplot_titles=("Altitude Original", "Altitude Filtered"))
                fig_alt.add_trace(go.Scatter(y=survey_df['Altitude'], mode='lines', name='Original'), row=1, col=1)
                fig_alt.add_trace(go.Scatter(y=survey_df['Altitude_filtered'], mode='lines', name='Filtered', line=dict(color='green')), row=2, col=1)
                fig_alt.update_layout(height=500)
                st.plotly_chart(fig_alt, use_container_width=True)
        
        else:
            st.warning("⚠️ Tidak ada data untuk diplot setelah seleksi line.")
        
        # Download hasil
        st.header("💾 Download Data Hasil")
        output_cols = ['datetime', 'Latitude', 'Longitude', 'Easting', 'Northing',
                       'Field', 'Field_filtered', 'Altitude', 'Altitude_filtered',
                       'Depth', 'Line_Name', 'IGRF', 'Diurnal_Correction', 'TMI']
        output_cols = [col for col in output_cols if col in survey_df.columns]
        output_df = survey_df[output_cols]
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "marine_magnetic_processed.csv", "text/csv")

else:
    st.info("⬅️ Silakan upload file Excel atau CSV dengan kolom yang diperlukan.")

st.markdown("---")
st.caption("Catatan: Untuk IGRF, pastikan `geomag` terinstal. Tanda `*` pada kolom numerik otomatis diubah menjadi NaN dan diinterpolasi.")
