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
from math import radians, sin, cos, sqrt, atan2

# ================== UTILITY FUNCTIONS ==================

def clean_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 
                    'Altitude', 'Depth', 'Fbase']
    df = clean_numeric_columns(df, numeric_cols)
    return df

def parse_datetime(df):
    try:
        df['datetime'] = pd.to_datetime(df['Reading_Date'].astype(str) + ' ' + df['Reading_Time'].astype(str), utc=True)
    except Exception as e:
        raise ValueError(f"Gagal parse datetime: {e}")
    return df

def separate_base_and_survey(df):
    survey_df = df[df['Field'].notna()].copy()
    base_df = df[df['Tbase'].notna() & df['Fbase'].notna()].copy()
    if not base_df.empty:
        if 'Reading_Date' in base_df.columns:
            base_df['base_datetime'] = pd.to_datetime(base_df['Reading_Date'].astype(str) + ' ' + base_df['Tbase'].astype(str), utc=True, errors='coerce')
        else:
            if not survey_df.empty:
                ref_date = survey_df['datetime'].min().date()
                base_df['base_datetime'] = pd.to_datetime(ref_date.strftime('%Y-%m-%d') + ' ' + base_df['Tbase'].astype(str), utc=True, errors='coerce')
            else:
                base_df['base_datetime'] = pd.to_datetime('1970-01-01 ' + base_df['Tbase'].astype(str), utc=True, errors='coerce')
            st.warning("Kolom Reading_Date tidak ditemukan untuk data base. Menggunakan tanggal survei pertama.")
        base_df = base_df.dropna(subset=['base_datetime'])
    return survey_df, base_df

def hampel_filter(series, window_size=5, n_sigmas=3.0):
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
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    if series.isna().any():
        series = series.fillna(series.median())
    return filtfilt(b, a, series)

def apply_filter(series, method, **params):
    if method == 'Hampel (despiking)':
        cleaned, _ = hampel_filter(series, window_size=params.get('window', 5), n_sigmas=params.get('threshold', 3.0))
        result = interpolate_nan(cleaned, method='cubic')
    elif method == 'Moving Average':
        result = moving_average(series, window=params.get('window', 5))
    elif method == 'Savitzky-Golay':
        window = params.get('window', 11)
        if window % 2 == 0:
            window += 1
        temp = series.interpolate(method='linear', limit_direction='both')
        result = savgol_filter(temp, window_length=window, polyorder=3)
        result = pd.Series(result, index=series.index)
    elif method == 'Butterworth Lowpass':
        result = butterworth_filter(series, cutoff=params.get('cutoff', 0.1), fs=1.0, order=4)
    else:
        result = series.copy()
    return result

def compute_diurnal_correction(survey_df, base_df, reference_method='first'):
    if base_df.empty:
        return np.zeros(len(survey_df))
    base_df = base_df.dropna(subset=['base_datetime']).sort_values('base_datetime')
    if base_df.empty:
        return np.zeros(len(survey_df))
    survey_df_valid = survey_df.dropna(subset=['datetime'])
    if survey_df_valid.empty:
        return np.zeros(len(survey_df))
    base_ts = base_df['base_datetime'].astype('int64') // 10**9
    base_vals = base_df['Fbase'].values
    survey_ts = survey_df_valid['datetime'].astype('int64') // 10**9
    interpolated = np.interp(survey_ts, base_ts, base_vals)
    if reference_method == 'first':
        ref_val = base_vals[0]
    elif reference_method == 'mean':
        ref_val = np.mean(base_vals)
    else:
        ref_val = 0.0
    correction = np.zeros(len(survey_df))
    correction[survey_df_valid.index] = interpolated - ref_val
    return correction

def compute_distance_along_line(df):
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

# ================== MAIN STREAMLIT APP ==================

st.set_page_config(page_title="Pengolahan Data Magnetik Kelautan", layout="wide")
st.title("🌊 Pengolahan Data Magnetik Kelautan + Peta Anomali")

uploaded_file = st.sidebar.file_uploader("📂 Upload file Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.subheader("📋 Data Awal (10 baris pertama)")
    st.dataframe(df_raw.head(10))

    try:
        df_raw = parse_datetime(df_raw)
    except Exception as e:
        st.error(f"❌ Gagal parse datetime: {e}")
        st.stop()

    survey_df, base_df = separate_base_and_survey(df_raw)
    if survey_df.empty:
        st.error("❌ Tidak ada data survei (kolom Field kosong)")
        st.stop()
    if base_df.empty:
        st.warning("⚠️ Tidak ada data base (Tbase/Fbase kosong). Koreksi diurnal tidak dilakukan.")

    st.sidebar.header("🔧 Parameter Filtering")
    
    # Filter Field
    field_method = st.sidebar.selectbox("Filter Field", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay", "Butterworth Lowpass"])
    field_params = {}
    if field_method == "Hampel (despiking)":
        field_params['window'] = st.sidebar.slider("Window Hampel", 3, 21, 5, 2)
        field_params['threshold'] = st.sidebar.slider("Threshold sigma", 1.0, 5.0, 3.0, 0.5)
    elif field_method in ["Moving Average", "Savitzky-Golay"]:
        field_params['window'] = st.sidebar.slider("Window size", 3, 51, 11, 2)
    elif field_method == "Butterworth Lowpass":
        field_params['cutoff'] = st.sidebar.slider("Cutoff frequency (0-0.5)", 0.01, 0.5, 0.1, 0.01)
    
    # Filter Altitude
    alt_method = st.sidebar.selectbox("Filter Altitude", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay"])
    alt_params = {}
    if alt_method == "Hampel (despiking)":
        alt_params['window'] = st.sidebar.slider("Window Alt Hampel", 3, 21, 5, 2)
        alt_params['threshold'] = st.sidebar.slider("Threshold sigma Alt", 1.0, 5.0, 3.0, 0.5)
    elif alt_method in ["Moving Average", "Savitzky-Golay"]:
        alt_params['window'] = st.sidebar.slider("Window size Alt", 3, 51, 11, 2)

    # *************** MANUAL IGRF INPUT ***************
    st.sidebar.header("🧲 IGRF Source (Manual)")
    igrf_option = st.sidebar.radio(
        "Pilih cara input IGRF:",
        ["Constant value", "Upload IGRF file (CSV)", "Skip IGRF (set to 0)"]
    )
    
    constant_igrf = None
    igrf_file = None
    if igrf_option == "Constant value":
        constant_igrf = st.sidebar.number_input("Masukkan nilai IGRF konstan (nT):", value=45000.0, step=100.0, format="%.1f")
    elif igrf_option == "Upload IGRF file (CSV)":
        igrf_file = st.sidebar.file_uploader("Upload CSV dengan kolom 'datetime' atau index dan 'IGRF'", type=['csv'])
        if igrf_file is not None:
            st.sidebar.success("File IGRF terupload.")
    # else: Skip -> IGRF = 0

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
                ref_method = st.selectbox("Metode referensi base", ['first', 'mean', 'none'], key='ref_method')
                diurnal_corr = compute_diurnal_correction(survey_df, base_df, reference_method=ref_method if ref_method != 'none' else 'constant')
                survey_df['Diurnal_Correction'] = diurnal_corr
            else:
                survey_df['Diurnal_Correction'] = 0.0

            # ---------- IGRF Manual ----------
            if igrf_option == "Constant value":
                survey_df['IGRF'] = constant_igrf
            elif igrf_option == "Upload IGRF file (CSV)" and igrf_file is not None:
                igrf_df = pd.read_csv(igrf_file)
                # Try to merge based on 'datetime' column if present
                if 'datetime' in igrf_df.columns:
                    igrf_df['datetime'] = pd.to_datetime(igrf_df['datetime'], utc=True)
                    survey_df = survey_df.merge(igrf_df[['datetime', 'IGRF']], on='datetime', how='left')
                else:
                    # Assume same order (merge by index)
                    if len(igrf_df) == len(survey_df):
                        survey_df['IGRF'] = igrf_df['IGRF'].values
                    else:
                        st.error("Panjang file IGRF tidak sama dengan data survei. Gunakan kolom 'datetime' untuk pencocokan.")
                        survey_df['IGRF'] = np.nan
            else:  # Skip
                survey_df['IGRF'] = 0.0

            # Fill any missing IGRF with 0 or constant? We'll set to 0
            survey_df['IGRF'] = survey_df['IGRF'].fillna(0.0)

            # TMI calculation
            survey_df['TMI'] = survey_df['Field_filtered'] - survey_df['IGRF'] - survey_df['Diurnal_Correction']
            
            st.session_state['survey_df'] = survey_df
            st.success("✅ Proses selesai!")

    if 'survey_df' in st.session_state:
        survey_df = st.session_state['survey_df']
        
        st.subheader("📊 Hasil setelah koreksi (10 baris pertama)")
        st.dataframe(survey_df[['datetime', 'Field_filtered', 'IGRF', 'Diurnal_Correction', 'TMI']].head(10))

        # ========== VISUALIZATIONS ==========
        st.header("🗺️ Visualisasi Peta dan Lintasan Anomali")
        
        if 'Line_Name' in survey_df.columns:
            available_lines = survey_df['Line_Name'].dropna().unique()
            if len(available_lines) > 0:
                selected_lines = st.multiselect("Pilih Line Name", available_lines, default=available_lines)
                plot_df = survey_df[survey_df['Line_Name'].isin(selected_lines)]
            else:
                plot_df = survey_df
        else:
            plot_df = survey_df
            st.info("Kolom Line_Name tidak ditemukan.")
        
        if not plot_df.empty:
            # Map
            fig_map = px.scatter_mapbox(plot_df, lat="Latitude", lon="Longitude", color="TMI", size=2,
                                        hover_name="Line_Name" if 'Line_Name' in plot_df else None,
                                        hover_data=["TMI"], color_continuous_scale="Viridis")
            fig_map.update_layout(mapbox_style="open-street-map", mapbox_zoom=8, margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
            # 2D scatter
            fig_scatter = px.scatter(plot_df, x="Longitude", y="Latitude", color="TMI",
                                     color_continuous_scale="Viridis", title="Lintasan diwarnai TMI")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Profile per line
            st.subheader("📉 Profil Anomali TMI")
            if 'Line_Name' in plot_df.columns:
                for line in plot_df['Line_Name'].unique():
                    line_df = plot_df[plot_df['Line_Name'] == line].sort_values('datetime')
                    if len(line_df) > 1:
                        dist = compute_distance_along_line(line_df)
                        fig_line = go.Figure()
                        fig_line.add_trace(go.Scatter(x=dist/1000, y=line_df['TMI'], mode='lines+markers', name=line))
                        fig_line.update_layout(title=f"Line {line}", xaxis_title="Jarak (km)", yaxis_title="TMI (nT)")
                        st.plotly_chart(fig_line, use_container_width=True)
            else:
                if len(plot_df) > 1:
                    dist = compute_distance_along_line(plot_df)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=dist/1000, y=plot_df['TMI'], mode='lines+markers'))
                    fig_line.update_layout(xaxis_title="Jarak (km)", yaxis_title="TMI (nT)")
                    st.plotly_chart(fig_line, use_container_width=True)
        
        # Download result
        st.header("💾 Download Data")
        output_cols = ['datetime', 'Latitude', 'Longitude', 'Easting', 'Northing',
                       'Field', 'Field_filtered', 'Altitude', 'Altitude_filtered',
                       'Depth', 'Line_Name', 'IGRF', 'Diurnal_Correction', 'TMI']
        output_cols = [c for c in output_cols if c in survey_df.columns]
        output_df = survey_df[output_cols]
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "marine_magnetic_processed.csv", "text/csv")
else:
    st.info("⬅️ Upload file Excel atau CSV dengan kolom yang diperlukan.")
