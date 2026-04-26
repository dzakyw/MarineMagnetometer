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

# ================== Fungsi Utilities ==================
def load_data(uploaded_file):
    """Load file xlsx atau csv"""
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return df

def parse_datetime(df):
    """Gabungkan Reading_Date dan Reading_Time menjadi kolom datetime"""
    try:
        df['datetime'] = pd.to_datetime(df['Reading_Date'].astype(str) + ' ' + df['Reading_Time'].astype(str))
    except:
        if 'datetime' not in df.columns:
            raise ValueError("Tidak dapat menggabungkan Reading_Date dan Reading_Time. Periksa format.")
    return df

def separate_base_and_survey(df):
    """Pisahkan baris data base (Tbase dan Fbase tidak null) dan data survei (Field tidak null)"""
    survey_df = df[df['Field'].notna()].copy()
    base_df = df[df['Tbase'].notna() & df['Fbase'].notna()].copy()
    if not base_df.empty:
        base_df['base_datetime'] = pd.to_datetime(base_df['Tbase'])
    return survey_df, base_df

def hampel_filter(series, window_size=5, n_sigmas=3.0):
    rolling_median = series.rolling(window=window_size, center=True).median()
    mad = series.rolling(window=window_size, center=True).apply(
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
        interpolated = series.interpolate(method='linear')
    return pd.Series(interpolated, index=idx)

def moving_average(series, window=5):
    return series.rolling(window=window, center=True).mean()

def butterworth_filter(series, cutoff=0.1, fs=1.0, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
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
        result = savgol_filter(series, window_length=window, polyorder=3)
    elif method == 'Butterworth Lowpass':
        result = butterworth_filter(series, cutoff=params.get('cutoff', 0.1), fs=1.0, order=4)
    else:
        result = series.copy()
    return result

def compute_igrf(lat, lon, alt_m, datetime_obj):
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
    """Hitung jarak kumulatif (meter) berdasarkan koordinat lon/lat (Haversine)"""
    from math import radians, sin, cos, sqrt, atan2
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371000  # radius bumi meter
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

# ================== Main Streamlit App ==================
st.set_page_config(page_title="Marine Magnetic Data Processing & Mapping", layout="wide")
st.title("Pengolahan Data Magnetik Kelautan + Peta Anomali")

uploaded_file = st.sidebar.file_uploader("Upload file Excel/CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.subheader("Data Awal (10 baris pertama)")
    st.dataframe(df_raw.head(10))

    # Parsing datetime
    try:
        df_raw = parse_datetime(df_raw)
    except Exception as e:
        st.error(f"Gagal parse datetime: {e}")
        st.stop()

    survey_df, base_df = separate_base_and_survey(df_raw)
    if survey_df.empty:
        st.error("Tidak ada data survei (kolom Field kosong)")
        st.stop()
    if base_df.empty:
        st.warning("Tidak ada data base. Koreksi diurnal tidak dilakukan.")
    
    # Sidebar filter parameters
    st.sidebar.header("Parameter Filtering")
    
    # Filter Field
    field_method = st.sidebar.selectbox("Filter Field", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay", "Butterworth Lowpass"])
    field_params = {}
    if field_method == "Hampel (despiking)":
        field_params['window'] = st.sidebar.slider("Window Hampel", 3, 21, 5, 2)
        field_params['threshold'] = st.sidebar.slider("Threshold sigma", 1.0, 5.0, 3.0, 0.5)
    elif field_method in ["Moving Average", "Savitzky-Golay"]:
        field_params['window'] = st.sidebar.slider("Window size", 3, 51, 11, 2)
    elif field_method == "Butterworth Lowpass":
        field_params['cutoff'] = st.sidebar.slider("Cutoff frequency", 0.01, 0.5, 0.1, 0.01)
    
    # Filter Altitude
    alt_method = st.sidebar.selectbox("Filter Altitude", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay"])
    alt_params = {}
    if alt_method == "Hampel (despiking)":
        alt_params['window'] = st.sidebar.slider("Window Alt Hampel", 3, 21, 5, 2)
        alt_params['threshold'] = st.sidebar.slider("Threshold sigma Alt", 1.0, 5.0, 3.0, 0.5)
    elif alt_method in ["Moving Average", "Savitzky-Golay"]:
        alt_params['window'] = st.sidebar.slider("Window Alt", 3, 51, 11, 2)
    
    # Tombol proses
    if st.button("Proses Filter & Koreksi"):
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
            diurnal_corr = compute_diurnal_correction(survey_df, base_df, reference_method=ref_method if ref_method != 'none' else 'constant')
            survey_df['Diurnal_Correction'] = diurnal_corr
        else:
            survey_df['Diurnal_Correction'] = 0.0
        
        # IGRF
        progress_bar = st.progress(0)
        igrf_vals = []
        for i, row in survey_df.iterrows():
            igrf = compute_igrf(row['Latitude'], row['Longitude'], row['Altitude_filtered'], row['datetime'])
            igrf_vals.append(igrf)
            progress_bar.progress((i+1)/len(survey_df))
        survey_df['IGRF'] = igrf_vals
        survey_df['TMI'] = survey_df['Field_filtered'] - survey_df['IGRF'] - survey_df['Diurnal_Correction']
        
        st.session_state['survey_df'] = survey_df
        st.success("Proses selesai!")
    
    # Jika sudah ada data di session_state
    if 'survey_df' in st.session_state:
        survey_df = st.session_state['survey_df']
        
        # Tampilkan hasil numerik
        st.subheader("Hasil setelah koreksi")
        st.dataframe(survey_df[['datetime', 'Field_filtered', 'IGRF', 'Diurnal_Correction', 'TMI']].head(10))
        
        # ========== PLOTTING PETA DAN LINTASAN ==========
        st.header("Visualisasi Peta dan Lintasan Anomali")
        
        # Pilih line name
        available_lines = survey_df['Line_Name'].dropna().unique()
        if len(available_lines) > 0:
            selected_lines = st.multiselect("Pilih Line Name untuk ditampilkan", available_lines, default=available_lines)
            plot_df = survey_df[survey_df['Line_Name'].isin(selected_lines)]
        else:
            plot_df = survey_df
            st.info("Tidak ada kolom Line_Name, plot semua data.")
        
        if not plot_df.empty:
            # 1. Peta interaktif dengan mapbox (butuh token gratis, tapi bisa pakai open-street-map)
            st.subheader("Peta Lintasan dengan Warna TMI")
            # Menggunakan plotly scatter_mapbox tanpa token (style='open-street-map')
            fig_map = px.scatter_mapbox(plot_df, lat="Latitude", lon="Longitude", 
                                        color="TMI", size=2,
                                        hover_name="Line_Name", hover_data=["TMI", "Field_filtered"],
                                        color_continuous_scale="Viridis",
                                        title="Anomali TMI sepanjang lintasan")
            fig_map.update_layout(mapbox_style="open-street-map", mapbox_zoom=8)
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
            
            # 2. Plot 2D: Longitude vs Latitude dengan warna TMI
            st.subheader("Plot Lintasan (Longitude vs Latitude) - Warna TMI")
            fig_scatter = px.scatter(plot_df, x="Longitude", y="Latitude", color="TMI",
                                     hover_data=["Line_Name", "TMI"], 
                                     color_continuous_scale="Viridis",
                                     title="Lintasan Survei")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 3. Plot profil anomali sepanjang jarak (per line)
            st.subheader("Profil Anomali TMI Sepanjang Lintasan")
            for line in plot_df['Line_Name'].unique():
                line_df = plot_df[plot_df['Line_Name'] == line].sort_values('datetime')
                if len(line_df) > 1:
                    # Hitung jarak kumulatif
                    dist = compute_distance_along_line(line_df)
                    fig_line = go.Figure()
                    fig_line.add_trace(go.Scatter(x=dist/1000, y=line_df['TMI'], mode='lines+markers', name=line))
                    fig_line.update_layout(title=f"Line: {line}",
                                           xaxis_title="Jarak (km)",
                                           yaxis_title="TMI (nT)")
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.write(f"Line {line} tidak cukup titik untuk plot profil.")
            
            # 4. (Opsional) Plot perbandingan Field, IGRF, TMI
            st.subheader("Perbandingan Field, IGRF, dan TMI (sampel 500 titik pertama)")
            sample = plot_df.head(500)
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(y=sample['Field_filtered'], mode='lines', name='Field (filtered)'))
            fig_compare.add_trace(go.Scatter(y=sample['IGRF'], mode='lines', name='IGRF'))
            fig_compare.add_trace(go.Scatter(y=sample['TMI'], mode='lines', name='TMI'))
            fig_compare.update_layout(title="Kurva Field, IGRF, TMI", xaxis_title="Index", yaxis_title="nT")
            st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk diplot.")
        
        # Download hasil
        st.header("Download Data Hasil")
        output_cols = ['datetime', 'Latitude', 'Longitude', 'Easting', 'Northing',
                       'Field', 'Field_filtered', 'Altitude', 'Altitude_filtered',
                       'Depth', 'Line_Name', 'IGRF', 'Diurnal_Correction', 'TMI']
        output_df = survey_df[output_cols]
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "marine_magnetic_processed.csv", "text/csv")
        
else:
    st.info("Upload file Excel/CSV dengan kolom yang sesuai.")
