import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline
from scipy.stats import median_abs_deviation
from math import radians, sin, cos, sqrt, atan2

# ================== UTILITY FUNCTIONS ==================

def clean_string_placeholders(df, columns):
    """Ganti string 'nan', 'NaN', '*', '' dengan np.nan pada kolom yang ditentukan."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
    return df

def clean_numeric_columns(df, columns):
    """Bersihkan placeholder dan konversi ke numerik, yang tidak bisa jadi NaN."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_data(uploaded_file):
    """Load Excel (all sheets) or CSV (single). Returns dict of DataFrames."""
    if uploaded_file.name.endswith('.xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        sheets = {}
        for sheet in sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            # Bersihkan placeholder di semua kolom potensial
            all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                        'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
            df = clean_string_placeholders(df, all_cols)
            numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
            df = clean_numeric_columns(df, numeric_cols)
            sheets[sheet] = df
        return sheets
    else:
        # CSV single sheet
        df = pd.read_csv(uploaded_file)
        all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                    'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
        df = clean_string_placeholders(df, all_cols)
        numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
        df = clean_numeric_columns(df, numeric_cols)
        return {'data': df}

def parse_datetime(df, sheet_name):
    """Gabungkan Reading_Date dan Reading_Time menjadi datetime, hapus baris tidak valid."""
    # Pastikan kolom date/time tidak mengandung placeholder (sudah diubah jadi NaN)
    df_clean = df.dropna(subset=['Reading_Date', 'Reading_Time']).copy()
    if len(df_clean) == 0:
        raise ValueError(f"Sheet '{sheet_name}': Tidak ada baris dengan Reading_Date dan Reading_Time yang valid.")
    
    datetime_str = df_clean['Reading_Date'].astype(str) + ' ' + df_clean['Reading_Time'].astype(str)
    
    # Coba berbagai format
    try:
        dt = pd.to_datetime(datetime_str, utc=True, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except (ValueError, TypeError):
        try:
            dt = pd.to_datetime(datetime_str, utc=True, format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
        except (ValueError, TypeError):
            try:
                dt = pd.to_datetime(datetime_str, utc=True, format='mixed')
            except (ValueError, TypeError):
                dt = pd.to_datetime(datetime_str, utc=True, errors='coerce')
    
    valid_mask = dt.notna()
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        example = datetime_str[~valid_mask].iloc[0] if n_invalid > 0 else ''
        raise ValueError(f"Sheet '{sheet_name}': {n_invalid} baris tidak dapat di-parse. Contoh gagal: '{example}'")
    
    df_clean['datetime'] = dt
    return df_clean

def separate_base_and_survey(df, sheet_name):
    """Pisahkan data survei dan base. Base menggunakan Tbase/Fbase dari sheet yang sama."""
    survey_df = df[df['Field'].notna()].copy()
    base_df = df[df['Tbase'].notna() & df['Fbase'].notna()].copy()
    
    if not base_df.empty:
        # Konstruksi datetime untuk base
        if 'Reading_Date' in base_df.columns:
            base_df['base_datetime'] = pd.to_datetime(base_df['Reading_Date'].astype(str) + ' ' + base_df['Tbase'].astype(str),
                                                      utc=True, errors='coerce')
        else:
            if not survey_df.empty:
                ref_date = survey_df['datetime'].min().date()
                base_df['base_datetime'] = pd.to_datetime(ref_date.strftime('%Y-%m-%d') + ' ' + base_df['Tbase'].astype(str),
                                                          utc=True, errors='coerce')
            else:
                base_df['base_datetime'] = pd.to_datetime('1970-01-01 ' + base_df['Tbase'].astype(str),
                                                          utc=True, errors='coerce')
            st.warning(f"Sheet '{sheet_name}': Kolom Reading_Date tidak ditemukan untuk data base. Menggunakan tanggal survei pertama.")
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

st.set_page_config(page_title="Marine Magnetic Processing (Multi‑Sheet)", layout="wide")
st.title("🌊 Pengolahan Data Magnetik Kelautan – Multi Sheet")

uploaded_file = st.sidebar.file_uploader("📂 Upload file Excel (multi‑sheet) atau CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Load all sheets (sudah dibersihkan)
    all_sheets = load_data(uploaded_file)
    sheet_names = list(all_sheets.keys())
    st.subheader(f"📑 Sheet yang terdeteksi: {', '.join(sheet_names)}")
    
    st.sidebar.header("🔧 Parameter Filtering (diterapkan ke semua sheet)")
    
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

    # Manual IGRF (global untuk semua sheet)
    st.sidebar.header("🧲 IGRF Source (Manual, berlaku untuk semua sheet)")
    igrf_option = st.sidebar.radio(
        "Pilih cara input IGRF:",
        ["Constant value", "Upload IGRF file (CSV)", "Skip IGRF (set to 0)"]
    )
    constant_igrf = None
    igrf_file = None
    if igrf_option == "Constant value":
        constant_igrf = st.sidebar.number_input("Nilai IGRF konstan (nT):", value=45000.0, step=100.0)
    elif igrf_option == "Upload IGRF file (CSV)":
        igrf_file = st.sidebar.file_uploader("Upload CSV dengan kolom 'datetime' atau index dan 'IGRF'", type=['csv'])
        if igrf_file:
            st.sidebar.success("File IGRF terupload.")

    anomaly_type = st.sidebar.selectbox("Peta Anomali menggunakan:", ["Field_filtered", "TMI"])

    if st.button("🚀 Proses Semua Sheet"):
        all_results = []
        progress_bar = st.progress(0)
        for idx, sheet in enumerate(sheet_names):
            st.write(f"⏳ Memproses sheet: **{sheet}**")
            df_raw = all_sheets[sheet].copy()
            
            # Parse datetime (setelah sebelumnya sudah dibersihkan placeholder)
            try:
                df_raw = parse_datetime(df_raw, sheet)
            except Exception as e:
                st.error(f"Sheet {sheet}: {e}")
                continue
            
            survey_df, base_df = separate_base_and_survey(df_raw, sheet)
            if survey_df.empty:
                st.warning(f"Sheet {sheet}: Tidak ada data survei (Field kosong). Dilewati.")
                continue
            
            # Filter Field (jika ada)
            if field_method != "None":
                survey_df['Field_filtered'] = apply_filter(survey_df['Field'], field_method, **field_params)
            else:
                survey_df['Field_filtered'] = survey_df['Field']
            
            # Filter Altitude (jika ada, dan hanya jika kolom Altitude tidak kosong semua)
            if alt_method != "None" and survey_df['Altitude'].notna().any():
                survey_df['Altitude_filtered'] = apply_filter(survey_df['Altitude'], alt_method, **alt_params)
            else:
                survey_df['Altitude_filtered'] = survey_df['Altitude']
            
            # Koreksi diurnal (menggunakan base_df dari sheet ini)
            if not base_df.empty:
                diurnal_corr = compute_diurnal_correction(survey_df, base_df, reference_method='first')
                survey_df['Diurnal_Correction'] = diurnal_corr
            else:
                survey_df['Diurnal_Correction'] = 0.0
                st.info(f"Sheet {sheet}: Tidak ada data base -> koreksi diurnal = 0")
            
            # IGRF Manual (global)
            if igrf_option == "Constant value":
                survey_df['IGRF'] = constant_igrf
            elif igrf_option == "Upload IGRF file (CSV)" and igrf_file is not None:
                igrf_df = pd.read_csv(igrf_file)
                if 'datetime' in igrf_df.columns:
                    igrf_df['datetime'] = pd.to_datetime(igrf_df['datetime'], utc=True)
                    survey_df = survey_df.merge(igrf_df[['datetime', 'IGRF']], on='datetime', how='left')
                else:
                    if len(igrf_df) == len(survey_df):
                        survey_df['IGRF'] = igrf_df['IGRF'].values
                    else:
                        st.error(f"Sheet {sheet}: Panjang file IGRF tidak sama dengan data survei. IGRF diisi 0.")
                        survey_df['IGRF'] = 0.0
            else:
                survey_df['IGRF'] = 0.0
            
            survey_df['IGRF'] = survey_df['IGRF'].fillna(0.0)
            survey_df['TMI'] = survey_df['Field_filtered'] - survey_df['IGRF'] - survey_df['Diurnal_Correction']
            
            # Tambahkan kolom sheet name
            survey_df['Sheet_Name'] = sheet
            
            all_results.append(survey_df)
            progress_bar.progress((idx+1)/len(sheet_names))
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            st.session_state['final_df'] = final_df
            st.success(f"✅ Selesai! Total {len(final_df)} titik dari {len(all_results)} sheet.")
        else:
            st.error("Tidak ada data yang berhasil diproses.")
    
    if 'final_df' in st.session_state:
        final_df = st.session_state['final_df']
        sheets_present = final_df['Sheet_Name'].unique()
        
        st.subheader("📊 Hasil gabungan (10 baris pertama)")
        st.dataframe(final_df[['Sheet_Name', 'datetime', 'Field', 'Field_filtered', 'IGRF', 'Diurnal_Correction', 'TMI']].head(10))
        
        # Pilih sheet untuk ditampilkan di plot
        selected_sheets = st.multiselect("Pilih sheet untuk ditampilkan di plot", sheets_present, default=sheets_present)
        plot_df = final_df[final_df['Sheet_Name'].isin(selected_sheets)].copy()
        
        if not plot_df.empty:
            # ========== 1. PLOT PERBANDINGAN FIELD sebelum/sesudah filter ==========
            st.header("📈 Perbandingan Field Original vs Filtered (per sheet)")
            fig_field, ax_field = plt.subplots(figsize=(12, 5))
            for sheet in selected_sheets:
                df_sheet = plot_df[plot_df['Sheet_Name'] == sheet].sort_values('datetime')
                ax_field.plot(df_sheet['Field'].values, '--', alpha=0.5, label=f'{sheet} Original')
                ax_field.plot(df_sheet['Field_filtered'].values, '-', alpha=0.8, label=f'{sheet} Filtered')
            ax_field.set_xlabel('Index (urut waktu per sheet)')
            ax_field.set_ylabel('nT')
            ax_field.set_title('Field Original vs Filtered')
            ax_field.legend(loc='best', fontsize=8, ncol=2)
            ax_field.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_field)
            plt.close(fig_field)
            
            # ========== 2. PLOT TMI dengan sumbu x = waktu ==========
            st.header("📉 Total Magnetic Intensity (TMI) setelah koreksi")
            fig_tmi, ax_tmi = plt.subplots(figsize=(12, 5))
            for sheet in selected_sheets:
                df_sheet = plot_df[plot_df['Sheet_Name'] == sheet].sort_values('datetime')
                ax_tmi.plot(df_sheet['datetime'], df_sheet['TMI'], label=sheet)   # x = datetime, y = TMI
            ax_tmi.set_xlabel('Waktu (UTC)')
            ax_tmi.set_ylabel('nT')
            ax_tmi.set_title('TMI')
            ax_tmi.legend()
            ax_tmi.grid(True, linestyle=':', alpha=0.5)
            # Optional: format tanggal agar lebih rapi
            fig_tmi.autofmt_xdate()  # memutar label tanggal jika perlu
            st.pyplot(fig_tmi)
            plt.close(fig_tmi)
            
            # ========== 3. PETA LINTASAN HITAM ==========
            st.header("🗺️ Peta Lintasan Survei (Garis Hitam) dengan Titik Awal & Akhir per Sheet")
            fig_track, ax_track = plt.subplots(figsize=(10, 8))
            for sheet in selected_sheets:
                df_sheet = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=['Latitude', 'Longitude']).sort_values('datetime')
                if not df_sheet.empty:
                    ax_track.plot(df_sheet['Longitude'], df_sheet['Latitude'], linewidth=1.5, label=sheet)
                    first = df_sheet.iloc[0]
                    last = df_sheet.iloc[-1]
                    ax_track.plot(first['Longitude'], first['Latitude'], 'go', markersize=6)
                    ax_track.plot(last['Longitude'], last['Latitude'], 'ro', markersize=6)
                    ax_track.annotate(f"{sheet}\nStart: {first['datetime'].strftime('%H:%M:%S')}", 
                                      (first['Longitude'], first['Latitude']), textcoords="offset points", xytext=(5,5), fontsize=7)
                    ax_track.annotate(f"End: {last['datetime'].strftime('%H:%M:%S')}", 
                                      (last['Longitude'], last['Latitude']), textcoords="offset points", xytext=(5,-10), fontsize=7)
            ax_track.set_xlabel('Longitude')
            ax_track.set_ylabel('Latitude')
            ax_track.set_title('Lintasan Survei (warna berbeda per sheet)')
            ax_track.legend(loc='best')
            ax_track.grid(True, linestyle=':', alpha=0.5)
            st.pyplot(fig_track)
            plt.close(fig_track)
            
            # ========== 4. PETA ANOMALI MAGNET ==========
            st.header(f"🗺️ Peta Anomali Magnet ({anomaly_type})")
            combine_anom = st.checkbox("Gabungkan semua sheet dalam satu plot", value=True)
            if combine_anom:
                fig_anom, ax_anom = plt.subplots(figsize=(10, 8))
                anomaly_df = plot_df.dropna(subset=[anomaly_type, 'Latitude', 'Longitude'])
                if not anomaly_df.empty:
                    vmin = anomaly_df[anomaly_type].min()
                    vmax = anomaly_df[anomaly_type].max()
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sc = ax_anom.scatter(anomaly_df['Longitude'], anomaly_df['Latitude'],
                                         c=anomaly_df[anomaly_type], s=10, cmap='viridis', norm=norm)
                    plt.colorbar(sc, ax=ax_anom, label=f'{anomaly_type} (nT)')
                    ax_anom.set_xlabel('Longitude')
                    ax_anom.set_ylabel('Latitude')
                    ax_anom.set_title(f'Distribusi {anomaly_type} (semua sheet)')
                    ax_anom.grid(True, linestyle=':', alpha=0.5)
                    st.pyplot(fig_anom)
                else:
                    st.warning("Tidak ada data anomali valid.")
                plt.close(fig_anom)
            else:
                for sheet in selected_sheets:
                    anomaly_df = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=[anomaly_type, 'Latitude', 'Longitude'])
                    if not anomaly_df.empty:
                        fig_anom, ax_anom = plt.subplots(figsize=(8, 6))
                        vmin = anomaly_df[anomaly_type].min()
                        vmax = anomaly_df[anomaly_type].max()
                        norm = Normalize(vmin=vmin, vmax=vmax)
                        sc = ax_anom.scatter(anomaly_df['Longitude'], anomaly_df['Latitude'],
                                             c=anomaly_df[anomaly_type], s=10, cmap='viridis', norm=norm)
                        plt.colorbar(sc, ax=ax_anom, label=f'{anomaly_type} (nT)')
                        ax_anom.set_xlabel('Longitude')
                        ax_anom.set_ylabel('Latitude')
                        ax_anom.set_title(f'{sheet} - {anomaly_type}')
                        ax_anom.grid(True, linestyle=':', alpha=0.5)
                        st.pyplot(fig_anom)
                        plt.close(fig_anom)
            
            # ========== 5. PROFIL ANOMALI SEPANJANG JARAK ==========
            st.header("📏 Profil Anomali Sepanjang Jarak")
            for sheet in selected_sheets:
                anomaly_df = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=[anomaly_type, 'Latitude', 'Longitude']).sort_values('datetime')
                if len(anomaly_df) > 1:
                    dist = compute_distance_along_line(anomaly_df)
                    fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
                    ax_prof.plot(dist/1000, anomaly_df[anomaly_type], 'b-', linewidth=1, marker='.', markersize=2)
                    ax_prof.set_xlabel('Jarak (km)')
                    ax_prof.set_ylabel(f'{anomaly_type} (nT)')
                    ax_prof.set_title(f'Sheet {sheet}')
                    ax_prof.grid(True, linestyle=':', alpha=0.5)
                    st.pyplot(fig_prof)
                    plt.close(fig_prof)
                else:
                    st.info(f"Sheet {sheet}: Tidak cukup titik untuk profil.")
            
            # ========== 6. DOWNLOAD DATA ==========
            st.header("💾 Download Data Hasil (gabungan semua sheet)")
            output_cols = ['Sheet_Name', 'datetime', 'Latitude', 'Longitude', 'Easting', 'Northing',
                           'Field', 'Field_filtered', 'Altitude', 'Altitude_filtered',
                           'Depth', 'Line_Name', 'IGRF', 'Diurnal_Correction', 'TMI']
            output_cols = [c for c in output_cols if c in final_df.columns]
            output_df = final_df[output_cols]
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "marine_magnetic_all_sheets.csv", "text/csv")
else:
    st.info("⬅️ Upload file Excel (bisa multi‑sheet) atau CSV.")
