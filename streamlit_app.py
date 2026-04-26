import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline, RBFInterpolator, griddata
from scipy.stats import median_abs_deviation
from math import radians, sin, cos, sqrt, atan2

# ================== UTILITY FUNCTIONS ==================

def clean_string_placeholders(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
    return df

def clean_numeric_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_data(uploaded_file):
    if uploaded_file.name.endswith('.xlsx'):
        xl = pd.ExcelFile(uploaded_file)
        sheet_names = xl.sheet_names
        sheets = {}
        for sheet in sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                        'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
            df = clean_string_placeholders(df, all_cols)
            numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
            df = clean_numeric_columns(df, numeric_cols)
            sheets[sheet] = df
        return sheets
    else:
        df = pd.read_csv(uploaded_file)
        all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                    'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
        df = clean_string_placeholders(df, all_cols)
        numeric_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
        df = clean_numeric_columns(df, numeric_cols)
        return {'data': df}

def parse_datetime(df, sheet_name):
    for col in ['Reading_Date', 'Reading_Time']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
    df_clean = df.dropna(subset=['Reading_Date', 'Reading_Time']).copy()
    if len(df_clean) == 0:
        raise ValueError(f"Sheet '{sheet_name}': Tidak ada baris dengan Reading_Date dan Reading_Time yang valid.")
    datetime_str = df_clean['Reading_Date'].astype(str) + ' ' + df_clean['Reading_Time'].astype(str)
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
    survey_df = df[df['Field'].notna()].copy()
    base_df = df[df['Tbase'].notna() & df['Fbase'].notna()].copy()
    if not base_df.empty:
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

def gridded_anomaly_map(x, y, z, method='cubic', grid_resolution=50):
    """Buat grid dari data tidak teratur.
    x, y: longitude, latitude (array 1D)
    z: TMI atau anomaly
    method: 'linear', 'cubic', 'rbf'
    grid_resolution: jumlah titik grid per sumbu
    """
    # Definisikan batas grid (perluas sedikit dari range data)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    # Tambahkan margin 5%
    margin = max((x_max - x_min)*0.05, 0.01)
    x_grid = np.linspace(x_min - margin, x_max + margin, grid_resolution)
    y_grid = np.linspace(y_min - margin, y_max + margin, grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    if method == 'rbf':
        # RBFInterpolator butuh input 2D (N,2)
        points = np.column_stack((x, y))
        values = z
        # Kernel bisa diatur, default multiquadric
        rbf = RBFInterpolator(points, values, kernel='thin_plate_spline', smoothing=0.0)
        Z = rbf(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    else:
        # griddata dengan method linear atau cubic
        Z = griddata((x, y), z, (X, Y), method=method)
    return X, Y, Z

# ================== MAIN STREAMLIT APP ==================

st.set_page_config(page_title="Marine Magnetic Processing with Gridding", layout="wide")
st.title("🌊 Pengolahan Data Magnetik Kelautan – Gridding Anomali + Lintasan Hitam")

uploaded_file = st.sidebar.file_uploader("📂 Upload file Excel (multi‑sheet) atau CSV", type=['xlsx', 'csv'])

if uploaded_file is not None:
    all_sheets = load_data(uploaded_file)
    sheet_names = list(all_sheets.keys())
    st.subheader(f"📑 Sheet yang terdeteksi: {', '.join(sheet_names)}")
    
    st.sidebar.header("🔧 Parameter Filtering")
    
    field_method = st.sidebar.selectbox("Filter Field", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay", "Butterworth Lowpass"])
    field_params = {}
    if field_method == "Hampel (despiking)":
        field_params['window'] = st.sidebar.slider("Window Hampel", 3, 21, 5, 2)
        field_params['threshold'] = st.sidebar.slider("Threshold sigma", 1.0, 5.0, 3.0, 0.5)
    elif field_method in ["Moving Average", "Savitzky-Golay"]:
        field_params['window'] = st.sidebar.slider("Window size", 3, 51, 11, 2)
    elif field_method == "Butterworth Lowpass":
        field_params['cutoff'] = st.sidebar.slider("Cutoff frequency (0-0.5)", 0.01, 0.5, 0.1, 0.01)
    
    alt_method = st.sidebar.selectbox("Filter Altitude", ["None", "Hampel (despiking)", "Moving Average", "Savitzky-Golay"])
    alt_params = {}
    if alt_method == "Hampel (despiking)":
        alt_params['window'] = st.sidebar.slider("Window Alt Hampel", 3, 21, 5, 2)
        alt_params['threshold'] = st.sidebar.slider("Threshold sigma Alt", 1.0, 5.0, 3.0, 0.5)
    elif alt_method in ["Moving Average", "Savitzky-Golay"]:
        alt_params['window'] = st.sidebar.slider("Window size Alt", 3, 51, 11, 2)

    st.sidebar.header("🧲 IGRF Source (Manual)")
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
    
    st.sidebar.header("🗺️ Gridding Options")
    gridding_method = st.sidebar.selectbox("Metode gridding", ["Tanpa Grid (scatter)", "Linear", "Cubic", "RBF (Thin Plate Spline)"])
    grid_resolution = st.sidebar.slider("Resolusi grid (jumlah titik)", 30, 150, 60, 10)
    show_track_lines = st.sidebar.checkbox("Tampilkan lintasan hitam di atas grid", value=True)

    if st.button("🚀 Proses Semua Sheet"):
        all_results = []
        progress_bar = st.progress(0)
        for idx, sheet in enumerate(sheet_names):
            st.write(f"⏳ Memproses sheet: **{sheet}**")
            df_raw = all_sheets[sheet].copy()
            try:
                df_raw = parse_datetime(df_raw, sheet)
            except Exception as e:
                st.error(f"Sheet {sheet}: {e}")
                continue
            survey_df, base_df = separate_base_and_survey(df_raw, sheet)
            if survey_df.empty:
                st.warning(f"Sheet {sheet}: Tidak ada data survei. Dilewati.")
                continue
            if field_method != "None":
                survey_df['Field_filtered'] = apply_filter(survey_df['Field'], field_method, **field_params)
            else:
                survey_df['Field_filtered'] = survey_df['Field']
            if alt_method != "None" and survey_df['Altitude'].notna().any():
                survey_df['Altitude_filtered'] = apply_filter(survey_df['Altitude'], alt_method, **alt_params)
            else:
                survey_df['Altitude_filtered'] = survey_df['Altitude']
            if not base_df.empty:
                diurnal_corr = compute_diurnal_correction(survey_df, base_df, reference_method='first')
                survey_df['Diurnal_Correction'] = diurnal_corr
            else:
                survey_df['Diurnal_Correction'] = 0.0
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
                        survey_df['IGRF'] = 0.0
            else:
                survey_df['IGRF'] = 0.0
            survey_df['IGRF'] = survey_df['IGRF'].fillna(0.0)
            survey_df['TMI'] = survey_df['Field_filtered'] - survey_df['IGRF'] - survey_df['Diurnal_Correction']
            survey_df['Sheet_Name'] = sheet
            all_results.append(survey_df)
            progress_bar.progress((idx+1)/len(sheet_names))
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            st.session_state['final_df'] = final_df
            st.success(f"✅ Selesai! Total {len(final_df)} titik dari {len(all_results)} sheet.")
        else:
            st.error("Tidak ada data yang diproses.")
    
    if 'final_df' in st.session_state:
        final_df = st.session_state['final_df']
        sheets_present = final_df['Sheet_Name'].unique()
        st.subheader("📊 Hasil gabungan")
        st.dataframe(final_df[['Sheet_Name', 'datetime', 'Field', 'Field_filtered', 'TMI']].head(10))
        
        selected_sheets = st.multiselect("Pilih sheet untuk ditampilkan", sheets_present, default=sheets_present)
        plot_df = final_df[final_df['Sheet_Name'].isin(selected_sheets)].copy()
        if not plot_df.empty:
            # Plot perbandingan Field
            st.header("📈 Perbandingan Field Original vs Filtered")
            fig_field, ax_field = plt.subplots(figsize=(12, 4))
            for sheet in selected_sheets:
                df_sheet = plot_df[plot_df['Sheet_Name'] == sheet].sort_values('datetime')
                ax_field.plot(df_sheet['Field'].values, '--', alpha=0.5, label=f'{sheet} Original')
                ax_field.plot(df_sheet['Field_filtered'].values, '-', alpha=0.8, label=f'{sheet} Filtered')
            ax_field.set_xlabel('Index')
            ax_field.set_ylabel('nT')
            ax_field.legend(loc='best', ncol=2)
            ax_field.grid(True, alpha=0.3)
            st.pyplot(fig_field)
            plt.close(fig_field)
            
            # Plot TMI
            # ========== 2. PLOT TMI per sheet dengan sumbu x waktu ==========
            st.header("📉 Total Magnetic Intensity (TMI) setelah koreksi (per sheet)")
            for sheet in selected_sheets:
                df_sheet = plot_df[plot_df['Sheet_Name'] == sheet].sort_values('datetime')
                if not df_sheet.empty:
                    fig_tmi, ax_tmi = plt.subplots(figsize=(12, 5))
                    ax_tmi.plot(df_sheet['datetime'], df_sheet['TMI'], 'b-', linewidth=1, label=sheet)
                    ax_tmi.set_xlabel('Waktu (UTC)')
                    ax_tmi.set_ylabel('TMI (nT)')
                    ax_tmi.set_title(f'TMI - Sheet {sheet}')
                    ax_tmi.legend()
                    ax_tmi.grid(True, linestyle=':', alpha=0.5)
                    # Rotasi label agar tidak bertumpuk
                    plt.xticks(rotation=45)
                    st.pyplot(fig_tmi)
                    plt.close(fig_tmi)
                else:
                    st.info(f"Sheet {sheet}: Tidak ada data TMI untuk diplot.")
            
            # ========== GRIDDING & PETA ANOMALI ==========
            st.header(f"🗺️ Peta Anomali {anomaly_type} dengan Gridding ({gridding_method})")
            # Ambil semua data koordinat dan anomali (tanpa NaN)
            grid_df = plot_df.dropna(subset=['Longitude', 'Latitude', anomaly_type]).copy()
            if len(grid_df) < 4:
                st.warning("Tidak cukup titik untuk membuat grid (minimal 4 titik).")
            else:
                x = grid_df['Longitude'].values
                y = grid_df['Latitude'].values
                z = grid_df[anomaly_type].values
                
                # Buat grid sesuai pilihan
                if gridding_method == "Tanpa Grid (scatter)":
                    fig_anom, ax_anom = plt.subplots(figsize=(10, 8))
                    sc = ax_anom.scatter(x, y, c=z, s=10, cmap='jet', norm=Normalize(vmin=z.min(), vmax=z.max()))
                    plt.colorbar(sc, ax=ax_anom, label=f'{anomaly_type} (nT)')
                    if show_track_lines:
                        for sheet in selected_sheets:
                            line_df = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=['Longitude', 'Latitude']).sort_values('datetime')
                            ax_anom.plot(line_df['Longitude'], line_df['Latitude'], 'k-', linewidth=1, alpha=0.7)
                    ax_anom.set_xlabel('Longitude')
                    ax_anom.set_ylabel('Latitude')
                    ax_anom.set_title(f'Scatter plot {anomaly_type} (tanpa grid)')
                    ax_anom.grid(True, alpha=0.3)
                    st.pyplot(fig_anom)
                    plt.close(fig_anom)
                else:
                    # Pilih method untuk griddata
                    if gridding_method == "Linear":
                        grid_meth = 'linear'
                    elif gridding_method == "Cubic":
                        grid_meth = 'cubic'
                    elif gridding_method == "RBF (Thin Plate Spline)":
                        grid_meth = 'rbf'
                    else:
                        grid_meth = 'linear'
                    
                    try:
                        X, Y, Z_grid = gridded_anomaly_map(x, y, z, method=grid_meth, grid_resolution=grid_resolution)
                        fig_anom, ax_anom = plt.subplots(figsize=(10, 8))
                        # Plot grid sebagai kontur atau pcolormesh
                        cf = ax_anom.contourf(X, Y, Z_grid, levels=20, cmap='jet', alpha=0.8)
                        plt.colorbar(cf, ax=ax_anom, label=f'{anomaly_type} (nT)')
                        # Overlay lintasan hitam
                        if show_track_lines:
                            for sheet in selected_sheets:
                                line_df = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=['Longitude', 'Latitude']).sort_values('datetime')
                                ax_anom.plot(line_df['Longitude'], line_df['Latitude'], 'k-', linewidth=1.5, alpha=0.8, label=sheet if len(selected_sheets)==1 else None)
                            if len(selected_sheets) > 1:
                                ax_anom.legend(fontsize=8)
                        ax_anom.set_xlabel('Longitude')
                        ax_anom.set_ylabel('Latitude')
                        ax_anom.set_title(f'Gridded {anomaly_type} ({gridding_method}) dengan lintasan hitam')
                        ax_anom.grid(True, alpha=0.3)
                        st.pyplot(fig_anom)
                        plt.close(fig_anom)
                    except Exception as e:
                        st.error(f"Gagal membuat grid: {e}")
            
            # ========== PROFIL JARAK ==========
            st.header("📏 Profil Anomali Sepanjang Jarak")
            for sheet in selected_sheets:
                prof_df = plot_df[plot_df['Sheet_Name'] == sheet].dropna(subset=[anomaly_type, 'Longitude', 'Latitude']).sort_values('datetime')
                if len(prof_df) > 1:
                    dist = compute_distance_along_line(prof_df)
                    fig_prof, ax_prof = plt.subplots(figsize=(10, 4))
                    ax_prof.plot(dist/1000, prof_df[anomaly_type], 'b-', linewidth=1, marker='.', markersize=2)
                    ax_prof.set_xlabel('Jarak (km)')
                    ax_prof.set_ylabel(f'{anomaly_type} (nT)')
                    ax_prof.set_title(f'Sheet {sheet}')
                    ax_prof.grid(True, alpha=0.3)
                    st.pyplot(fig_prof)
                    plt.close(fig_prof)
            
            # ========== DOWNLOAD ==========
            st.header("💾 Download Data")
            output_cols = ['Sheet_Name', 'datetime', 'Latitude', 'Longitude', 'Easting', 'Northing',
                           'Field', 'Field_filtered', 'Altitude', 'Altitude_filtered', 'Depth', 'Line_Name',
                           'IGRF', 'Diurnal_Correction', 'TMI']
            output_cols = [c for c in output_cols if c in final_df.columns]
            output_df = final_df[output_cols]
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "marine_magnetic_gridded.csv", "text/csv")
else:
    st.info("⬅️ Upload file Excel atau CSV.")
