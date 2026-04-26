import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline, griddata, RBFInterpolator
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
        sheets = {}
        xl = pd.ExcelFile(uploaded_file)
        for sheet in xl.sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sheet)
            all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                        'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
            df = clean_string_placeholders(df, all_cols)
            num_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
            df = clean_numeric_columns(df, num_cols)
            sheets[sheet] = df
        return sheets
    else:
        df = pd.read_csv(uploaded_file)
        all_cols = ['Reading_Date', 'Reading_Time', 'Latitude', 'Longitude', 'Easting', 'Northing',
                    'Field', 'Altitude', 'Depth', 'Fbase', 'Tbase']
        df = clean_string_placeholders(df, all_cols)
        num_cols = ['Latitude', 'Longitude', 'Easting', 'Northing', 'Field', 'Altitude', 'Depth', 'Fbase']
        df = clean_numeric_columns(df, num_cols)
        return {'data': df}

def parse_datetime(df, sheet_name):
    for col in ['Reading_Date', 'Reading_Time']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace(['nan', 'NaN', '*', ''], np.nan)
    df_clean = df.dropna(subset=['Reading_Date', 'Reading_Time']).copy()
    if len(df_clean) == 0:
        raise ValueError(f"Sheet '{sheet_name}': Tidak ada baris dengan tanggal & waktu valid.")
    dt_str = df_clean['Reading_Date'].astype(str) + ' ' + df_clean['Reading_Time'].astype(str)
    try:
        dt = pd.to_datetime(dt_str, utc=True, format='%Y-%m-%d %H:%M:%S', errors='raise')
    except:
        try:
            dt = pd.to_datetime(dt_str, utc=True, format='%Y-%m-%d %H:%M:%S.%f', errors='raise')
        except:
            try:
                dt = pd.to_datetime(dt_str, utc=True, format='mixed')
            except:
                dt = pd.to_datetime(dt_str, utc=True, errors='coerce')
    if dt.isna().any():
        n_invalid = dt.isna().sum()
        raise ValueError(f"Sheet '{sheet_name}': {n_invalid} baris gagal di-parse datetime.")
    df_clean['datetime'] = dt
    return df_clean

def separate_base_and_survey(df, sheet_name):
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
        base_df = base_df.dropna(subset=['base_datetime'])
    return survey_df, base_df

def hampel_filter(series, window=5, n_sigma=3.0):
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    mad = series.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: median_abs_deviation(x, nan_policy='omit'), raw=True)
    mad = mad.fillna(mad.median())
    diff = np.abs(series - rolling_median)
    outlier = diff > (n_sigma * mad)
    cleaned = series.copy()
    cleaned[outlier] = np.nan
    return cleaned, outlier

def interpolate_nan(series, method='cubic'):
    idx = series.index
    valid = ~np.isnan(series.values)
    if method == 'cubic' and np.sum(valid) > 3:
        cs = CubicSpline(idx[valid], series.values[valid])
        return pd.Series(cs(idx), index=idx)
    else:
        return series.interpolate(method='linear', limit_direction='both')

def moving_average(series, window=5):
    return series.rolling(window=window, center=True, min_periods=1).mean()

def butterworth_filter(series, cutoff=0.1, fs=1.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if series.isna().any():
        series = series.fillna(series.median())
    return filtfilt(b, a, series)

def apply_filter(series, method, **params):
    if method == 'Hampel (despiking)':
        cleaned, _ = hampel_filter(series, window=params.get('window',5), n_sigma=params.get('threshold',3.0))
        return interpolate_nan(cleaned, 'cubic')
    elif method == 'Moving Average':
        return moving_average(series, window=params.get('window',5))
    elif method == 'Savitzky-Golay':
        window = params.get('window',11)
        if window%2==0: window+=1
        temp = series.interpolate(method='linear', limit_direction='both')
        return pd.Series(savgol_filter(temp, window, polyorder=3), index=series.index)
    elif method == 'Butterworth Lowpass':
        return butterworth_filter(series, cutoff=params.get('cutoff',0.1))
    else:
        return series.copy()

def compute_diurnal_correction(survey_df, base_df):
    if base_df.empty:
        return np.zeros(len(survey_df))
    base_df = base_df.dropna(subset=['base_datetime']).sort_values('base_datetime')
    if len(base_df) < 2 or survey_df.empty:
        return np.zeros(len(survey_df))
    base_sec = base_df['base_datetime'].astype('int64') // 10**9
    survey_sec = survey_df['datetime'].astype('int64') // 10**9
    interp_vals = np.interp(survey_sec, base_sec, base_df['Fbase'].values)
    ref = base_df['Fbase'].iloc[0]  # first base value
    return interp_vals - ref

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2-lat1)
    dlam = radians(lon2-lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def compute_distance_along_line(df):
    dist = [0.0]
    for i in range(1, len(df)):
        d = haversine_distance(df.iloc[i-1]['Longitude'], df.iloc[i-1]['Latitude'],
                               df.iloc[i]['Longitude'], df.iloc[i]['Latitude'])
        dist.append(dist[-1] + d)
    return np.array(dist)

def gridded_anomaly_map(x, y, z, method='cubic', resolution=80):
    # method: 'linear', 'cubic', 'rbf'
    xi = np.linspace(x.min()-0.002, x.max()+0.002, resolution)
    yi = np.linspace(y.min()-0.002, y.max()+0.002, resolution)
    X, Y = np.meshgrid(xi, yi)
    if method == 'rbf':
        rbf = RBFInterpolator(np.column_stack((x, y)), z, kernel='thin_plate_spline', smoothing=0.0)
        Z = rbf(np.column_stack((X.ravel(), Y.ravel()))).reshape(X.shape)
    else:
        Z = griddata((x, y), z, (X, Y), method=method)
    return X, Y, Z

# ================== MAIN APP ==================

st.set_page_config(page_title="Marine Magnetic Full Processing", layout="wide")
st.title("🌊 Pengolahan Data Magnetik Kelautan - Gridding Anomali Penuh")

uploaded = st.sidebar.file_uploader("Upload file Excel/CSV", type=['xlsx','csv'])

if uploaded is not None:
    all_sheets = load_data(uploaded)
    sheet_names = list(all_sheets.keys())
    st.info(f"Sheet terdeteksi: {', '.join(sheet_names)}")
    
    # Sidebar filter parameters
    st.sidebar.header("🔧 Filter Parameters")
    field_method = st.sidebar.selectbox("Filter Field", ["None","Hampel (despiking)","Moving Average","Savitzky-Golay","Butterworth Lowpass"])
    field_win = st.sidebar.slider("Field window", 3, 51, 11, 2) if field_method!="None" else 11
    field_thresh = st.sidebar.slider("Hampel sigma", 1.0, 5.0, 3.0, 0.5) if field_method=="Hampel (despiking)" else 3.0
    alt_method = st.sidebar.selectbox("Filter Altitude", ["None","Hampel (despiking)","Moving Average","Savitzky-Golay"])
    alt_win = st.sidebar.slider("Altitude window", 3, 51, 11, 2) if alt_method!="None" else 11
    
    # IGRF manual
    st.sidebar.header("🧲 IGRF Source")
    igrf_opt = st.sidebar.radio("IGRF", ["Constant","Skip (0)","Upload file"])
    const_igrf = st.sidebar.number_input("Constant IGRF (nT)", value=45000.0, step=100.0) if igrf_opt=="Constant" else 0.0
    igrf_file = st.sidebar.file_uploader("Upload IGRF CSV", type=['csv']) if igrf_opt=="Upload file" else None
    
    anomaly_var = st.sidebar.selectbox("Anomaly for map", ["TMI","Field_filtered"])
    grid_method = st.sidebar.selectbox("Gridding method", ["Cubic","Linear","RBF"])
    grid_res = st.sidebar.slider("Grid resolution", 50, 200, 100)
    show_track = st.sidebar.checkbox("Show track lines (black)", True)
    
    if st.button("🚀 Process All Sheets"):
        all_dfs = []
        progress = st.progress(0)
        for i, sheet in enumerate(sheet_names):
            st.write(f"Processing: {sheet}")
            df = all_sheets[sheet].copy()
            try:
                df = parse_datetime(df, sheet)
            except Exception as e:
                st.error(f"{sheet}: {e}")
                continue
            surv, base = separate_base_and_survey(df, sheet)
            if surv.empty:
                st.warning(f"{sheet}: No Field data, skip.")
                continue
            # filter Field
            if field_method != "None":
                surv['Field_filtered'] = apply_filter(surv['Field'], field_method, window=field_win, threshold=field_thresh)
            else:
                surv['Field_filtered'] = surv['Field']
            # filter Altitude
            if alt_method != "None" and surv['Altitude'].notna().any():
                surv['Altitude_filtered'] = apply_filter(surv['Altitude'], alt_method, window=alt_win)
            else:
                surv['Altitude_filtered'] = surv['Altitude']
            # diurnal correction (using its own base)
            surv['Diurnal_Corr'] = compute_diurnal_correction(surv, base)
            # IGRF
            if igrf_opt == "Constant":
                surv['IGRF'] = const_igrf
            elif igrf_opt == "Upload file" and igrf_file is not None:
                igrf_df = pd.read_csv(igrf_file)
                if 'datetime' in igrf_df.columns:
                    igrf_df['datetime'] = pd.to_datetime(igrf_df['datetime'], utc=True)
                    surv = surv.merge(igrf_df[['datetime','IGRF']], on='datetime', how='left')
                    surv['IGRF'] = surv['IGRF'].fillna(0.0)
                else:
                    surv['IGRF'] = 0.0
            else:
                surv['IGRF'] = 0.0
            surv['TMI'] = surv['Field_filtered'] - surv['IGRF'] - surv['Diurnal_Corr']
            surv['Sheet_Name'] = sheet
            all_dfs.append(surv)
            progress.progress((i+1)/len(sheet_names))
        
        if all_dfs:
            final = pd.concat(all_dfs, ignore_index=True)
            st.session_state['final'] = final
            st.success(f"✅ Selesai! Total {len(final)} titik dari {len(all_dfs)} sheets.")
        else:
            st.error("No data processed.")
    
    if 'final' in st.session_state:
        final = st.session_state['final']
        sheets = final['Sheet_Name'].unique()
        
        st.subheader("Preview hasil")
        st.dataframe(final[['Sheet_Name','datetime','Field','Field_filtered','TMI']].head(10))
        
        # Plot Field comparison
        st.header("📈 Field Original vs Filtered")
        fig1, ax1 = plt.subplots(figsize=(12,4))
        for s in sheets:
            d = final[final['Sheet_Name']==s].sort_values('datetime')
            ax1.plot(d['Field'].values, '--', alpha=0.5, label=f'{s} original')
            ax1.plot(d['Field_filtered'].values, '-', alpha=0.8, label=f'{s} filtered')
        ax1.legend(loc='best', ncol=2)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)
        
        # Plot TMI time series
        st.header("📉 TMI after correction")
        fig2, ax2 = plt.subplots(figsize=(12,4))
        for s in sheets:
            d = final[final['Sheet_Name']==s].sort_values('datetime')
            ax2.plot(d['TMI'].values, label=s)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
        
        # =============== GRIDDED ANOMALY MAP (FULL AREA) ===============
        st.header(f"🗺️ Anomali {anomaly_var} - Interpolasi Seluruh Area (Gridding)")
        # Ambil data koordinat dan nilai anomali (hilangkan NaN)
        grid_data = final.dropna(subset=['Longitude','Latitude', anomaly_var]).copy()
        if len(grid_data) < 4:
            st.warning("Tidak cukup titik untuk gridding (min 4).")
        else:
            x = grid_data['Longitude'].values
            y = grid_data['Latitude'].values
            z = grid_data[anomaly_var].values
            
            # Pilih metode
            meth = grid_method.lower()
            if meth == 'rbf':
                meth_opt = 'rbf'
            elif meth == 'cubic':
                meth_opt = 'cubic'
            else:
                meth_opt = 'linear'
            
            try:
                Xg, Yg, Zg = gridded_anomaly_map(x, y, z, method=meth_opt, resolution=grid_res)
                
                fig3, ax3 = plt.subplots(figsize=(12, 10))
                cf = ax3.contourf(Xg, Yg, Zg, levels=50, cmap='viridis', alpha=0.85)
                cbar = plt.colorbar(cf, ax=ax3, label=f'{anomaly_var} (nT)')
                if show_track:
                    for s in sheets:
                        track = final[final['Sheet_Name']==s].dropna(subset=['Longitude','Latitude']).sort_values('datetime')
                        if len(track) > 1:
                            ax3.plot(track['Longitude'], track['Latitude'], 'k-', linewidth=1.5, alpha=0.8, label=s if len(sheets)<=5 else "")
                    if len(sheets) <= 5:
                        ax3.legend(fontsize=8)
                ax3.set_xlabel('Longitude')
                ax3.set_ylabel('Latitude')
                ax3.set_title(f'Gridded {anomaly_var} ({grid_method}) - Full coverage + track lines')
                ax3.grid(True, alpha=0.2)
                st.pyplot(fig3)
                plt.close(fig3)
            except Exception as e:
                st.error(f"Gridding error: {e}")
        
        # =============== PROFIL JARAK ===============
        st.header("📏 Profil Anomali Sepanjang Lintasan")
        for s in sheets:
            prof = final[final['Sheet_Name']==s].dropna(subset=['Longitude','Latitude',anomaly_var]).sort_values('datetime')
            if len(prof) > 1:
                dist = compute_distance_along_line(prof)
                fig4, ax4 = plt.subplots(figsize=(10,4))
                ax4.plot(dist/1000, prof[anomaly_var], 'b-', marker='.', markersize=2, linewidth=1)
                ax4.set_xlabel('Jarak (km)')
                ax4.set_ylabel(f'{anomaly_var} (nT)')
                ax4.set_title(f'Sheet {s}')
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.info(f"Sheet {s}: kurang titik untuk profil.")
        
        # Download final data
        st.header("💾 Download")
        out_cols = ['Sheet_Name','datetime','Longitude','Latitude','Easting','Northing','Field','Field_filtered',
                    'Altitude','Altitude_filtered','Depth','Line_Name','IGRF','Diurnal_Corr','TMI']
        out_cols = [c for c in out_cols if c in final.columns]
        csv = final[out_cols].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "marine_magnetic_processed.csv", "text/csv")

else:
    st.info("⬅️ Upload file Excel (multiple sheets) atau CSV.")
