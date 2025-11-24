import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import re
import unicodedata
import datetime as dt
import io

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn import metrics

# =========================
# 0. CẤU HÌNH FILE LƯU BẤT THƯỜNG MỚI
# =========================
ANOMALY_NEW_FILE = "anomalies_new.xlsx"


def load_new_anomalies():
    """
    Load file Excel lưu các tin đăng bất thường mới.
    Nếu chưa có file thì trả về DataFrame rỗng.
    """
    try:
        df_new = pd.read_excel(ANOMALY_NEW_FILE)
        if 'thoi_gian_dang' in df_new.columns:
            df_new['thoi_gian_dang'] = pd.to_datetime(df_new['thoi_gian_dang'],
                                                      errors='coerce')
        return df_new
    except Exception:
        return pd.DataFrame()


def append_new_anomaly(record: dict):
    """
    Thêm 1 bản ghi bất thường mới vào file Excel.
    """
    df_existing = load_new_anomalies()
    df_new_row = pd.DataFrame([record])
    df_all = pd.concat([df_existing, df_new_row], ignore_index=True)
    df_all.to_excel(ANOMALY_NEW_FILE, index=False)


# =========================
# 1. CẤU HÌNH CHUNG
# =========================
st.set_page_config(
    page_title="Dự đoán giá xe máy cũ và phát hiện bất thường",
    layout="centered"
)

# ==== CSS ====
st.markdown(
    """
    <style>
    /* Nền tổng thể */
    .stApp {
        background: linear-gradient(135deg, #fdfbff 0%, #f5f7ff 50%, #fff7f5 100%);
    }

    /* Khối nội dung trung tâm */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* Sidebar*/
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fef6ff 0%, #e0f2fe 50%, #fdf2f8 100%);
    }

    [data-testid="stSidebar"] * {
        font-size: 0.95rem;
    }

    /* Tiêu đề menu sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #374151;
        font-weight: 700;
    }

    /* Radio button trong sidebar */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 999px;
        padding: 4px 10px;
        margin-bottom: 4px;
    }

    /* Nút bấm*/
    .stButton>button {
        background: linear-gradient(90deg, #a5b4fc, #f9a8d4);
        color: #1f2933;
        border-radius: 999px;
        padding: 0.5rem 1.6rem;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 10px rgba(148, 163, 233, 0.4);
        transition: all 0.15s ease-in-out;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 14px rgba(244, 114, 182, 0.5);
        transform: translateY(-1px);
        filter: brightness(1.03);
    }

    .stButton>button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 2px 6px rgba(148, 163, 233, 0.4);
    }

    /* Dataframe card */
    .dataframe tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }

    /* Nhỏ lại font bảng một chút cho gọn */
    .stDataFrame, .stDataFrame table {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Banner
try:
    img = Image.open("Banner.PNG")
    img = img.resize((img.width, 350))
    st.image(img, use_container_width=True)
except Exception as e:
    st.warning(f"Không thể load Banner.png: {e}")

# ===== TIÊU ĐỀ CHÍNH: tô màu + canh giữa =====
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #e0f2fe 0%, #f5d0fe 50%, #fee2e2 100%);
        padding: 18px 25px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(148, 163, 233, 0.4);
    ">
        <h1 style="color:#111827; margin:0; font-size: 1.9rem; text-align:center;">
            Dự đoán giá xe máy cũ và phát hiện bất thường
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Hàm tạo header
def pastel_header(icon: str, text: str, color: str = "#e0f2fe"):
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            border-radius: 12px;
            padding: 10px 14px;
            margin: 18px 0 10px 0;
            border: 1px solid rgba(148, 163, 233, 0.5);
        ">
            <h3 style="margin:0; color:#111827; font-weight:650; font-size:1.1rem;">
                {icon} {text}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU
# =========================
def preprocessing_data(df, is_train=True):
    df = df.copy()
    # làm sạch tên cột: bỏ dấu, ký tự đặc biệt -> dạng snake_case
    d = {ord('đ'): 'd', ord('Đ'): 'D'}

    def clean_col(name: str) -> str:
        s = unicodedata.normalize('NFKD', str(name)).translate(d)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r'\W+', '_', s.lower()).strip('_')

    df.columns = [clean_col(c) for c in df.columns]

    # Xóa trùng href nếu có
    if 'href' in df.columns:
        df = df.drop_duplicates(subset='href', keep='first')

    # Chuẩn hóa cột giá nếu có
    if 'gia' in df.columns:
        def clean_price(value):
            if pd.isna(value):
                return np.nan
            text = str(value).lower().strip()
            text = text.replace(',', '.').replace(' ', '')
            # Nếu có 'đ' hoặc 'vnd', chia 1_000_000
            if 'đ' in text or 'vnd' in text:
                num = re.sub(r'[^0-9]', '', text)
                return float(num) / 1_000_000 if num else np.nan
            try:
                return float(text)
            except Exception:
                return np.nan

        df['gia'] = df['gia'].apply(clean_price)

    # Chuẩn hóa khoảng giá nếu có
    for col in ['khoang_gia_min', 'khoang_gia_max']:
        if col in df.columns:
            def clean_price_2(value):
                if pd.isna(value):
                    return np.nan
                text = str(value).lower().strip()
                text = text.replace(',', '.').replace(' ', '')
                num = re.sub(r'[^0-9\.]', '', text)
                if num == '':
                    return np.nan
                try:
                    return float(num)
                except Exception:
                    return np.nan
            df[col] = df[col].apply(clean_price_2)

    # Tạo feature tuoi_xe
    if 'nam_dang_ky' in df.columns:
        df['nam_dang_ky'] = df['nam_dang_ky'].replace('trước năm 1980', '1979')
        current_year = dt.date.today().year
        df['tuoi_xe'] = (current_year - pd.to_numeric(df['nam_dang_ky'], errors='coerce')).clip(lower=0)

    # Chuyển kiểu dữ liệu
    if 'so_km_da_di' in df.columns:
        df['so_km_da_di'] = pd.to_numeric(df['so_km_da_di'], errors='coerce')

    # Drop các cột không cần thiết
    drop_cols = [
        'id', 'tieu_de', 'dia_chi', 'mo_ta_chi_tiet',
        'href', 'trong_luong', 'chinh_sach_bao_hanh',
        'tinh_trang', 'nam_dang_ky'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Xử lý missing values sơ bộ
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'gia' in num_cols:
        num_cols.remove('gia')
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    # Nếu là train và có cột giá thì drop NA
    if is_train and 'gia' in df.columns:
        df = df.dropna(subset=['gia']).reset_index(drop=True)

    # Chuẩn hóa 1 số category
    if 'dung_tich_xe' in df.columns:
        df['dung_tich_xe'] = df['dung_tich_xe'].replace({
            'Không biết rõ': 'Khac',
            'Đang cập nhật': 'Khac',
            'Nhật Bản': 'Khac'
        })
    if 'xuat_xu' in df.columns:
        df['xuat_xu'] = df['xuat_xu'].replace('Bảo hành hãng', 'Dang cap nhat')

    if 'thuong_hieu' in df.columns and is_train:
        threshold = 10
        popular = df['thuong_hieu'].value_counts()
        popular = popular[popular >= threshold].index
        df['thuong_hieu'] = df['thuong_hieu'].apply(
            lambda x: x if x in popular else 'Hang khac'
        )

    if 'dong_xe' in df.columns and is_train:
        threshold = 10
        popular = df['dong_xe'].value_counts()
        popular = popular[popular >= threshold].index
        df['dong_xe'] = df['dong_xe'].apply(
            lambda x: x if x in popular else 'Khac'
        )

    # Phân khúc theo thương hiệu + loại bỏ outlier theo phân khúc
    if 'gia' in df.columns and 'thuong_hieu' in df.columns and is_train:
        if df.empty or df['thuong_hieu'].nunique() == 0:
            df['phan_khuc'] = np.nan
        else:
            brand_mean = df.groupby('thuong_hieu', as_index=False)['gia'].mean().rename(
                columns={'gia': 'mean_price'}
            )
            if brand_mean.empty:
                df['phan_khuc'] = np.nan
            else:
                brand_mean['phan_khuc'] = pd.cut(
                    brand_mean['mean_price'],
                    bins=[-float('inf'), 50, 100, float('inf')],
                    labels=['pho_thong', 'trung_cap', 'cao_cap'],
                    right=False
                )
                df = df.merge(
                    brand_mean[['thuong_hieu', 'phan_khuc']],
                    on='thuong_hieu',
                    how='left'
                )
                df['phan_khuc'] = df['phan_khuc'].astype('object')

        # Loại outlier theo IQR trong từng phân khúc
        def remove_outliers_by_brand(df_local, column,
                                     lower_percentile=0.25,
                                     upper_percentile=0.75,
                                     threshold=1.5):
            if column not in df_local.columns:
                return df_local

            def remove_group_outliers(group):
                Q1 = group[column].quantile(lower_percentile)
                Q3 = group[column].quantile(upper_percentile)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return group[(group[column] >= lower_bound) &
                             (group[column] <= upper_bound)]

            return df_local.groupby('phan_khuc', group_keys=False).apply(
                remove_group_outliers
            )

        remove_outlier_cols = [
            c for c in ['gia', 'so_km_da_di', 'tuoi_xe'] if c in df.columns
        ]
        for c in remove_outlier_cols:
            df = remove_outliers_by_brand(df, c)
        df = df.reset_index(drop=True)

    # SAU KHI LOẠI OUTLIER: xoá cột phan_khuc, KHÔNG đưa vào mô hình ML
    df = df.drop(columns=['phan_khuc'], errors='ignore')

    return df

# =========================
# 3. HÀM PHÁT HIỆN BẤT THƯỜNG
# =========================
def detect_anomalies(df, model, threshold=50, method='absolute'):
    """
    method:
        - 'absolute': dùng ngưỡng score tuyệt đối (>= threshold)
        - 'percentile': dùng phân vị score (threshold = 0–100, ví dụ 95 -> top 5%)
    """
    df = df.copy()

    # Dự đoán giá từ mô hình đã huấn luyện
    exclude_cols = ['gia', 'is_new']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df['gia_predict'] = model.predict(df[feature_cols])

    # Tính residual và z-score
    df['resid'] = df['gia'] - df['gia_predict']

    def compute_resid_z(df_local):
        if 'thuong_hieu' not in df_local.columns:
            global_mean = df_local['resid'].mean()
            global_std = df_local['resid'].std(ddof=0)
            if global_std > 0:
                df_local['resid_z'] = (df_local['resid'] - global_mean) / global_std
            else:
                df_local['resid_z'] = 0.0
            return df_local

        group_sizes = df_local['thuong_hieu'].value_counts()
        small_groups = group_sizes[group_sizes < 2].index

        df_local['resid_z'] = 0.0

        big_brands = group_sizes[group_sizes >= 2].index
        df_local.loc[df_local['thuong_hieu'].isin(big_brands), 'resid_z'] = \
            df_local.groupby('thuong_hieu')['resid'].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0)
                if x.std(ddof=0) > 0 else 0
            )

        global_mean = df_local['resid'].mean()
        global_std = df_local['resid'].std(ddof=0)
        if global_std > 0:
            mask = df_local['thuong_hieu'].isin(small_groups)
            df_local.loc[mask, 'resid_z'] = (
                df_local.loc[mask, 'resid'] - global_mean
            ) / global_std
        return df_local

    df = compute_resid_z(df)

    # Khoảng tin cậy dựa trên phân vị 10–90 của giá
    p10, p90 = np.percentile(df['gia'].dropna(), [10, 90])

    # Đảm bảo numeric
    for col in ['gia', 'khoang_gia_min', 'khoang_gia_max']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Vi phạm min/max nếu có khoảng giá
    if {'khoang_gia_min', 'khoang_gia_max'}.issubset(df.columns):
        df['vi_pham_minmax'] = (
            (df['gia'] < df['khoang_gia_min']) |
            (df['gia'] > df['khoang_gia_max'])
        ).astype(int)
    else:
        df['vi_pham_minmax'] = 0

    # Ngoài khoảng tin cậy
    df['ngoai_khoang_tin_cay'] = (
        (df['gia'] < p10) | (df['gia'] > p90)
    ).astype(int)

    # Isolation Forest trên một số feature numeric
    iso_features = [
        'gia', 'gia_predict', 'resid', 'resid_z',
        'so_km_da_di', 'tuoi_xe'
    ]
    iso_features = [c for c in iso_features if c in df.columns]

    if len(iso_features) > 0:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['iso_score'] = iso.fit_predict(df[iso_features])
        df['iso_score'] = df['iso_score'].apply(lambda x: 1 if x == -1 else 0)
    else:
        df['iso_score'] = 0

    # Tính điểm tổng hợp (0–100)
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    df['score'] = 100 * (
        (w1 * np.abs(df['resid_z']) +
         w2 * df['vi_pham_minmax'] +
         w3 * df['ngoai_khoang_tin_cay'] +
         w4 * df['iso_score'])
        / (w1 + w2 + w3 + w4)
    )

    # Áp dụng ngưỡng
    if method == 'percentile':
        perc = float(np.clip(threshold, 0, 100))
        threshold_value = np.percentile(df['score'], perc)
        df['is_anomaly'] = (df['score'] > threshold_value).astype(int)
    else:
        threshold_value = threshold
        df['is_anomaly'] = (df['score'] >= threshold_value).astype(int)

    df_result = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df_result, threshold_value

# =========================
# 4. LOAD DATA & TRAIN MODEL (MẶC ĐỊNH)
# =========================
@st.cache_data
def load_data(path="data_motobikes.xlsx"):
    df_raw = pd.read_excel(path)
    df_processed = preprocessing_data(df_raw, is_train=True)
    return df_raw, df_processed


@st.cache_resource
def train_rf_model(df_processed, n_estimators=200,
                   max_depth=None, random_state=42):
    df = df_processed.copy()
    if 'gia' not in df.columns:
        raise ValueError("Không tìm thấy cột 'gia' trong dữ liệu sau tiền xử lý")

    y = df['gia']
    X = df.drop(columns=['gia'])

    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Xác định numeric / categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # Dự đoán cho đánh giá
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_dict = {
        "train_R2": metrics.r2_score(y_train, y_train_pred),
        "test_R2": metrics.r2_score(y_test, y_test_pred),
        "train_RMSE": np.sqrt(
            metrics.mean_squared_error(y_train, y_train_pred)
        ),
        "test_RMSE": np.sqrt(
            metrics.mean_squared_error(y_test, y_test_pred)
        ),
        "train_MAE": metrics.mean_absolute_error(y_train, y_train_pred),
        "test_MAE": metrics.mean_absolute_error(y_test, y_test_pred),
    }

    return model, X_train, X_test, y_train, y_test, metrics_dict


# Thử load dữ liệu & train model mặc định
try:
    df_raw, df_processed = load_data()
except Exception as e:
    df_raw, df_processed = None, None
    st.error(f"Lỗi khi đọc data_motobikes.xlsx: {e}")

if df_processed is not None:
    try:
        (model_default,
         X_train_default,
         X_test_default,
         y_train_default,
         y_test_default,
         metrics_default) = train_rf_model(df_processed)
    except Exception as e:
        model_default = None
        st.error(f"Lỗi khi train mô hình mặc định: {e}")
else:
    model_default = None

# =========================
# 5. MENU CHÍNH (PHIÊN BẢN MỚI)
# =========================

menu_items = [
    "1. Mục tiêu dự án",
    "2. Đánh giá & báo cáo",
    "3. Dự đoán giá xe máy cũ",
    "4. Phát hiện bất thường - Người đăng tin",
    "5. Phát hiện bất thường – Admin",
    "6. Nhóm thực hiện"
]

choice = st.sidebar.radio("📂 Danh mục", menu_items)

# =========================
# 6. TỪNG MỤC MENU
# =========================

# ---------- 1. Business Problem ----------
if choice.startswith("1."):
    pastel_header("📌", "Mục tiêu dự án", "#fee2e2")

    st.markdown("""
    <style>
        .simple-text {
            font-family: Arial, sans-serif;
            font-size: 17px;
            line-height: 1.6;
            text-align: justify;
            margin-left: 18px;
            margin-right: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="simple-text">

    **Chợ Tốt** là nền tảng mua bán trực tuyến hàng đầu tại Việt Nam. Chợ Tốt cung cấp đa dạng các dòng sản phẩm: nhà cửa, xe ô tô, đồ điện tử đã qua sử dụng, vật nuôi, các dịch vụ gia đình và tuyển dụng.

    Thị trường xe máy cũ trên nền tảng Chợ Tốt rất phong phú về dòng xe, năm sản xuất, tình trạng sử dụng, giá cả,... khiến người mua khó khăn trong việc xác định giá hợp lý. Một vấn đề cần thiết nữa được đặt ra là phát hiện các tin đăng có dấu hiệu bất thường, như giá quá rẻ để thu hút người xem hoặc giá quá cao gây nhiễu thị trường.

    Dự án này tập trung vào phân khúc xe máy cũ với hai mục tiêu chính:

    **1. Dự đoán giá xe máy cũ**  
    Xây dựng mô hình học máy có khả năng dự đoán giá bán hợp lý của xe máy dựa trên các thông tin như hãng xe, dòng xe, loại xe, dung tích động cơ, xuất xứ, số km đã đi, tuổi xe và tình trạng sử dụng. Mục tiêu là hỗ trợ người mua và người bán đưa ra quyết định chính xác và nhanh chóng hơn.

    **2. Phát hiện các tin đăng bất thường**  
    Sử dụng các thuật toán phát hiện bất thường (anomaly detection) để nhận diện các tin đăng có giá quá khác biệt so với thị trường. Điều này giúp nền tảng:

    - Giảm thiểu rủi ro cho người mua (lừa đảo, thông tin sai lệch).
    - Nâng cao chất lượng dữ liệu và độ tin cậy của trang.
    - Hỗ trợ đội ngũ kiểm duyệt phát hiện sớm các trường hợp đáng nghi.

    Thông qua việc kết hợp mô hình dự đoán giá và hệ thống cảnh báo tin đăng bất thường, dự án mang lại giá trị thiết thực cho cả người dùng và nền tảng nhằm xây dựng thị trường mua bán xe máy cũ hiệu quả và đáng tin cậy trên chotot.com.

    </div>
    """, unsafe_allow_html=True)

# ---------- 2. Evaluation & Report ----------
elif choice.startswith("2."):
    pastel_header("📊", "Đánh giá & báo cáo", "#e0f2fe")

    # Hiển thị thông tin dữ liệu gốc
    if df_raw is not None:
        st.markdown("##### 🧾 Dữ liệu gốc")
        st.write(
            f"Số hàng: {df_raw.shape[0]}, "
            f"số cột: {df_raw.shape[1]}"
        )
        st.dataframe(df_raw.head())
    else:
        st.warning(
            "Unable to read the file data_motobikes.xlsx – "
            "please check the file path and file name."
        )

    # Kiểm tra dữ liệu & model mặc định
    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình. Vui lòng kiểm tra lại.")
    else:
        # ===== Kết quả xây dựng và lựa chọn mô hình (Select model.PNG) =====
        st.markdown("##### 📈 Kết quả của xây dựng và lựa chọn mô hình")
        try:
            img_select = Image.open("Select model.PNG")
            st.image(
                img_select,
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Select model.PNG': {e}")
        st.markdown("**Mô hình phù hợp nhất là Random Forest**")

        # ===== Visualization nằm CUỐI CÙNG =====
        st.markdown(
            "##### 📉 Trực quan hóa kết quả thực hiện"
        )

        # Hình 1: Price.PNG – Comparison of Actual Price and Predicted Price
        try:
            img_price = Image.open("Price.PNG")
            st.image(
                img_price,
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Price.PNG': {e}")
        
        # Hình 2: Anomaly_scores.PNG – Distribution of Anomaly Scores
        try:
            img_scores = Image.open("Anomaly_scores.PNG")
            st.image(
                img_scores,
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Anomaly_scores.PNG': {e}")

# ---------- 3. Predicting Used Motorbike Prices ----------
elif choice.startswith("3."):
    pastel_header("💰", "Dự đoán giá xe máy cũ", "#fef3c7")

    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình.")
    else:
        model_use = st.session_state.get("model_custom", model_default)
        df = df_processed.copy()

        # Lấy danh sách option từ dữ liệu đã xử lý
        def get_unique(col):
            return sorted(
                df[col].dropna().unique().tolist()
            ) if col in df.columns else []

        col1, col2 = st.columns(2)
        with col1:
            thuong_hieu = st.selectbox(
                "Thương hiệu (thuong_hieu)", get_unique('thuong_hieu')
            )
            dong_xe = st.selectbox(
                "Dòng xe (dong_xe)", get_unique('dong_xe')
            )
            loai_xe = st.selectbox(
                "Loại xe (loai_xe)",
                get_unique('loai_xe') if 'loai_xe' in df.columns else []
            )
            xuat_xu = st.selectbox(
                "Xuất xứ (xuat_xu)",
                get_unique('xuat_xu') if 'xuat_xu' in df.columns else []
            )
        with col2:
            dung_tich = st.selectbox(
                "Dung tích xe (dung_tich_xe)",
                get_unique('dung_tich_xe') if 'dung_tich_xe' in df.columns else []
            )
            tuoi_xe = st.slider("Tuổi xe (năm)", 0, 30, 5)
            so_km_da_di = st.number_input(
                "Số km đã đi (so_km_da_di)",
                min_value=0, value=30000, step=1000
            )

        # Chuẩn bị 1 dòng input theo các cột X đã dùng khi train
        sample = {}
        X_cols = df.drop(columns=['gia']).columns.tolist()

        for c in X_cols:
            if c == 'thuong_hieu':
                sample[c] = thuong_hieu
            elif c == 'dong_xe':
                sample[c] = dong_xe
            elif c == 'loai_xe':
                sample[c] = loai_xe
            elif c == 'xuat_xu':
                sample[c] = xuat_xu
            elif c == 'dung_tich_xe':
                sample[c] = dung_tich
            elif c == 'tuoi_xe':
                sample[c] = tuoi_xe
            elif c == 'so_km_da_di':
                sample[c] = so_km_da_di
            else:
                # với các cột khác, để NaN cho pipeline xử lý
                sample[c] = np.nan

        input_df = pd.DataFrame([sample])

        st.markdown("##### 📥 Thông tin xe")
        st.dataframe(input_df.drop(columns=['phan_khuc', 'khoang_gia_min', 'khoang_gia_max'],
                                   errors='ignore'))

        if st.button("Giá dự đoán"):
            try:
                y_pred = model_use.predict(input_df)[0]
                st.success(f"Giá dự đoán: {y_pred:.2f} triệu VND")
            except Exception as e:
                st.error(f"Lỗi khi gọi model.predict: {e}")

# ---------- 4. PHÁT HIỆN BẤT THƯỜNG - NGƯỜI ĐĂNG TIN ----------
elif choice.startswith("4."):
    pastel_header("🚨", "Phát hiện bất thường - Người đăng tin", "#ede9fe")

    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình.")
    else:
        # ưu tiên model_anom nếu có, sau đó model_custom, cuối cùng model_default
        model_use = st.session_state.get(
            "model_anom",
            st.session_state.get("model_custom", model_default)
        )
        df = df_processed.copy()

        def get_unique(col):
            return sorted(
                df[col].dropna().unique().tolist()
            ) if col in df.columns else []

        col1, col2 = st.columns(2)
        with col1:
            thuong_hieu = st.selectbox(
                "Thương hiệu (thuong_hieu)", get_unique('thuong_hieu')
            )
            dong_xe = st.selectbox(
                "Dòng xe (dong_xe)", get_unique('dong_xe')
            )
            loai_xe = st.selectbox(
                "Loại xe (loai_xe)",
                get_unique('loai_xe') if 'loai_xe' in df.columns else []
            )
            xuat_xu = st.selectbox(
                "Xuất xứ (xuat_xu)",
                get_unique('xuat_xu') if 'xuat_xu' in df.columns else []
            )
        with col2:
            dung_tich = st.selectbox(
                "Dung tích xe (dung_tich_xe)",
                get_unique('dung_tich_xe') if 'dung_tich_xe' in df.columns else []
            )
            tuoi_xe = st.slider("Tuổi xe (năm)", 0, 30, 5)
            so_km_da_di = st.number_input(
                "Số km đã đi (so_km_da_di)",
                min_value=0, value=30000, step=1000
            )
            khoang_gia_min = st.number_input(
                "Khoảng giá min (khoang_gia_min) - triệu VND",
                min_value=0.0, value=0.0
            )
            khoang_gia_max = st.number_input(
                "Khoảng giá max (khoang_gia_max) - triệu VND",
                min_value=0.0, value=0.0
            )

        gia_thuc_te = st.number_input(
            "Giá xe (triệu VND)", min_value=0.0, value=30.0
        )

        # Tạo 1 dòng data giống cấu trúc df_processed
        sample = {}
        X_cols = df.drop(columns=['gia']).columns.tolist()

        for c in X_cols:
            if c == 'thuong_hieu':
                sample[c] = thuong_hieu
            elif c == 'dong_xe':
                sample[c] = dong_xe
            elif c == 'loai_xe':
                sample[c] = loai_xe
            elif c == 'xuat_xu':
                sample[c] = xuat_xu
            elif c == 'dung_tich_xe':
                sample[c] = dung_tich
            elif c == 'tuoi_xe':
                sample[c] = tuoi_xe
            elif c == 'so_km_da_di':
                sample[c] = so_km_da_di
            elif c == 'khoang_gia_min':
                sample[c] = khoang_gia_min if khoang_gia_min > 0 else np.nan
            elif c == 'khoang_gia_max':
                sample[c] = khoang_gia_max if khoang_gia_max > 0 else np.nan
            else:
                sample[c] = np.nan

        sample['gia'] = gia_thuc_te

        input_df = pd.DataFrame([sample])

        st.markdown("##### 🆕 Thông tin xe")
        st.dataframe(input_df.drop(columns=['phan_khuc'], errors='ignore'))

        # dùng threshold và df_anom từ session nếu có
        df_anom = st.session_state.get("df_anom", None)
        threshold = st.session_state.get("anom_threshold", 50)

        # Khởi tạo biến lưu kết quả kiểm tra
        if 'anom_check_result' not in st.session_state:
            st.session_state['anom_check_result'] = None

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            check_clicked = st.button("Bước 1: Kiểm tra bất thường")
        with col_btn2:
            post_clicked = st.button("Bước 2: Đăng tin")

        # --- Bước 1: Kiểm tra bất thường ---
        if check_clicked:
            try:
                # Gộp vào dữ liệu hiện có để tính score ổn định hơn
                df_all = pd.concat([df, input_df], ignore_index=True)

                # ĐÁNH DẤU TIN MỚI
                df_all['is_new'] = 0
                df_all.loc[df_all.index[-1], 'is_new'] = 1  # dòng cuối là tin mới

                df_all_anom, thres_used = detect_anomalies(
                    df_all, model_use, threshold=threshold, method="absolute"
                )

                # LẤY ĐÚNG TIN MỚI SAU KHI SORT THEO SCORE
                new_row = df_all_anom[df_all_anom['is_new'] == 1].iloc[0]
                score_new = new_row['score']
                is_anom_new = int(new_row['is_anomaly'])
                gia_pred_new = new_row['gia_predict']

                st.write(
                    f"**Giá thị trường (giá dự đoán):** {gia_pred_new:.2f} triệu VND"
                )
                st.write(
                    f"**Giá tin đăng (giá thực tế):** {gia_thuc_te:.2f} triệu VND"
                )
                st.write(
                    f"**Chênh lệch (giá tin đăng - giá thị trường):** {new_row['resid']:.2f} triệu VND"
                )
                st.write(
                    f"**Điểm bất thường (anomaly score):** {score_new:.2f} "
                    f"(ngưỡng: {thres_used:.2f})"
                )

                # Lưu kết quả kiểm tra vào session_state
                st.session_state['anom_check_result'] = {
                    "is_anomaly": is_anom_new,
                    "input_df": input_df.to_dict(orient="list"),
                    "gia": float(new_row['gia']),
                    "gia_predict": float(new_row['gia_predict']),
                    "resid": float(new_row['resid']),
                    "score": float(new_row['score'])
                }

                if is_anom_new == 1:
                    st.error(
                        f"**Giá xe bất thường**.\n"
                        f"Chênh lệch: {new_row['resid']:.2f} triệu VND so với giá thị trường.\n\n"
                        f"Nếu bạn vẫn muốn đăng tin, hệ thống sẽ chuyển thông tin cho Admin quản lý ở bước **Đăng tin**."
                    )
                else:
                    st.success(
                        "**Giá xe phù hợp**. Bạn có thể bấm **Đăng tin** để hoàn tất."
                    )

            except Exception as e:
                st.error(f"Lỗi khi tính điểm bất thường: {e}")

        # --- Bước 2: Đăng tin ---
        if post_clicked:
            result = st.session_state.get('anom_check_result', None)
            if result is None:
                st.warning("Vui lòng bấm **Kiểm tra bất thường** trước khi **Đăng tin**.")
            else:
                if result["is_anomaly"] == 0:
                    # Giá phù hợp -> chỉ báo thành công, không lưu file
                    st.success("Đăng tin thành công!")
                else:
                    # Giá bất thường -> lưu Excel và báo chuyển cho Admin
                    try:
                        input_df_dict = result["input_df"]
                        input_df_post = pd.DataFrame(input_df_dict)

                        record = {}
                        record['thoi_gian_dang'] = dt.datetime.now()

                        # Lưu toàn bộ thông tin tin đăng
                        for c in input_df_post.columns:
                            record[c] = input_df_post.iloc[0][c]

                        record['gia_thuc_te'] = result["gia"]
                        record['gia_du_doan'] = result["gia_predict"]
                        record['chenh_lech'] = result["resid"]
                        record['ly_do_bat_thuong'] = (
                            f"Giá tin đăng lệch {result['resid']:.2f} triệu VND "
                            f"so với giá dự đoán"
                        )
                        record['anomaly_score'] = result["score"]

                        append_new_anomaly(record)

                        st.success(
                            "Đăng tin thành công. **Chuyển thông tin cho Admin quản lý.**"
                        )
                    except Exception as e:
                        st.error(f"Lỗi khi lưu thông tin bất thường: {e}")

                # Sau khi đăng tin xong, xoá kết quả kiểm tra để tránh lưu lại lần nữa
                st.session_state['anom_check_result'] = None

# ---------- 5. PHÁT HIỆN BẤT THƯỜNG - ADMIN ----------
elif choice.startswith("5."):
    pastel_header("🚨", "Phát hiện bất thường – Admin", "#ede9fe")

    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình.")
    else:
        # ưu tiên model_anom nếu có, sau đó model_custom, cuối cùng model_default
        model_use = st.session_state.get(
            "model_anom",
            st.session_state.get("model_custom", model_default)
        )
        df = df_processed.copy()

        sub1, sub2 = st.tabs(
            ["Thống kê bất thường trên dữ liệu gốc",
             "Thống kê bất thường mới"]
        )

        # --- Phần 1: Thống kê bất thường trên dữ liệu gốc ---
        with sub1:
            st.markdown("##### 📊 Thống kê bất thường trên dữ liệu gốc")
            try:
                # DÙNG PHÂN VỊ 95% -> TỶ LỆ MẪU BẤT THƯỜNG < 5%
                df_all_anom_goc, thres_goc = detect_anomalies(
                    df.copy(), model_use, threshold=95, method="percentile"
                )
                # Lưu lại cho phần khác nếu cần
                st.session_state["df_anom"] = df_all_anom_goc
                st.session_state["anom_threshold"] = thres_goc

                df_anom_goc = df_all_anom_goc[df_all_anom_goc['is_anomaly'] == 1].copy()

                tong_bat_thuong = df_anom_goc.shape[0]
                tong_mau = df_all_anom_goc.shape[0]
                ty_le = 100.0 * tong_bat_thuong / tong_mau if tong_mau > 0 else 0.0

                st.write(
                    f"**Tổng số lượng mẫu bất thường:** {tong_bat_thuong} "
                    f"(~{ty_le:.2f}% của toàn bộ dữ liệu gốc)"
                )

                if tong_bat_thuong > 0:
                    # Chuẩn bị các cột hiển thị theo yêu cầu
                    df_display = df_anom_goc.copy()
                    df_display = df_display.rename(columns={
                        'gia': 'gia_thuc_te',
                        'gia_predict': 'gia_du_doan',
                        'score': 'anomaly_score'
                    })
                    df_display['chenh_lech'] = df_anom_goc['resid']
                    df_display['ly_do_bat_thuong'] = df_anom_goc['resid'].apply(
                        lambda x: f"Chênh lệch {x:.2f} triệu VND so với giá dự đoán"
                    )

                    # XÓA CÁC CỘT KỸ THUẬT KHÔNG HIỂN THỊ/EXPORT
                    drop_cols_admin = [
                        'resid', 'resid_z', 'vi_pham_minmax',
                        'ngoai_khoang_tin_cay', 'iso_score', 'is_anomaly'
                    ]
                    df_display = df_display.drop(columns=drop_cols_admin, errors='ignore')

                    # SẮP XẾP THỨ TỰ CỘT
                    ordered_cols_admin = [
                        'thuong_hieu', 'dong_xe', 'so_km_da_di', 'loai_xe',
                        'dung_tich_xe', 'xuat_xu', 'tuoi_xe',
                        'khoang_gia_min', 'khoang_gia_max',
                        'gia_thuc_te', 'gia_du_doan', 'chenh_lech',
                        'ly_do_bat_thuong', 'anomaly_score'
                    ]
                    cols_exist_admin = [c for c in ordered_cols_admin if c in df_display.columns]
                    df_display = df_display[cols_exist_admin]

                    # Hiển thị 5 mẫu đầu tiên
                    st.markdown("**5 mẫu bất thường đầu tiên:**")
                    st.dataframe(df_display.head(5))

                    # Nút download Excel toàn bộ mẫu bất thường
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df_display.to_excel(writer, index=False,
                                            sheet_name="Anomalies_goc")
                    excel_buffer.seek(0)

                    st.download_button(
                        label="⬇️ Xuất Excel toàn bộ mẫu bất thường (dữ liệu gốc)",
                        data=excel_buffer,
                        file_name="anomalies_goc.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Không phát hiện mẫu bất thường nào trên dữ liệu gốc.")
            except Exception as e:
                st.error(f"Lỗi khi thống kê bất thường trên dữ liệu gốc: {e}")

        # --- Phần 2: Thống kê bất thường mới ---
        with sub2:
            st.markdown("##### 🆕 Thống kê các tin đăng bất thường mới")

            df_new_anom = load_new_anomalies()

            if df_new_anom.empty:
                st.info("Chưa có tin đăng bất thường mới nào được lưu.")
            else:
                # Đảm bảo cột thời gian
                if 'thoi_gian_dang' in df_new_anom.columns:
                    df_new_anom['thoi_gian_dang'] = pd.to_datetime(
                        df_new_anom['thoi_gian_dang'], errors='coerce'
                    )
                    # Bộ lọc theo thời gian
                    min_date = df_new_anom['thoi_gian_dang'].min().date()
                    max_date = df_new_anom['thoi_gian_dang'].max().date()

                    start_date, end_date = st.date_input(
                        "Chọn khoảng thời gian",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )

                    if isinstance(start_date, dt.date) and isinstance(end_date, dt.date):
                        mask = (
                            (df_new_anom['thoi_gian_dang'].dt.date >= start_date) &
                            (df_new_anom['thoi_gian_dang'].dt.date <= end_date)
                        )
                        df_filtered = df_new_anom.loc[mask].copy()
                    else:
                        df_filtered = df_new_anom.copy()
                else:
                    df_filtered = df_new_anom.copy()

                # Sắp xếp giảm dần theo thời gian
                if 'thoi_gian_dang' in df_filtered.columns:
                    df_filtered = df_filtered.sort_values(
                        by='thoi_gian_dang', ascending=False
                    )

                if 'gia_thuc_te' not in df_filtered.columns and 'gia' in df_filtered.columns:
                    df_filtered['gia_thuc_te'] = df_filtered['gia']
                if 'gia_du_doan' not in df_filtered.columns and 'gia_predict' in df_filtered.columns:
                    df_filtered['gia_du_doan'] = df_filtered['gia_predict']
                if 'anomaly_score' not in df_filtered.columns and 'score' in df_filtered.columns:
                    df_filtered['anomaly_score'] = df_filtered['score']
                if 'chenh_lech' not in df_filtered.columns and 'resid' in df_filtered.columns:
                    df_filtered['chenh_lech'] = df_filtered['resid']
                if 'ly_do_bat_thuong' not in df_filtered.columns and 'chenh_lech' in df_filtered.columns:
                    df_filtered['ly_do_bat_thuong'] = df_filtered['chenh_lech'].apply(
                        lambda x: f"Chênh lệch {x:.2f} triệu VND so với giá dự đoán"
                    )

                df_filtered = df_filtered.drop(columns=['gia'], errors='ignore')

                ordered_cols_new = [
                    'thoi_gian_dang', 'thuong_hieu', 'dong_xe', 'so_km_da_di',
                    'loai_xe', 'dung_tich_xe', 'xuat_xu', 'tuoi_xe',
                    'khoang_gia_min', 'khoang_gia_max',
                    'gia_thuc_te', 'gia_du_doan', 'chenh_lech',
                    'ly_do_bat_thuong', 'anomaly_score'
                ]
                cols_exist_new = [c for c in ordered_cols_new if c in df_filtered.columns]
                df_display_new = df_filtered[cols_exist_new]

                tong_tin_bat_thuong = df_display_new.shape[0]

                st.write(
                    f"**Tổng số lượng tin đăng bất thường:** {tong_tin_bat_thuong}"
                )

                if tong_tin_bat_thuong > 0:
                    st.markdown("**Danh sách các tin đăng bất thường:**")
                    st.dataframe(df_display_new)

                    # Xuất Excel theo bộ lọc thời gian
                    excel_buffer_new = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer_new, engine="openpyxl") as writer:
                        df_display_new.to_excel(writer, index=False,
                                                sheet_name="Anomalies_moi")
                    excel_buffer_new.seek(0)

                    st.download_button(
                        label="⬇️ Xuất Excel tất cả các tin đăng bất thường",
                        data=excel_buffer_new,
                        file_name="anomalies_moi_filtered.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# ---------- 6. Thông tin nhóm ----------
elif choice.startswith("6."):
    pastel_header("👥", "Thành viên", "#dcfce7")

    st.markdown("""

**Nguyễn Thị Xuân Mai**  
  - Email: nguyentxmai@gmail.com  
  - Phụ trách: Phát triển giao diện GUI cho Dự án 1 – Dự đoán giá xe máy và phát hiện bất thường 

**Trần Thị Yến Nhi**  
  - Email: yennhi1928@gmail.com  
  - Phụ trách: Phát triển giao diện GUI cho Dự án 2 – Gợi ý xe máy tương tự và phân khúc thị trường
""")