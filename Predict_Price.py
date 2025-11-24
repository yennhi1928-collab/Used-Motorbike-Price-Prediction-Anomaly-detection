import pandas as pd
import numpy as np
import re
import unicodedata
import datetime as dt

# hàm preprocessing
def preprocessing_data(df, is_train=True):
    df = df.copy()
    # làm sạch tên cột
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
            # Nếu chỉ là số, convert trực tiếp
            try:
                return float(text)
            except:
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
                except:
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
    drop_cols = ['id', 'tieu_de', 'dia_chi', 'mo_ta_chi_tiet', 
                 'href', 'trong_luong', 'chinh_sach_bao_hanh', 'tinh_trang','nam_dang_ky']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Xử lý missing values
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

    # Xử lý category hiếm
    if 'dung_tich_xe' in df.columns:
        df['dung_tich_xe'] = df['dung_tich_xe'].replace({
            'Không biết rõ': 'Khác',
            'Đang cập nhật': 'Khác',
            'Nhật Bản': 'Khác'
        })
    if 'xuat_xu' in df.columns:
        df['xuat_xu'] = df['xuat_xu'].replace('Bảo hành hãng', 'Đang cập nhật')
    if 'thuong_hieu' in df.columns and is_train:
        threshold = 10
        popular = df['thuong_hieu'].value_counts()
        popular = popular[popular >= threshold].index
        df['thuong_hieu'] = df['thuong_hieu'].apply(lambda x: x if x in popular else 'Hãng khác')
    if 'dong_xe' in df.columns and is_train:
        threshold = 10
        popular = df['dong_xe'].value_counts()
        popular = popular[popular >= threshold].index
        df['dong_xe'] = df['dong_xe'].apply(lambda x: x if x in popular else 'Khác')

    # Tạo phân khúc theo thương hiệu nếu có cột giá
    if 'gia' in df.columns and is_train:
        if df.empty or df['thuong_hieu'].nunique() == 0:
            df['phan_khuc'] = np.nan
        else:
            brand_mean = df.groupby('thuong_hieu', as_index=False)['gia'].mean().rename(columns={'gia': 'mean_price'})
            if brand_mean.empty:
                df['phan_khuc'] = np.nan
            else:
                brand_mean['phan_khuc'] = pd.cut(
                    brand_mean['mean_price'],
                    bins=[-float('inf'), 50, 100, float('inf')],
                    labels=['pho_thong', 'trung_cap', 'cao_cap'],
                    right=False
                )
                df = df.merge(brand_mean[['thuong_hieu', 'phan_khuc']], on='thuong_hieu', how='left')
                df['phan_khuc'] = df['phan_khuc'].astype('object')

        # Loại outlier nếu is_train
        if is_train:
            def remove_outliers_by_brand(df, column, lower_percentile=0.25, upper_percentile=0.75, threshold=1.5):
                if column not in df.columns:
                    return df
                def remove_group_outliers(group):
                    Q1 = group[column].quantile(lower_percentile)
                    Q3 = group[column].quantile(upper_percentile)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
                return df.groupby('phan_khuc', group_keys=False).apply(remove_group_outliers)
            remove_outlier_cols = [c for c in ['gia', 'so_km_da_di','tuoi_xe'] if c in df.columns]
            for c in remove_outlier_cols:
                df = remove_outliers_by_brand(df, c)
            df = df.reset_index(drop=True)

    print('Sau tiền xử lý:', df.shape)
    return df
    
# Hàm tín hiệu bất thường
from sklearn.ensemble import IsolationForest
def detect_anomalies(df, model, threshold=50, method='absolute'):
    # Dự đoán giá từ mô hình đã huấn luyện
    df['gia_predict'] = model.predict(df[['thuong_hieu', 'dong_xe', 'tuoi_xe', 'so_km_da_di', 'loai_xe', 'dung_tich_xe','xuat_xu']])
    # Tính residual và z-score theo từng thương hiệu
    df['resid'] = df['gia'] - df['gia_predict']
    def compute_resid_z(df):
        group_sizes = df['thuong_hieu'].value_counts()
        small_groups = group_sizes[group_sizes < 2].index

        df['resid_z'] = 0.0
        # Nhóm đủ lớn (>= 2 mẫu)
        df.loc[df['thuong_hieu'].isin(group_sizes[group_sizes >= 2].index), 'resid_z'] = \
            df.groupby('thuong_hieu')['resid'].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0
            )
        # Nhóm nhỏ (<2 mẫu) → fallback toàn cục
        global_mean = df['resid'].mean()
        global_std = df['resid'].std(ddof=0)
        if global_std > 0:
            mask = df['thuong_hieu'].isin(small_groups)
            df.loc[mask, 'resid_z'] = (df.loc[mask, 'resid'] - global_mean) / global_std
        return df

    df = compute_resid_z(df)
    # Tính khoảng tin cậy
    p10, p90 = np.percentile(df['gia'].dropna(), [10, 90])
    # Đánh dấu vi phạm giá min và max
    for col in ['gia', 'khoang_gia_min', 'khoang_gia_max']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if {'khoang_gia_min', 'khoang_gia_max'}.issubset(df.columns):
        df['vi_pham_minmax'] = ((df['gia'] < df['khoang_gia_min']) | 
                                (df['gia'] > df['khoang_gia_max'])).astype(int)
    else:
        df['vi_pham_minmax'] = 0
    df['ngoai_khoang_tin_cay'] = ((df['gia'] < p10) | (df['gia'] > p90)).astype(int)   
    # Unsupervisor learning
    iso_features = ['gia', 'gia_predict', 'resid', 'resid_z', 'so_km_da_di','tuoi_xe']
    iso_features = [c for c in iso_features if c in df.columns]
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['iso_score'] = iso.fit_predict(df[iso_features])
    df['iso_score'] = df['iso_score'].apply(lambda x: 1 if x == -1 else 0)
    # Tính điểm tổng hợp dựa trên trọng số
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    df['score'] = 100 * (
    (w1 * np.abs(df['resid_z']) +
     w2 * df['vi_pham_minmax'] +
     w3 * df['ngoai_khoang_tin_cay'] +
     w4 * df['iso_score'])
    / (w1 + w2 + w3 + w4)
)
    if method == 'percentile':
        threshold_value = np.percentile(df['score'], 95)
    else:
        threshold_value = threshold
        df['is_anomaly'] = (df['score'] >= threshold_value).astype(int)
        df_result = df.sort_values('score', ascending=False).reset_index(drop=True)
        print(f"Phát hiện {df_result['is_anomaly'].sum()} mẫu bất thường (threshold={threshold_value:.2f}, method={method})")
        return df_result