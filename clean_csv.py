"""
clean_csv.py — Clean/reshape EnergyPlus hourly outputs and build feature & target columns
Author: Hantao He | Project: photovoltaic_prediction | 2026-09
"""

import pandas as pd
import numpy as np
import glob
import os

# 1. 更新自变量列
independent_vars = [
    'FAR', 'Building Height', 'Building Amount', 'Shape Factor',
    'depth', 'frontage', 'rotation', 'SVF'
]

# 2. 因变量列保持不变
dependent_vars = [
    'hourly_EUI',
    'hourly_power_generation_roof',
    'hourly_power_generation_east',
    'hourly_power_generation_south',
    'hourly_power_generation_west',
    'hourly_power_generation_north'
]

# 3. 更新天气变量列
weather_vars = [
    'date',
    'dry_bulb_temperature',
    'relative_humidity',
    'DNI',
    'DHI',
    'GHI',
    'Solar Altitude'
]

output_cols = weather_vars + independent_vars + dependent_vars

INVALID_SIZE_KB = 1
INVALID_SIZE_BYTES = INVALID_SIZE_KB * 1024


def is_valid_file(filepath):
    try:
        return os.path.getsize(filepath) > INVALID_SIZE_BYTES
    except OSError:
        return False


def process_file(filepath, folder_name, weather_df):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  读取失败: {e}")
        return None

    df = df.reset_index(drop=True)

    # 提取标量参数
    scalar_map = {
        'FAR': df['FAR'].dropna().iloc[0] if 'FAR' in df.columns else 0,
        'depth': df['depth'].dropna().iloc[0] if 'depth' in df.columns else 0,
        'frontage': df['frontage'].dropna().iloc[0] if 'frontage' in df.columns else 0,
        'rotation': df['rotation'].dropna().iloc[0] if 'rotation' in df.columns else 0,
        'SVF': df['SVF'].dropna().iloc[0] if 'SVF' in df.columns else 0,
        'story': df['story'].dropna().iloc[0] if 'story' in df.columns else 1,
        'AH': df['AH'].dropna().iloc[0] if 'AH' in df.columns else 0,
        'TH': df['TH'].dropna().iloc[0] if 'TH' in df.columns else 0,
        'TBV': df['TBV'].dropna().iloc[0] if 'TBV' in df.columns else 0,
        'ABSA': df['ABSA'].dropna().iloc[0] if 'ABSA' in df.columns else 0
    }

    for col, val in scalar_map.items():
        df[col] = val

    # 合并新的天气数据
    if len(df) != len(weather_df):
        return None

    for w_col in weather_vars:
        df[w_col] = weather_df[w_col].values

    # 计算派生自变量
    df['Building Height'] = scalar_map['story'] * 3
    df['Building Amount'] = scalar_map['TH'] / scalar_map['AH'] if scalar_map['AH'] != 0 else 0
    df['Shape Factor'] = scalar_map['ABSA'] / scalar_map['TBV'] if scalar_map['TBV'] != 0 else 0

    # 计算因变量
    denom = (df['depth'] * df['frontage'] * df['story'] * df['TH'])
    df['hourly_EUI'] = (df['cooling'] * df['AH']) / denom

    # 发电量计算
    factor_roof = 0.60 * 0.12 * 0.9 / 1000
    factor_facade = 0.53 * 0.12 * 0.9 / 1000
    cos_alt = np.cos(np.radians(df['Solar Altitude'].clip(lower=0)))

    df['hourly_power_generation_roof'] = df['annual_hourly_radiation_roof'] * df['depth'] * df['frontage'] * factor_roof
    df['hourly_power_generation_east'] = df['annual_hourly_radiation_wall_east'] * 3 * df['story'] * df[
        'depth'] * factor_facade * cos_alt
    df['hourly_power_generation_south'] = df['annual_hourly_radiation_wall_south'] * 3 * df['story'] * df[
        'frontage'] * factor_facade * cos_alt
    df['hourly_power_generation_west'] = df['annual_hourly_radiation_wall_west'] * 3 * df['story'] * df[
        'depth'] * factor_facade * cos_alt
    df['hourly_power_generation_north'] = df['annual_hourly_radiation_wall_north'] * 3 * df['story'] * df[
        'frontage'] * factor_facade * cos_alt

    # 清理
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    df_cleaned = df[output_cols].copy()
    df_cleaned['building_type'] = folder_name
    df_cleaned['source_file'] = f"{folder_name}_{os.path.basename(filepath)}"

    return df_cleaned


def main():
    weather_file = os.path.join('weather_data', 'weather_data.csv')
    if not os.path.exists(weather_file):
        print(f"❌ 错误：未找到天气文件 {weather_file}")
        return

    # --- 核心修改：基于 1999 年自动生成 date 列 ---
    weather_df = pd.read_csv(weather_file).reset_index(drop=True)
    weather_len = len(weather_df)

    # 自动创建 1999 年的时间序列（1999 是平年，共 8760 小时）
    weather_df['date'] = pd.date_range(start='1999-01-01 00:00:00', periods=weather_len, freq='h')
    weather_df['date'] = weather_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"✅ 成功读取天气数据并分配 1999 年日期，长度: {weather_len}")
    # ------------------------------------------

    root_dir = 'simulation_result'
    if not os.path.exists(root_dir):
        print(f"❌ 错误：根目录 {os.path.abspath(root_dir)} 不存在！")
        return

    cleaned_dfs = []

    print("\n--- 开始深度处理文件 ---")
    type_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for type_folder in type_folders:
        type_path = os.path.join(root_dir, type_folder)
        files = glob.glob(os.path.join(type_path, 'data_output_*.csv'))

        if not files:
            continue

        for filepath in files:
            if not is_valid_file(filepath):
                continue

            df_c = process_file(filepath, type_folder, weather_df)

            if df_c is not None:
                cleaned_dfs.append(df_c)
                if len(cleaned_dfs) % 50 == 0:
                    print(f"  📝 已成功处理 {len(cleaned_dfs)} 个文件...")
            else:
                print(f"  ❌ 失败 {os.path.basename(filepath)}: 数据行数不匹配")

    print("--- 处理结束 ---\n")

    if cleaned_dfs:
        combined_df = pd.concat(cleaned_dfs, ignore_index=True)
        combined_df.to_csv('all_cleaned_data.csv', index=False)
        print(f"🚀 大功告成！成功合并 {len(cleaned_dfs)} 个文件至 all_cleaned_data.csv")
    else:
        print("🛑 最终结果为空：没有找到任何符合条件的 CSV 数据。")


if __name__ == '__main__':
    main()

# --- 最终检查脚本部分 ---
import os

output_file = 'all_cleaned_data.csv'
if os.path.exists(output_file):
    print("\n" + "·" * 50)
    print("🔎 正在执行最终输出文件深度检查...")
    check_df = pd.read_csv(output_file)
    rows, cols = check_df.shape
    building_count = len(check_df['source_file'].unique())
    print(f"📈 数据规模: {rows} 行 x {cols} 列")
    print(f"📋 包含表头: {check_df.columns.tolist()}")

    expected_rows = building_count * 8760
    if rows == expected_rows:
        print(f"✅ 对齐检查: 记录数与 1999 年 8760 小时完全匹配。")
    else:
        print(f"❌ 对齐检查: 记录数({rows}) 与理论值({expected_rows}) 不符！")
    print("·" * 50 + "\n")