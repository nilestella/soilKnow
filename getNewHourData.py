import requests
import numpy as np
from PIL import Image
import io
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置中文字体
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 颜色范围映射
COLOR_RANGES = [
    ((90, 0, 141), (0.000, 0.040)),
    ((164, 0, 69), (0.040, 0.080)),
    ((249, 0, 7), (0.080, 0.120)),
    ((252, 64, 0), (0.120, 0.160)),
    ((255, 140, 0), (0.160, 0.200)),
    ((255, 195, 0), (0.200, 0.240)),
    ((255, 236, 0), (0.240, 0.280)),
    ((192, 237, 0), (0.280, 0.320)),
    ((86, 203, 13), (0.320, 0.360)),
    ((11, 189, 49), (0.360, 0.400)),
    ((2, 228, 156), (0.400, 0.440)),
    ((0, 243, 252), (0.440, 0.480)),
    ((0, 120, 249), (0.480, 0.520)),
    ((0, 2, 254), (0.520, 0.560))
]

# 目标像素坐标
TARGET_POINT = (1255, 1025)

# 不同深度的URL模板
DEPTH_TEMPLATES = {
    "0-5cm": "http://image.data.cma.cn/vis/NAFP_CLDAS2.0_RT_JPG_SM000005/{date_dir}/NAFP_CLDAS2.0_RT_JPG_SM000005_{filename}",
    "0-10cm": "http://image.data.cma.cn/vis/NAFP_CLDAS2.0_RT_JPG_SM000010/{date_dir}/NAFP_CLDAS2.0_RT_JPG_SM000010_{filename}",
    "10-40cm": "http://image.data.cma.cn/vis/NAFP_CLDAS2.0_RT_JPG_SM010040/{date_dir}/NAFP_CLDAS2.0_RT_JPG_SM010040_{filename}",
    "40-100cm": "http://image.data.cma.cn/vis/NAFP_CLDAS2.0_RT_JPG_SM040100/{date_dir}/NAFP_CLDAS2.0_RT_JPG_SM040100_{filename}",
    "100-200cm": "http://image.data.cma.cn/vis/NAFP_CLDAS2.0_RT_JPG_SM100200/{date_dir}/NAFP_CLDAS2.0_RT_JPG_SM100200_{filename}"
}


def download_and_process_image(url):
    """下载并处理图像"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"❌ 图像处理失败: {url} ({str(e)})")
        return None


def extract_colorbar(img_array, colorbar_xrange=(1660, 1700), colorbar_yrange=(800, 1380)):
    """从图像中提取颜色条"""
    colorbar = img_array[colorbar_yrange[0]:colorbar_yrange[1], colorbar_xrange[0]:colorbar_xrange[1], :]
    if colorbar.size == 0:
        raise ValueError("颜色条范围无效")
    return colorbar


def create_colormap(colorbar):
    """创建颜色映射表"""
    height = colorbar.shape[0]
    colormap = {}
    value_map = {}
    num_colors = len(COLOR_RANGES)
    block_height = height // num_colors

    for i, (color, val_range) in enumerate(COLOR_RANGES):
        y_start = i * block_height + block_height // 4
        y_end = (i + 1) * block_height - block_height // 4
        mid_col = colorbar.shape[1] // 2
        color_patch = colorbar[y_start:y_end, mid_col - 2:mid_col + 2, :]
        avg_color = np.mean(color_patch, axis=(0, 1)).astype(int)
        avg_color = tuple(avg_color[:3])
        colormap[avg_color] = val_range
        value_map[avg_color] = np.mean(val_range)

    return colormap, value_map


def get_soil_moisture(img_array, colormap, value_map, x, y):
    """获取指定坐标的土壤湿度值"""
    colors = np.array(list(colormap.keys()))
    mid_values = np.array(list(value_map.values()))

    # 获取5x5区域
    y_start, y_end = max(0, y - 2), min(img_array.shape[0], y + 3)
    x_start, x_end = max(0, x - 2), min(img_array.shape[1], x + 3)
    neighborhood = img_array[y_start:y_end, x_start:x_end, :]

    # 处理透明像素
    alpha = neighborhood[:, :, 3]
    mask = alpha > 200
    pixels = neighborhood[mask, :3]

    if pixels.size == 0:
        return np.nan

    # 找到最接近的颜色
    diff = pixels[:, np.newaxis, :] - colors[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    closest_idx = np.argmin(distances, axis=1)
    values = mid_values[closest_idx]

    return np.nanmean(values)


def process_single_timepoint(current_time):
    """处理单个时间点的所有深度数据"""
    results = {"time": current_time.strftime("%m/%d/%Y %H:00")}

    for depth, template in DEPTH_TEMPLATES.items():
        # 生成文件名和URL
        filename = current_time.strftime("%Y%m%d%H") + ".png"
        date_dir = current_time.strftime("%Y%m%d")
        url = template.format(date_dir=date_dir, filename=filename)

        # 处理图像
        img_array = download_and_process_image(url)
        if img_array is not None:
            try:
                colorbar = extract_colorbar(img_array)
                colormap, value_map = create_colormap(colorbar)
                moisture = get_soil_moisture(img_array, colormap, value_map, *TARGET_POINT)
                results[depth] = moisture
                print(f"✅ {current_time.strftime('%m/%d/%Y %H:00')} {depth} 处理成功 | 湿度: {moisture:.3f}")
            except Exception as e:
                results[depth] = np.nan
                print(f"❌ {current_time.strftime('%m/%d/%Y %H:00')} {depth} 数据处理失败: {str(e)}")
        else:
            results[depth] = np.nan
            print(f"❌ {current_time.strftime('%m/%d/%Y %H:00')} {depth} 图像下载失败")

    return results


def append_to_excel(new_data, output_file):
    """将新数据追加到现有Excel文件"""
    try:
        # 如果文件存在，读取现有数据
        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
        else:
            existing_df = pd.DataFrame(columns=["time", "0-5cm", "0-10cm", "10-40cm", "40-100cm", "100-200cm"])

        # 创建新数据的DataFrame
        new_df = pd.DataFrame([new_data])

        # 合并数据并去重
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=["time"], keep="last")

        # 按时间排序
        combined_df["time"] = pd.to_datetime(combined_df["time"])
        combined_df = combined_df.sort_values("time")
        combined_df["time"] = combined_df["time"].dt.strftime("%m/%d/%Y %H:00")

        # 保存到Excel
        combined_df.to_excel(output_file, index=False)
        print(f"✅ 数据已追加到: {output_file}")

    except Exception as e:
        print(f"❌ 保存数据失败: {str(e)}")


def hourly_task():
    """每小时执行的任务"""
    # 获取当前时间的前两小时（整点）
    current_time = datetime.now()
    target_time = current_time - timedelta(hours=3)
    target_time = target_time.replace(minute=0, second=0, microsecond=0)

    print(f"\n⏰ 开始处理 {target_time.strftime('%m/%d/%Y %H:00')} 的数据...")

    try:
        # 处理数据
        result = process_single_timepoint(target_time)

        # 追加到Excel文件
        append_to_excel(result, "./hour.xlsx")

    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")

    # 计算下一次执行时间（下一个整点）
    next_run = (current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    sleep_time = (next_run - datetime.now()).total_seconds()
    print(f"⏳ 下次执行时间: {next_run.strftime('%m/%d/%Y %H:00:00')}")

    return max(sleep_time, 0)  # 确保不返回负数


def main():
    print("🚀 土壤湿度数据采集程序已启动")
    print("📝 数据将追加到: hour_copy.xlsx")
    print("🔄 每小时自动获取前两小时的数据")

    while True:
        try:
            sleep_time = hourly_task()
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n🛑 程序已手动停止")
            break
        except Exception as e:
            print(f"❌ 发生未预期错误: {str(e)}")
            time.sleep(3600)  # 出错时等待1小时再重试


if __name__ == "__main__":
    main()