import sys
import os
import pickle
import subprocess
import tempfile
import threading
import time
import numpy as np
import mitsuba as mi
from matplotlib.colors import LinearSegmentedColormap

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QGroupBox,
    QFormLayout, QComboBox, QCheckBox, QColorDialog, QSizePolicy, QScrollArea,
    QFrame, QSplitter, QProgressBar, QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QKeySequence
from PyQt5.QtCore import Qt, QSettings, QThread, QObject, QTimer, pyqtSignal


_UI_LANG = "zh"
CUSTOM_CMAP_TOKEN = "(自定义txt)"

_UI_TEXT = {
    "zh": {
        "lang_tip": "切换为英文界面",
        "cloud_group": "1. 点云",
        "btn_file": "选择文件",
        "btn_folder": "选择文件夹",
        "batch_check": "批量渲染文件夹内全部 .txt",
        "file_none": "单文件：未选择",
        "folder_none": "文件夹：未选择",
        "file_selected": "单文件：{path}",
        "folder_selected": "文件夹：{path}",
        "label_group": "2. 标签",
        "labels_in_cloud": "点云含标签",
        "labels_in_cloud_tip": "打开后从点云文件指定列读取标签（单文件和批量都生效）。XYZ 仍使用前三列。",
        "col": "列",
        "col_tip": "点云文件中标签所在列，从 1 开始计数，默认第 4 列。批量时文件夹内所有文件使用同一列。",
        "multiclass": "多分类",
        "multiclass_tip": "打开后允许 2 个以上类别，颜色按当前色轴自动分配；关闭则仅支持二分类，使用手动配色。",
        "btn_label_file": "选择单独标签文件",
        "btn_clear": "清除",
        "label_none": "未选择",
        "label_embedded": "每个点云文件都使用第 {col} 列作为标签",
        "noise_group": "3. 噪声（可选）",
        "noise_dir_tip": "批量时噪声文件名必须与干净点云相同，例如 clean/a.txt 对应 noise/a.txt",
        "noise_file": "单文件：{path}",
        "noise_batch": "批量文件夹：{path}（文件名需与干净点云相同）",
        "list_sep": "；",
        "param_group": "4. 渲染参数",
        "ball_radius": "主点球半径",
        "noise_radius": "噪声球半径",
        "floor_z": "底板高度 z",
        "camera_origin": "相机 origin",
        "color_group": "5. 着色",
        "colormap": "色轴",
        "cmap_custom": "(自定义txt)",
        "btn_load_cmap": "载入 txt",
        "cmap_custom_current": "当前：自定义 {name}",
        "cmap_custom_empty": "当前：自定义（尚未载入）",
        "cmap_builtin": "当前：内置 {name}",
        "caption_gradient": "渐变色轴预览（无标签时使用）",
        "caption_multi_n": "多分类色轴预览（{n} 类，按当前色轴自动分配）",
        "caption_multi_empty": "多分类色轴预览（载入标签后按当前色轴自动分配）",
        "class0": "类别 0",
        "class1": "类别 1",
        "output_group": "6. 输出",
        "btn_xml": "指定 XML 路径",
        "btn_auto": "恢复自动",
        "output_manual": "手动：{path}",
        "output_auto": "自动：{path}",
        "output_default": "单文件模式将自动在点云同目录生成 XML",
        "btn_start": "开始渲染",
        "btn_reload": "从 TIFF 载入预览",
        "preview_group": "预览",
        "preview_placeholder": "渲染结果将显示在这里",
        "log_group": "日志",
        "zmin": "zmin: {value}",
        "zmax": "zmax: {value}",
        "zmin_empty": "zmin: -",
        "zmax_empty": "zmax: -",
        "dlg_wait": "请稍候",
        "dlg_error": "错误",
        "dlg_info": "提示",
        "dlg_rendering": "正在渲染",
        "dlg_yes": "是",
        "dlg_no": "否",
        "close_prompt": "渲染尚未完成。要取消渲染并关闭窗口吗？",
        "cmap_fail": "色轴载入失败",
        "choose_cmap": "选择自定义色轴 txt",
        "choose_class_color": "选择类别 {index} 颜色",
        "choose_cloud": "选择点云文件",
        "choose_cloud_dir": "选择点云文件夹",
        "choose_noise": "选择噪声点云文件",
        "choose_noise_dir": "选择噪声点云文件夹",
        "choose_label": "选择标签文件",
        "choose_xml": "选择输出 XML 路径",
        "filter_txt": "TXT 文件 (*.txt)",
        "filter_label": "TXT 文件 (*.txt);;CSV 文件 (*.csv);;所有文件 (*)",
        "filter_xml": "XML 文件 (*.xml)",
        "rendering_wait": "正在渲染，请稍候…",
        "rendering_elapsed": "正在渲染，请稍候… 已用时 {seconds} 秒",
        "preparing": "准备中",
        "queue_status": "队列中共 {n} 个任务",
        "cancel_render": "取消渲染",
        "cancelling": "正在取消…",
        "cancelling_title": "正在取消渲染，请稍候…",
        "stopping": "正在停止当前任务",
        "cancel_requested": "⏹️ 已请求取消渲染",
        "render_cancelled": "⏹️ 渲染已取消。未完成的图像不会保存。",
        "busy_stop": "当前正在渲染。如需停止，请点击“取消渲染”或按 Esc。",
        "need_folder": "请先选择点云文件夹，或取消批量模式。",
        "need_file": "请先选择点云文件（或启用批量并选择文件夹）。",
        "no_txt": "该文件夹下未找到 .txt 点云文件。",
        "reload_busy": "正在渲染。请等待完成，或先取消当前渲染。",
        "reload_need_file": "请先在单文件模式下选择点云并完成一次渲染。",
        "reload_missing": "未找到 TIFF 文件，请先渲染或确认路径。",
        "reload_ok": "🔄 已从 TIFF 重新载入预览",
        "reload_fail": "⚠️ 预览载入失败：{error}",
        "variant_fail": "⚠️ 设置 Mitsuba 变体失败：{error}\n将尝试使用 'scalar_rgb'",
        "variant_fail2": "⚠️ 备用变体设置仍失败：{error}",
        "batch_noise_hint": "批量噪声将按文件名与干净点云一一对应，例如 clean/a.txt 对应 noise/a.txt",
        "noise_cleared": "已取消噪声点云文件/文件夹",
        "label_cleared": "已取消标签文件",
        "output_reset": "已取消手动输出路径，恢复为自动生成",
        "read_cloud_fail": "⚠️ 读取点云失败，无法更新 z 范围：{error}",
        "label_read_fail": "⚠️ 标签文件读取失败：{error}",
        "label_reload_fail": "⚠️ 按当前标签设置重新读取失败：{error}",
        "cmap_loaded": "🎨 已载入自定义色轴：{path}",
        "color_updated": "🎨 已更新类别 {index} 配色：{hex}",
        "mode_multi": "多分类",
        "mode_binary": "二分类",
        "batch_labels": "🏷️ 批量模式将从每个点云文件的第 {col} 列读取{mode}标签。",
        "batch_ignore_label": "⚠️ 当前批量模式会忽略单个标签文件，请勾选“点云含标签”并指定标签列。",
        "batch_noise_dir": "🌫️ 批量噪声文件夹：{path}。噪声文件名必须与对应干净点云相同。",
        "batch_ignore_noise": "⚠️ 批量模式不会使用单个噪声文件，请选择噪声文件夹，并保证文件名与干净点云一一对应。",
        "batch_list": "📁 将批量渲染 {n} 个文件：",
        "noise_missing": "⚠️ 未找到与 {name} 对应的噪声文件（文件名需相同），本文件不叠加噪声。",
        "batch_start": "开始批量渲染...",
        "batch_done": "🎉 批量渲染完成！如需查看单个结果，请切换到单文件模式并载入 TIFF 预览。",
        "single_start": "开始渲染（单文件）...",
        "single_done": "🎉 单文件渲染完成。",
        "rendering_file": "📄 渲染文件：{path}",
        "noise_added": "🌫️ 已叠加噪声：{path}",
        "noise_fail": "⚠️ 噪声点云读取失败：{error}",
        "labels_from_col": "🏷️ 已从点云第 {col} 列读取标签（{n} 类）",
        "labels_from_file": "🏷️ 已载入标签文件：{path}（{n} 类）",
        "labels_fallback": "⚠️ 标签文件读取失败，将回退到渐变色轴：{error}",
        "writing_tiff": "正在写入 TIFF...",
        "render_done_files": "✅ 渲染完成：\n{xml}\n{exr}\n{tiff}",
        "status_rendering": "正在渲染 {i}/{total}：{name}",
        "render_fail": "❌ 渲染失败：{path} - {error}",
        "render_abort": "❌ 渲染任务异常中止：{error}",
        "proc_cancel": "⏹️ 已取消，正在停止渲染进程。",
        "proc_crash": "❌ 渲染进程异常退出（代码 {code}）。{detail}",
        "proc_cancelled": "⏹️ 渲染已取消。",
        "proc_abort": "❌ 渲染任务异常中止：{error}",
        "err_empty_labels": "标签为空。",
        "err_binary": "当前为二分类模式，检测到 {n} 个类别。请勾选“多分类标签”。",
        "err_empty_cloud": "点云文件为空。",
        "err_need_xyz": "点云至少需要 x y z 三列。",
        "err_col_start": "标签列必须从 1 开始计数。",
        "err_col_missing": "已勾选“点云含标签”，但文件只有 {ncols} 列，无法读取第 {col} 列。",
        "err_count_mismatch": "标签数 {nlabels} 与点数 {npoints} 不一致。",
        "err_cmap_rows": "自定义色轴文件至少需要两行 RGB。",
        "err_label_mismatch": "标签数量与点云点数不一致。",
        "err_tiff_shape": "无法识别的图像形状：{shape}",
        "err_tiff_hw": "TIFF 需要 H×W×3 图像",
        "err_tiff_type": "不支持的 TIFF 字段类型：{typ}",
        "err_tiff_endian": "只支持小端未压缩 TIFF",
        "err_tiff_tag": "TIFF 缺少字段 {tag}",
        "err_tiff_compress": "只支持未压缩、逐像素排列的 TIFF",
        "err_tiff_strip": "只支持单条带 TIFF",
        "err_tiff_format": "不支持的 TIFF 像素格式：bits={bits} format={sample_format}",
    },
    "en": {
        "lang_tip": "Switch to Chinese",
        "cloud_group": "1. Point cloud",
        "btn_file": "Choose file",
        "btn_folder": "Choose folder",
        "batch_check": "Batch-render every .txt in the folder",
        "file_none": "File: none",
        "folder_none": "Folder: none",
        "file_selected": "File: {path}",
        "folder_selected": "Folder: {path}",
        "label_group": "2. Labels",
        "labels_in_cloud": "Labels in point cloud",
        "labels_in_cloud_tip": "Read labels from a chosen column. XYZ stays in the first three columns. Applies to single files and batches.",
        "col": "Column",
        "col_tip": "Label column, counted from 1. Default is column 4. Batch mode uses the same column for every file.",
        "multiclass": "Multiclass",
        "multiclass_tip": "Allow more than two classes and assign colors from the current colormap. Off: binary labels only, using the manual colors.",
        "btn_label_file": "Choose label file",
        "btn_clear": "Clear",
        "label_none": "None",
        "label_embedded": "Every file uses column {col} as labels",
        "noise_group": "3. Noise (optional)",
        "noise_dir_tip": "In batch mode the noise file name must match the clean cloud, for example clean/a.txt and noise/a.txt.",
        "noise_file": "File: {path}",
        "noise_batch": "Batch folder: {path} (file names must match the clean clouds)",
        "list_sep": "; ",
        "param_group": "4. Render settings",
        "ball_radius": "Sphere radius",
        "noise_radius": "Noise radius",
        "floor_z": "Floor height z",
        "camera_origin": "Camera origin",
        "color_group": "5. Color",
        "colormap": "Colormap",
        "cmap_custom": "Custom txt",
        "btn_load_cmap": "Load txt",
        "cmap_custom_current": "Current: custom {name}",
        "cmap_custom_empty": "Current: custom (not loaded)",
        "cmap_builtin": "Current: built-in {name}",
        "caption_gradient": "Colormap preview (used when there are no labels)",
        "caption_multi_n": "Multiclass preview ({n} classes, assigned from the current colormap)",
        "caption_multi_empty": "Multiclass preview (assigned from the current colormap after labels load)",
        "class0": "Class 0",
        "class1": "Class 1",
        "output_group": "6. Output",
        "btn_xml": "Set XML path",
        "btn_auto": "Use automatic",
        "output_manual": "Manual: {path}",
        "output_auto": "Automatic: {path}",
        "output_default": "XML is created next to the point cloud (single-file mode)",
        "btn_start": "Start render",
        "btn_reload": "Load preview from TIFF",
        "preview_group": "Preview",
        "preview_placeholder": "The render will appear here",
        "log_group": "Log",
        "zmin": "zmin: {value}",
        "zmax": "zmax: {value}",
        "zmin_empty": "zmin: -",
        "zmax_empty": "zmax: -",
        "dlg_wait": "Please wait",
        "dlg_error": "Error",
        "dlg_info": "Notice",
        "dlg_rendering": "Rendering",
        "dlg_yes": "Yes",
        "dlg_no": "No",
        "close_prompt": "The render is still running. Cancel it and close the window?",
        "cmap_fail": "Could not load colormap",
        "choose_cmap": "Choose custom colormap txt",
        "choose_class_color": "Choose color for class {index}",
        "choose_cloud": "Choose point cloud",
        "choose_cloud_dir": "Choose point cloud folder",
        "choose_noise": "Choose noise point cloud",
        "choose_noise_dir": "Choose noise folder",
        "choose_label": "Choose label file",
        "choose_xml": "Choose output XML path",
        "filter_txt": "TXT files (*.txt)",
        "filter_label": "TXT files (*.txt);;CSV files (*.csv);;All files (*)",
        "filter_xml": "XML files (*.xml)",
        "rendering_wait": "Rendering, please wait…",
        "rendering_elapsed": "Rendering, please wait… {seconds} s elapsed",
        "preparing": "Preparing",
        "queue_status": "{n} job(s) in the queue",
        "cancel_render": "Cancel render",
        "cancelling": "Cancelling…",
        "cancelling_title": "Cancelling the render, please wait…",
        "stopping": "Stopping the current job",
        "cancel_requested": "⏹️ Cancel requested",
        "render_cancelled": "⏹️ Render cancelled. Unfinished images were not saved.",
        "busy_stop": "A render is already running. Click Cancel render or press Esc to stop it.",
        "need_folder": "Choose a point cloud folder, or turn off batch mode.",
        "need_file": "Choose a point cloud file, or turn on batch mode and choose a folder.",
        "no_txt": "No .txt point clouds were found in that folder.",
        "reload_busy": "A render is running. Wait for it to finish, or cancel it first.",
        "reload_need_file": "In single-file mode, choose a point cloud and finish one render first.",
        "reload_missing": "TIFF not found. Render first, or check the path.",
        "reload_ok": "🔄 Preview reloaded from TIFF",
        "reload_fail": "⚠️ Could not load preview: {error}",
        "variant_fail": "⚠️ Could not set the Mitsuba variant: {error}\nFalling back to 'scalar_rgb'.",
        "variant_fail2": "⚠️ Fallback variant also failed: {error}",
        "batch_noise_hint": "Batch noise files are matched by name, for example clean/a.txt with noise/a.txt.",
        "noise_cleared": "Noise file and folder cleared.",
        "label_cleared": "Label file cleared.",
        "output_reset": "Manual output path cleared. XML paths are automatic again.",
        "read_cloud_fail": "⚠️ Could not read the point cloud, so the z range was not updated: {error}",
        "label_read_fail": "⚠️ Could not read the label file: {error}",
        "label_reload_fail": "⚠️ Could not reread labels with the current settings: {error}",
        "cmap_loaded": "🎨 Loaded custom colormap: {path}",
        "color_updated": "🎨 Updated class {index} color: {hex}",
        "mode_multi": "multiclass",
        "mode_binary": "binary",
        "batch_labels": "🏷️ Batch mode reads {mode} labels from column {col} of every file.",
        "batch_ignore_label": "⚠️ Batch mode ignores a single label file. Enable “Labels in point cloud” and set the column.",
        "batch_noise_dir": "🌫️ Batch noise folder: {path}. Noise file names must match the clean clouds.",
        "batch_ignore_noise": "⚠️ Batch mode does not use a single noise file. Choose a noise folder whose file names match the clean clouds.",
        "batch_list": "📁 Batch render of {n} files:",
        "noise_missing": "⚠️ No noise file matches {name} (names must match). This file will render without noise.",
        "batch_start": "Starting batch render...",
        "batch_done": "🎉 Batch render finished. To view one result, switch to single-file mode and load the TIFF preview.",
        "single_start": "Starting render (single file)...",
        "single_done": "🎉 Single-file render finished.",
        "rendering_file": "📄 Rendering file: {path}",
        "noise_added": "🌫️ Noise added: {path}",
        "noise_fail": "⚠️ Could not read noise: {error}",
        "labels_from_col": "🏷️ Labels read from column {col} ({n} classes)",
        "labels_from_file": "🏷️ Label file loaded: {path} ({n} classes)",
        "labels_fallback": "⚠️ Could not read the label file. Falling back to the colormap: {error}",
        "writing_tiff": "Writing TIFF...",
        "render_done_files": "✅ Render finished:\n{xml}\n{exr}\n{tiff}",
        "status_rendering": "Rendering {i}/{total}: {name}",
        "render_fail": "❌ Render failed: {path} - {error}",
        "render_abort": "❌ Render stopped: {error}",
        "proc_cancel": "⏹️ Cancelled. Stopping the render process.",
        "proc_crash": "❌ Render process exited unexpectedly (code {code}).{detail}",
        "proc_cancelled": "⏹️ Render cancelled.",
        "proc_abort": "❌ Render task stopped: {error}",
        "err_empty_labels": "Labels are empty.",
        "err_binary": "Binary mode found {n} classes. Enable “Multiclass”.",
        "err_empty_cloud": "The point cloud file is empty.",
        "err_need_xyz": "The point cloud needs at least x, y, and z columns.",
        "err_col_start": "The label column is counted from 1.",
        "err_col_missing": "“Labels in point cloud” is on, but the file has only {ncols} columns, so column {col} cannot be read.",
        "err_count_mismatch": "Label count {nlabels} does not match point count {npoints}.",
        "err_cmap_rows": "A custom colormap needs at least two RGB rows.",
        "err_label_mismatch": "The number of labels does not match the number of points.",
        "err_tiff_shape": "Unrecognized image shape: {shape}",
        "err_tiff_hw": "TIFF needs an H×W×3 image",
        "err_tiff_type": "Unsupported TIFF field type: {typ}",
        "err_tiff_endian": "Only little-endian uncompressed TIFF is supported",
        "err_tiff_tag": "TIFF is missing field {tag}",
        "err_tiff_compress": "Only uncompressed chunky TIFF is supported",
        "err_tiff_strip": "Only single-strip TIFF is supported",
        "err_tiff_format": "Unsupported TIFF pixel format: bits={bits} format={sample_format}",
    },
}


def set_ui_lang(lang):
    global _UI_LANG
    _UI_LANG = "en" if lang == "en" else "zh"


def tr(key, **kwargs):
    table = _UI_TEXT.get(_UI_LANG, _UI_TEXT["zh"])
    text = table.get(key)
    if text is None:
        text = _UI_TEXT["zh"].get(key, key)
    return text.format(**kwargs) if kwargs else text


# =======================================
# 原始渲染与工具函数
# =======================================


def sample_with_radius(points, radius):
    points = np.asarray(points)
    mask = np.ones(points.shape[0], dtype=bool)
    samples = []
    r2 = radius * radius

    for i in range(points.shape[0]):
        if mask[i]:
            d2 = np.sum((points - points[i]) ** 2, axis=1)
            mask[d2 <= r2] = False
            mask[i] = True  # keep current point
            samples.append(points[i])

    return np.asarray(samples) if samples else np.empty((0, points.shape[1]))

def PoissonDiskSampling(points, width=24, height=24, depth=0, target_num_points=1000):
    low = 0.0
    high = np.linalg.norm([width, height, depth])
    best_samples = points.copy()

    while low <= high:
        mid = (low + high) / 2
        samples = sample_with_radius(points, mid)
        if samples.shape[0] < target_num_points:
            high = mid - 1e-5
        else:
            best_samples = samples
            low = mid + 1e-5

    # 若采样点数不足，则补点（添加微扰，避免重复）
    if best_samples.shape[0] < target_num_points:
        needed = target_num_points - best_samples.shape[0]
        idx = np.random.randint(0, best_samples.shape[0], size=needed)
        jitter = 1e-4 * np.random.randn(needed, 3)
        best_samples = np.vstack((best_samples, best_samples[idx] + jitter))

    # 若超出，则随机采样
    if best_samples.shape[0] > target_num_points:
        idx = np.random.permutation(best_samples.shape[0])[:target_num_points]
        best_samples = best_samples[idx]

    return best_samples

def normalize_pointcloud(v):
    center = v.mean(axis=0, keepdims=True)
    v = v - center
    scale = (1 / np.abs(v).max()) * 0.999999
    v = v * scale
    return v, center, scale


def bitmap_to_rgb(bitmap):
    """Mitsuba 位图转为 H×W×3 float32。不经过 OpenEXR / imageio。"""
    array = np.asarray(bitmap)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(tr("err_tiff_shape", shape=array.shape))
    return np.ascontiguousarray(array, dtype=np.float32)


def _tiff_short(value):
    return int(value).to_bytes(2, "little")


def _tiff_long(value):
    return int(value).to_bytes(4, "little")


def write_float_tiff(path, rgb):
    """写未压缩的小端 float32 RGB TIFF。

    打包后的 imageio / imagecodecs 会和 Mitsuba 的动态库冲突并访问冲突，
    所以这里不用第三方图像库。
    """
    rgb = np.ascontiguousarray(rgb, dtype="<f4")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(tr("err_tiff_hw"))
    height, width, _ = rgb.shape
    data = rgb.tobytes()

    tag_count = 13
    ifd_bytes = 2 + tag_count * 12 + 4
    extra_start = 8 + ifd_bytes
    bits_off = extra_start
    format_off = bits_off + 6
    xres_off = format_off + 6
    yres_off = xres_off + 8
    image_off = yres_off + 8
    if image_off % 4:
        image_off += 4 - (image_off % 4)

    def entry(tag, typ, count, value):
        return _tiff_short(tag) + _tiff_short(typ) + _tiff_long(count) + value

    entries = [
        entry(256, 4, 1, _tiff_long(width)),
        entry(257, 4, 1, _tiff_long(height)),
        entry(258, 3, 3, _tiff_long(bits_off)),
        entry(259, 3, 1, _tiff_short(1) + b"\x00\x00"),
        entry(262, 3, 1, _tiff_short(2) + b"\x00\x00"),
        entry(273, 4, 1, _tiff_long(image_off)),
        entry(277, 3, 1, _tiff_short(3) + b"\x00\x00"),
        entry(278, 4, 1, _tiff_long(height)),
        entry(279, 4, 1, _tiff_long(len(data))),
        entry(282, 5, 1, _tiff_long(xres_off)),
        entry(283, 5, 1, _tiff_long(yres_off)),
        entry(284, 3, 1, _tiff_short(1) + b"\x00\x00"),
        entry(339, 3, 3, _tiff_long(format_off)),
    ]
    ifd = _tiff_short(len(entries)) + b"".join(entries) + _tiff_long(0)
    extra = bytearray(image_off - extra_start)
    extra[0:6] = b"".join(_tiff_short(v) for v in (32, 32, 32))
    extra[format_off - extra_start:format_off - extra_start + 6] = b"".join(
        _tiff_short(v) for v in (3, 3, 3)
    )
    extra[xres_off - extra_start:xres_off - extra_start + 8] = _tiff_long(72) + _tiff_long(1)
    extra[yres_off - extra_start:yres_off - extra_start + 8] = _tiff_long(72) + _tiff_long(1)

    with open(path, "wb") as handle:
        handle.write(b"II" + _tiff_short(42) + _tiff_long(8))
        handle.write(ifd)
        handle.write(extra)
        handle.write(data)


def _tiff_values(blob, typ, count, raw):
    if typ == 3:
        if count == 1:
            return [int.from_bytes(raw[:2], "little")]
        offset = int.from_bytes(raw, "little")
        return [
            int.from_bytes(blob[offset + 2 * i:offset + 2 * i + 2], "little")
            for i in range(count)
        ]
    if typ == 4:
        if count == 1:
            return [int.from_bytes(raw, "little")]
        offset = int.from_bytes(raw, "little")
        return [
            int.from_bytes(blob[offset + 4 * i:offset + 4 * i + 4], "little")
            for i in range(count)
        ]
    raise ValueError(tr("err_tiff_type", typ=typ))


def read_tiff(path):
    """读取本程序写出的未压缩 RGB TIFF（float32 或 uint8）。"""
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 8 or blob[:2] != b"II" or int.from_bytes(blob[2:4], "little") != 42:
        raise ValueError(tr("err_tiff_endian"))
    offset = int.from_bytes(blob[4:8], "little")
    count = int.from_bytes(blob[offset:offset + 2], "little")
    tags = {}
    pos = offset + 2
    for _ in range(count):
        tag = int.from_bytes(blob[pos:pos + 2], "little")
        typ = int.from_bytes(blob[pos + 2:pos + 4], "little")
        number = int.from_bytes(blob[pos + 4:pos + 8], "little")
        tags[tag] = (typ, number, blob[pos + 8:pos + 12])
        pos += 12

    def values(tag, default=None):
        if tag not in tags:
            if default is None:
                raise ValueError(tr("err_tiff_tag", tag=tag))
            return default
        typ, number, raw = tags[tag]
        return _tiff_values(blob, typ, number, raw)

    width = values(256)[0]
    height = values(257)[0]
    bits = values(258)
    if values(259)[0] != 1 or values(284, [1])[0] != 1:
        raise ValueError(tr("err_tiff_compress"))
    samples = values(277, [1])[0]
    sample_format = values(339, [1] * samples)
    strips = values(273)
    counts = values(279)
    if len(strips) != 1 or len(counts) != 1:
        raise ValueError(tr("err_tiff_strip"))
    start = strips[0]
    raw = blob[start:start + counts[0]]
    if bits == [32, 32, 32] and sample_format[:3] == [3, 3, 3]:
        array = np.frombuffer(raw, dtype="<f4").reshape(height, width, 3)
    elif bits == [8, 8, 8]:
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    else:
        raise ValueError(tr("err_tiff_format", bits=bits, sample_format=sample_format))
    return np.array(array, copy=True)


def to_tiff(exr_img, out_path):
    write_float_tiff(out_path, bitmap_to_rgb(mi.Bitmap(exr_img)))


def make_colorbar_pixmap(cmap, width=260, height=24):
    grad = np.linspace(0, 1, width, dtype=np.float32)
    grad = np.tile(grad, (height, 1))
    rgb = cmap(grad)[..., :3]
    return numpy_rgb_to_qpixmap(rgb)


def class_color_palette(labels, cmap, binary_colors=None, multiclass=False):
    unique_values = np.unique(np.asarray(labels).reshape(-1))
    sorted_vals = sorted(unique_values.tolist())
    n = len(sorted_vals)
    if n == 0:
        raise ValueError(tr("err_empty_labels"))
    if not multiclass and n > 2:
        raise ValueError(tr("err_binary", n=n))

    if multiclass:
        if n == 1:
            palette = [tuple(float(c) for c in cmap(0.5)[:3])]
        else:
            palette = [tuple(float(c) for c in cmap(i / (n - 1))[:3]) for i in range(n)]
    else:
        base = list(binary_colors) if binary_colors is not None else [(1.0, 0.2, 0.2), (0.2, 0.45, 1.0)]
        palette = [base[min(i, len(base) - 1)] for i in range(n)]

    label_map = {value: idx for idx, value in enumerate(sorted_vals)}
    return label_map, palette


def make_class_legend_pixmap(palette, width=360, height=24):
    palette = np.asarray(palette, dtype=np.float32)
    n = max(1, palette.shape[0])
    img = np.zeros((height, width, 3), dtype=np.float32)
    edges = np.linspace(0, width, n + 1).astype(int)
    for i in range(n):
        img[:, edges[i]:edges[i + 1]] = palette[i]
    return numpy_rgb_to_qpixmap(img)


def make_solid_color_pixmap(rgb, width=48, height=24):
    rgb = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    img = np.tile(rgb, (height, width, 1))
    return numpy_rgb_to_qpixmap(img)


def tonemap_to_8bit(rgb):
    """将浮点 HDR/线性图简单 tone-map 到 8bit 以供预览（百分位截断）。"""
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32)
    lo, hi = np.percentile(rgb, [1, 99])
    if hi <= lo:
        lo, hi = float(rgb.min()), float(rgb.max())
    rgb = np.clip((rgb - lo) / max(1e-8, (hi - lo)), 0.0, 1.0)
    rgb = (rgb ** (1/2.2))
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def numpy_rgb_to_qpixmap(rgb):
    """rgb: [H, W, 3] 0..1 或 0..255"""
    if rgb.dtype != np.uint8:
        rgb = tonemap_to_8bit(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# 通过可调 lookat origin 生成 xml 头
def generate_xml_head(origin):
    ox, oy, oz = origin
    return f"""
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{ox},{oy},{oz}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="3200"/>
            <integer name="height" value="2400"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""


xml_ball_segment = """
    <shape type="sphere">
        <float name="radius" value="{}"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

class RenderCancelled(Exception):
    """用户在渲染完成前取消了任务。"""


def _raise_if_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise RenderCancelled()


def generate_xml_tail(floor_z):
    return f"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="{floor_z}"/>
        </transform>
    </shape>
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


def save_mistuba(points_txt, noise_points_txt=None, xml_filename="scene.xml",
                 ball_size=0.011, ball_size_noisy=0.004, camera_origin=(-3, -3, 3),
                 floor_z=-0.15, cmap=None, labels=None, label_colors=None,
                 multiclass=False, cancel_check=None, run_render=None):
    """将点云写入 Mitsuba XML 并渲染为 EXR。

    cancel_check 返回 True 时中止。run_render(scene) 用于替换 mi.render，
    以便在渲染过程中响应取消。
    """
    if cmap is None:
        from matplotlib import cm
        cmap = cm.get_cmap('viridis')

    if label_colors is None:
        label_colors = ((1.0, 0.2, 0.2), (0.2, 0.45, 1.0))

    xml_segments = [generate_xml_head(camera_origin)]

    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != points_txt.shape[0]:
            raise ValueError(tr("err_label_mismatch"))
        label_map, palette = class_color_palette(
            labels, cmap, binary_colors=label_colors, multiclass=multiclass
        )
    else:
        # 使用 z 值归一化作为 feature
        z = points_txt[:, 2]
        zmin, zmax = z.min(), z.max()
        feature = (z - zmin) / max(1e-12, (zmax - zmin))

    point_count = points_txt.shape[0]
    for i in range(point_count):
        if (i & 2047) == 0:
            _raise_if_cancelled(cancel_check)
        if labels is not None:
            class_idx = label_map[labels[i]]
            color = palette[class_idx]
        else:
            color = cmap(feature[i])[:3]
        xml_segments.append(xml_ball_segment.format(
            ball_size,
            points_txt[i, 0], points_txt[i, 1], points_txt[i, 2],
            color[0], color[1], color[2]
        ))

    if noise_points_txt is not None:
        noise_count = noise_points_txt.shape[0]
        for i in range(noise_count):
            if (i & 2047) == 0:
                _raise_if_cancelled(cancel_check)
            color = (0.7, 0.7, 0.7)
            xml_segments.append(xml_ball_segment.format(
                ball_size_noisy,
                noise_points_txt[i, 0], noise_points_txt[i, 1], noise_points_txt[i, 2],
                color[0], color[1], color[2]
            ))

    _raise_if_cancelled(cancel_check)
    xml_segments.append(generate_xml_tail(floor_z))

    with open(xml_filename, 'w') as f:
        f.write("".join(xml_segments))

    _raise_if_cancelled(cancel_check)
    scene = mi.load_file(xml_filename)
    _raise_if_cancelled(cancel_check)
    if run_render is None:
        img = mi.render(scene)
    else:
        img = run_render(scene)
    _raise_if_cancelled(cancel_check)
    bitmap = mi.Bitmap(img)
    bitmap.write(xml_filename.replace(".xml", ".exr"))
    return bitmap_to_rgb(bitmap)


# =======================================
# 色轴：内置与自定义
# =======================================

BUILTIN_CMAPS = ["viridis", "jet", "magma", "plasma", "cividis", "turbo"]

def load_custom_cmap_from_txt(txt_path):
    """
    支持格式：
    - 每行 3 个数：R G B，允许空格或逗号分隔。
    - 数值范围可为 0..1 或 0..255（自动判别）。
    - 至少 2 行。
    """
    data = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            line = line.replace(",", " ")
            parts = [p for p in line.split() if p]
            if len(parts) < 3:
                continue
            r, g, b = map(float, parts[:3])
            data.append([r, g, b])
    if len(data) < 2:
        raise ValueError(tr("err_cmap_rows"))

    arr = np.asarray(data, dtype=np.float32)
    if np.nanmax(arr) > 1.0:
        arr = np.clip(arr / 255.0, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)
    return LinearSegmentedColormap.from_list("custom_txt_cmap", arr.tolist())


def get_builtin_cmap(name):
    from matplotlib import cm
    try:
        return cm.get_cmap(name)
    except Exception:
        return cm.get_cmap("viridis")


def validate_labels(labels, multiclass=False):
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        raise ValueError(tr("err_empty_labels"))
    unique_values = np.unique(labels)
    if not multiclass and unique_values.size > 2:
        raise ValueError(tr("err_binary", n=int(unique_values.size)))
    return labels


def load_labels_from_txt(label_path, multiclass=False):
    return validate_labels(np.loadtxt(label_path), multiclass=multiclass)


def load_pointcloud(path, with_labels=False, multiclass=False, label_col=4):
    data = np.asarray(np.loadtxt(path))
    if data.size == 0:
        raise ValueError(tr("err_empty_cloud"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(tr("err_need_xyz"))

    points = data[:, :3]
    labels = None
    if with_labels:
        col = int(label_col)
        if col < 1:
            raise ValueError(tr("err_col_start"))
        idx = col - 1
        if data.shape[1] <= idx:
            raise ValueError(tr("err_col_missing", ncols=data.shape[1], col=col))
        labels = validate_labels(data[:, idx], multiclass=multiclass)
        if labels.shape[0] != points.shape[0]:
            raise ValueError(tr("err_count_mismatch", nlabels=labels.shape[0], npoints=points.shape[0]))
    return points, labels


def mitsuba_package_dir():
    return os.path.dirname(os.path.abspath(mi.__file__))


def collect_mitsuba_search_paths():
    paths = []
    try:
        fr = mi.Thread.thread().file_resolver()
        paths.extend(str(fr[i]) for i in range(len(fr)))
    except Exception:
        pass
    pkg = mitsuba_package_dir()
    for extra in (pkg, os.path.join(pkg, "plugins")):
        if extra not in paths:
            paths.append(extra)
    return paths


def apply_mitsuba_search_paths(paths):
    # 不要调用 register_external_thread。在这个 CUDA 构建上，注册外部线程后再
    # load_file 会直接访问冲突；后台线程里加载场景同样会崩溃。渲染必须留在进程主线程。
    thread = mi.Thread.thread()
    fr = mi.FileResolver()
    pkg = mitsuba_package_dir()
    plugins = os.path.join(pkg, "plugins")
    fr.prepend(plugins)
    fr.append(pkg)
    for p in paths or []:
        if p and os.path.abspath(p) not in (os.path.abspath(plugins), os.path.abspath(pkg)):
            fr.append(str(p))
    thread.set_file_resolver(fr)


def execute_render_job(job, log, cancel_check=None, run_render=None):
    """在工作线程中执行单个渲染任务。log 为可调用对象。"""
    _raise_if_cancelled(cancel_check)
    file_path = job["file_path"]
    log(tr("rendering_file", path=file_path))
    terrain, embedded_labels = load_pointcloud(
        file_path,
        with_labels=job["use_embedded_labels"],
        multiclass=job["multiclass"],
        label_col=job["label_col"],
    )
    terrain = normalize_pointcloud(terrain)[0]
    zmin = float(terrain[:, 2].min())
    zmax = float(terrain[:, 2].max())

    noise = None
    noise_path = job.get("noise_path")
    if noise_path:
        try:
            noise = np.loadtxt(noise_path)[:, :3]
            noise = normalize_pointcloud(noise)[0]
            log(tr("noise_added", path=noise_path))
        except Exception as e:
            log(tr("noise_fail", error=e))

    labels = None
    if job["use_embedded_labels"]:
        labels = embedded_labels
        n_cls = int(np.unique(labels).size) if labels is not None else 0
        log(tr("labels_from_col", col=job["label_col"], n=n_cls))
    elif job.get("label_file"):
        try:
            labels = load_labels_from_txt(job["label_file"], multiclass=job["multiclass"])
            if labels.shape[0] != terrain.shape[0]:
                raise ValueError(tr("err_count_mismatch", nlabels=labels.shape[0], npoints=terrain.shape[0]))
            n_cls = int(np.unique(labels).size)
            log(tr("labels_from_file", path=job["label_file"], n=n_cls))
        except Exception as e:
            log(tr("labels_fallback", error=e))
            labels = None

    _raise_if_cancelled(cancel_check)
    xml_path = job["xml_path"]
    exr_path = xml_path.replace(".xml", ".exr")
    tiff_path = xml_path.replace(".xml", ".tiff")
    rgb = save_mistuba(
        terrain,
        noise,
        xml_path,
        job["ball_size"],
        job["noise_ball_size"],
        job["camera_origin"],
        job["floor_z"],
        cmap=job["cmap"],
        labels=labels,
        label_colors=job["label_colors"],
        multiclass=job["multiclass"],
        cancel_check=cancel_check,
        run_render=run_render,
    )
    _raise_if_cancelled(cancel_check)
    log(tr("writing_tiff"))
    write_float_tiff(tiff_path, rgb)
    log(tr("render_done_files", xml=xml_path, exr=exr_path, tiff=tiff_path))

    preview_img = rgb if job.get("preview") else None

    return {
        "zmin": zmin,
        "zmax": zmax,
        "labels": labels,
        "preview_img": preview_img,
    }


class SampledCmap:
    """把色图采样成数组，方便传给独立渲染进程。"""

    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float64)

    def __call__(self, x):
        samples = self.samples
        last = samples.shape[0] - 1
        if last <= 0:
            color = samples[0]
        else:
            x = float(x)
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            pos = x * last
            index = int(pos)
            if index >= last:
                color = samples[-1]
            else:
                mix = pos - index
                color = samples[index] * (1.0 - mix) + samples[index + 1] * mix
        return (float(color[0]), float(color[1]), float(color[2]), 1.0)


def pack_cmap(cmap, samples=256):
    xs = np.linspace(0.0, 1.0, samples)
    colors = np.stack([np.asarray(cmap(float(x))[:3], dtype=np.float64) for x in xs], axis=0)
    return SampledCmap(colors)


def _append_render_line(log_path, line):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_jobs_subprocess(jobs, variant, search_paths, log_path):
    """在独立进程的主线程里渲染。

    这个 Mitsuba CUDA 构建不是线程安全的：只要进程里还有别的线程，
    load_file 就会访问冲突。所以这里不能再开线程或队列。
    """
    def emit(kind, text):
        _append_render_line(log_path, kind + "\t" + text.replace("\n", "\n" + kind + "\t"))

    try:
        try:
            mi.set_variant(variant)
        except Exception:
            mi.set_variant("scalar_rgb")
        # 父进程传来的搜索路径指向它自己的解压目录，子进程不能用。
        del search_paths
        if jobs:
            set_ui_lang(jobs[0].get("lang", "zh"))
        apply_mitsuba_search_paths(collect_mitsuba_search_paths())
        total = len(jobs)
        for i, job in enumerate(jobs, 1):
            name = os.path.basename(job["file_path"])
            emit("STATUS", tr("status_rendering", i=i, total=total, name=name))
            emit("LOG", f"—— [{i}/{total}] ——")
            try:
                result = execute_render_job(job, lambda message: emit("LOG", message))
            except Exception as e:
                emit("LOG", tr("render_fail", path=job["file_path"], error=e))
                continue
            emit("Z", f"{float(result['zmin'])}\t{float(result['zmax'])}")
            if result["labels"] is not None:
                label_path = f"{log_path}.labels{i}.npy"
                np.save(label_path, result["labels"])
                emit("LABELS", label_path)
            if result["preview_img"] is not None:
                emit("PREVIEW", job["xml_path"].replace(".xml", ".tiff"))
    except Exception as e:
        emit("LOG", tr("render_abort", error=e))
    emit("DONE", "1")


def run_render_job_cli(job_path, log_path):
    with open(job_path, "rb") as handle:
        payload = pickle.load(handle)
    render_jobs_subprocess(
        payload["jobs"],
        payload["variant"],
        payload["search_paths"],
        log_path,
    )


def _render_process_command(job_path, log_path):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--render-job", job_path, log_path]
    return [sys.executable, os.path.abspath(__file__), "--render-job", job_path, log_path]


def _render_subprocess_env():
    """子进程不要继承父进程解压目录里的 DLL 路径。"""
    env = os.environ.copy()
    mei = getattr(sys, "_MEIPASS", None)
    if not mei:
        return env
    root = os.path.normcase(os.path.abspath(mei))
    kept = []
    for part in env.get("PATH", "").split(os.pathsep):
        if not part:
            continue
        try:
            norm = os.path.normcase(os.path.abspath(part))
        except Exception:
            kept.append(part)
            continue
        if norm == root or norm.startswith(root + os.sep):
            continue
        kept.append(part)
    env["PATH"] = os.pathsep.join(kept)
    for key in list(env):
        if key.startswith("_PYI") or key.startswith("_MEI"):
            env.pop(key, None)
    return env


class RenderWorker(QObject):
    log_msg = pyqtSignal(str)
    status = pyqtSignal(str)
    z_range = pyqtSignal(float, float)
    labels_ready = pyqtSignal(object)
    preview_ready = pyqtSignal(object)
    finished = pyqtSignal(bool)

    def __init__(self, jobs, variant, search_paths=None):
        super().__init__()
        self.jobs = jobs
        self.variant = variant
        self.search_paths = list(search_paths or [])
        self._cancel_event = threading.Event()
        self._proc = None

    def request_cancel(self):
        """结束独立渲染进程。CUDA 渲染不能从别的线程里安全打断。"""
        self._cancel_event.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _handle_line(self, line):
        if "\t" not in line:
            return
        kind, payload = line.split("\t", 1)
        if kind == "LOG":
            self.log_msg.emit(payload)
        elif kind == "STATUS":
            self.status.emit(payload)
        elif kind == "Z":
            zmin, zmax = payload.split("\t")
            self.z_range.emit(float(zmin), float(zmax))
        elif kind == "LABELS" and os.path.exists(payload):
            self.labels_ready.emit(np.load(payload))
        elif kind == "PREVIEW" and os.path.exists(payload):
            self.preview_ready.emit(read_tiff(payload))

    def _consume_log(self, log_path, offset, pending):
        if not os.path.exists(log_path):
            return offset, pending
        with open(log_path, "r", encoding="utf-8") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
        if not chunk:
            return offset, pending
        pending += chunk
        lines = pending.split("\n")
        pending = lines.pop()
        for line in lines:
            if line:
                self._handle_line(line)
        return offset, pending

    def run(self):
        cancelled = False
        work_dir = tempfile.mkdtemp(prefix="mistuba-render-")
        job_path = os.path.join(work_dir, "job.pkl")
        log_path = os.path.join(work_dir, "log.txt")
        with open(job_path, "wb") as handle:
            pickle.dump(
                {"jobs": self.jobs, "variant": self.variant, "search_paths": self.search_paths},
                handle,
            )
        error_path = os.path.join(work_dir, "stderr.txt")
        error_handle = open(error_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            _render_process_command(job_path, log_path),
            stdout=error_handle,
            stderr=subprocess.STDOUT,
            env=_render_subprocess_env(),
        )
        self._proc = proc
        offset = 0
        pending = ""
        try:
            while True:
                if self._cancel_event.is_set():
                    cancelled = True
                    if proc.poll() is None:
                        proc.terminate()
                    self.log_msg.emit(tr("proc_cancel"))
                    break
                offset, pending = self._consume_log(log_path, offset, pending)
                if proc.poll() is not None:
                    time.sleep(0.05)
                    offset, pending = self._consume_log(log_path, offset, pending)
                    if pending:
                        self._handle_line(pending)
                    if proc.returncode not in (0, None) and not cancelled:
                        detail = ""
                        error_handle.flush()
                        if os.path.exists(error_path):
                            with open(error_path, "r", encoding="utf-8", errors="replace") as handle:
                                tail = handle.read()[-500:].strip()
                            if tail:
                                detail = "\n" + tail
                        self.log_msg.emit(tr("proc_crash", code=proc.returncode, detail=detail))
                    break
                time.sleep(0.1)
        except Exception as e:
            if self._cancel_event.is_set():
                cancelled = True
                self.log_msg.emit(tr("proc_cancelled"))
            else:
                self.log_msg.emit(tr("proc_abort", error=e))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            error_handle.close()
            self._proc = None
            self.finished.emit(cancelled)


def rgb_to_hex(rgb):
    rgb255 = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0) * 255.0
    r, g, b = rgb255.astype(np.uint8).tolist()
    return f"#{r:02X}{g:02X}{b:02X}"


def qcolor_to_rgb_tuple(color):
    return (color.redF(), color.greenF(), color.blueF())


def _style_path_label(label):
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setStyleSheet("color: #444; font-size: 12px;")


# =======================================
# UI 实现（增强版）
# =======================================

class RenderUI(QWidget):
    def __init__(self):
        super().__init__()

        self.settings = QSettings("GitHubCopilot", "MistubaRenderUI")
        saved_lang = self.settings.value("ui_lang", "zh", type=str)
        self.lang = "en" if saved_lang == "en" else "zh"
        set_ui_lang(self.lang)

        self.setWindowTitle("MISTUBA RENDER (version:25/11/10/Author:sly)")
        self.setMinimumSize(960, 640)
        self.resize(1280, 820)
        self.setStyleSheet("""
            QGroupBox { font-weight: bold; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton#btnStart { min-height: 34px; font-weight: bold; }
        """)

        # 文件路径变量
        self.terrain_file = ""
        self.terrain_dir = ""    # 新增：点云文件夹
        self.noise_file = ""
        self.noise_dir = ""
        self.label_file = ""
        self.output_xml = ""
        self.output_xml_manual = False

        # 色轴
        self.cmap = get_builtin_cmap("viridis")
        self.custom_cmap_path = None
        self.label_colors = [(1.0, 0.2, 0.2), (0.2, 0.45, 1.0)]
        self._label_class_count = 0

        self.zmin = None
        self.zmax = None
        self._preview_pix = None

        # ========== 左侧：设置 ==========
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(10, 10, 8, 10)
        left.setSpacing(8)

        self.cloud_box = QGroupBox(tr("cloud_group"))
        cloud_layout = QVBoxLayout()
        cloud_btns = QHBoxLayout()
        self.btn_terrain = QPushButton(tr("btn_file"))
        self.btn_terrain.clicked.connect(self.choose_terrain)
        cloud_btns.addWidget(self.btn_terrain)
        self.btn_terrain_dir = QPushButton(tr("btn_folder"))
        self.btn_terrain_dir.clicked.connect(self.choose_terrain_dir)
        cloud_btns.addWidget(self.btn_terrain_dir)
        self.batch_check = QCheckBox(tr("batch_check"))
        self.batch_check.setChecked(False)
        cloud_btns.addWidget(self.batch_check)
        cloud_btns.addStretch(1)
        cloud_layout.addLayout(cloud_btns)
        self.lbl_terrain = QLabel(tr("file_none"))
        self.lbl_terrain_dir = QLabel(tr("folder_none"))
        _style_path_label(self.lbl_terrain)
        _style_path_label(self.lbl_terrain_dir)
        cloud_layout.addWidget(self.lbl_terrain)
        cloud_layout.addWidget(self.lbl_terrain_dir)
        self.cloud_box.setLayout(cloud_layout)
        left.addWidget(self.cloud_box)

        self.label_box = QGroupBox(tr("label_group"))
        label_layout = QVBoxLayout()
        embed_row = QHBoxLayout()
        self.labels_in_cloud_check = QCheckBox(tr("labels_in_cloud"))
        self.labels_in_cloud_check.setChecked(False)
        self.labels_in_cloud_check.setToolTip(tr("labels_in_cloud_tip"))
        embed_row.addWidget(self.labels_in_cloud_check)
        self.lbl_col = QLabel(tr("col"))
        embed_row.addWidget(self.lbl_col)
        self.label_col_box = QSpinBox()
        self.label_col_box.setRange(1, 64)
        self.label_col_box.setValue(4)
        self.label_col_box.setToolTip(tr("col_tip"))
        embed_row.addWidget(self.label_col_box)
        self.multiclass_check = QCheckBox(tr("multiclass"))
        self.multiclass_check.setChecked(False)
        self.multiclass_check.setToolTip(tr("multiclass_tip"))
        embed_row.addWidget(self.multiclass_check)
        embed_row.addStretch(1)
        label_layout.addLayout(embed_row)

        label_file_row = QHBoxLayout()
        self.btn_label = QPushButton(tr("btn_label_file"))
        self.btn_label.clicked.connect(self.choose_label)
        label_file_row.addWidget(self.btn_label)
        self.btn_clear_label = QPushButton(tr("btn_clear"))
        self.btn_clear_label.clicked.connect(self.clear_label)
        label_file_row.addWidget(self.btn_clear_label)
        label_file_row.addStretch(1)
        label_layout.addLayout(label_file_row)
        self.lbl_label = QLabel(tr("label_none"))
        _style_path_label(self.lbl_label)
        label_layout.addWidget(self.lbl_label)
        self.label_box.setLayout(label_layout)
        left.addWidget(self.label_box)

        self.noise_box = QGroupBox(tr("noise_group"))
        noise_layout = QVBoxLayout()
        noise_row = QHBoxLayout()
        self.btn_noise = QPushButton(tr("btn_file"))
        self.btn_noise.clicked.connect(self.choose_noise)
        noise_row.addWidget(self.btn_noise)
        self.btn_noise_dir = QPushButton(tr("btn_folder"))
        self.btn_noise_dir.setToolTip(tr("noise_dir_tip"))
        self.btn_noise_dir.clicked.connect(self.choose_noise_dir)
        noise_row.addWidget(self.btn_noise_dir)
        self.btn_clear_noise = QPushButton(tr("btn_clear"))
        self.btn_clear_noise.clicked.connect(self.clear_noise)
        noise_row.addWidget(self.btn_clear_noise)
        noise_row.addStretch(1)
        noise_layout.addLayout(noise_row)
        self.lbl_noise = QLabel(tr("label_none"))
        _style_path_label(self.lbl_noise)
        noise_layout.addWidget(self.lbl_noise)
        self.noise_box.setLayout(noise_layout)
        left.addWidget(self.noise_box)

        self.param_box = QGroupBox(tr("param_group"))
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ball_size_box = QDoubleSpinBox()
        self.ball_size_box.setDecimals(6)
        self.ball_size_box.setRange(0.0001, 1.0)
        self.ball_size_box.setSingleStep(0.001)
        self.ball_size_box.setValue(0.011)
        form.addRow(tr("ball_radius"), self.ball_size_box)
        self.lbl_ball = form.labelForField(self.ball_size_box)

        self.noise_ball_size_box = QDoubleSpinBox()
        self.noise_ball_size_box.setDecimals(6)
        self.noise_ball_size_box.setRange(0.0001, 1.0)
        self.noise_ball_size_box.setSingleStep(0.001)
        self.noise_ball_size_box.setValue(0.004)
        form.addRow(tr("noise_radius"), self.noise_ball_size_box)
        self.lbl_noise_radius = form.labelForField(self.noise_ball_size_box)

        self.floor_z_box = QDoubleSpinBox()
        self.floor_z_box.setDecimals(4)
        self.floor_z_box.setRange(-1000.0, 1000.0)
        self.floor_z_box.setSingleStep(0.01)
        self.floor_z_box.setValue(-0.15)
        form.addRow(tr("floor_z"), self.floor_z_box)
        self.lbl_floor = form.labelForField(self.floor_z_box)

        cam_layout = QHBoxLayout()
        self.cam_x = QDoubleSpinBox(); self.cam_x.setRange(-1000, 1000); self.cam_x.setValue(-3.0)
        self.cam_y = QDoubleSpinBox(); self.cam_y.setRange(-1000, 1000); self.cam_y.setValue(-3.0)
        self.cam_z = QDoubleSpinBox(); self.cam_z.setRange(-1000, 1000); self.cam_z.setValue(3.0)
        for w in (self.cam_x, self.cam_y, self.cam_z):
            w.setDecimals(4); w.setSingleStep(0.1)
        cam_layout.addWidget(QLabel("X")); cam_layout.addWidget(self.cam_x)
        cam_layout.addWidget(QLabel("Y")); cam_layout.addWidget(self.cam_y)
        cam_layout.addWidget(QLabel("Z")); cam_layout.addWidget(self.cam_z)
        cam_wrap = QWidget(); cam_wrap.setLayout(cam_layout)
        form.addRow(tr("camera_origin"), cam_wrap)
        self.lbl_camera = form.labelForField(cam_wrap)
        self.param_box.setLayout(form)
        left.addWidget(self.param_box)

        self.color_box = QGroupBox(tr("color_group"))
        color_layout = QVBoxLayout()
        cmap_row = QHBoxLayout()
        self.lbl_cmap_name = QLabel(tr("colormap"))
        cmap_row.addWidget(self.lbl_cmap_name)
        self.cmap_combo = QComboBox()
        for name in BUILTIN_CMAPS:
            self.cmap_combo.addItem(name, name)
        self.cmap_combo.addItem(tr("cmap_custom"), CUSTOM_CMAP_TOKEN)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        cmap_row.addWidget(self.cmap_combo, 1)
        self.btn_load_custom_cmap = QPushButton(tr("btn_load_cmap"))
        self.btn_load_custom_cmap.clicked.connect(self.load_custom_cmap)
        cmap_row.addWidget(self.btn_load_custom_cmap)
        color_layout.addLayout(cmap_row)
        self.lbl_custom_cmap = QLabel(tr("cmap_builtin", name="viridis"))
        _style_path_label(self.lbl_custom_cmap)
        color_layout.addWidget(self.lbl_custom_cmap)

        self.lbl_colorbar_caption = QLabel(tr("caption_gradient"))
        color_layout.addWidget(self.lbl_colorbar_caption)
        self.colorbar_label = QLabel()
        self.colorbar_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.colorbar_label.setFixedHeight(28)
        color_layout.addWidget(self.colorbar_label)

        binary_row = QHBoxLayout()
        self.btn_label0_color = QPushButton(tr("class0"))
        self.btn_label0_color.clicked.connect(lambda: self.choose_label_color(0))
        binary_row.addWidget(self.btn_label0_color)
        self.label0_preview = QLabel("0")
        self.label0_preview.setAlignment(Qt.AlignCenter)
        self.label0_preview.setFixedSize(36, 24)
        binary_row.addWidget(self.label0_preview)
        self.btn_label1_color = QPushButton(tr("class1"))
        self.btn_label1_color.clicked.connect(lambda: self.choose_label_color(1))
        binary_row.addWidget(self.btn_label1_color)
        self.label1_preview = QLabel("1")
        self.label1_preview.setAlignment(Qt.AlignCenter)
        self.label1_preview.setFixedSize(36, 24)
        binary_row.addWidget(self.label1_preview)
        binary_row.addStretch(1)
        color_layout.addLayout(binary_row)
        self.color_box.setLayout(color_layout)
        left.addWidget(self.color_box)

        self.output_box = QGroupBox(tr("output_group"))
        output_layout = QVBoxLayout()
        output_row = QHBoxLayout()
        self.btn_output = QPushButton(tr("btn_xml"))
        self.btn_output.clicked.connect(self.choose_output_xml)
        output_row.addWidget(self.btn_output)
        self.btn_clear_output = QPushButton(tr("btn_auto"))
        self.btn_clear_output.clicked.connect(self.clear_output_xml)
        output_row.addWidget(self.btn_clear_output)
        output_row.addStretch(1)
        output_layout.addLayout(output_row)
        self.lbl_output = QLabel(tr("output_default"))
        _style_path_label(self.lbl_output)
        output_layout.addWidget(self.lbl_output)
        self.output_box.setLayout(output_layout)
        left.addWidget(self.output_box)

        left.addStretch(1)

        btn_start = QPushButton(tr("btn_start"))
        btn_start.setObjectName("btnStart")
        btn_start.clicked.connect(self.render_scene)
        self.btn_start = btn_start
        left.addWidget(btn_start)
        btn_reload_preview = QPushButton(tr("btn_reload"))
        btn_reload_preview.clicked.connect(self.reload_tiff_preview)
        self.btn_reload_preview = btn_reload_preview
        left.addWidget(btn_reload_preview)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(360)
        left_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # ========== 右侧：预览 + 日志 ==========
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(8, 10, 10, 10)
        right.setSpacing(8)

        self.preview_box = QGroupBox(tr("preview_group"))
        pv = QVBoxLayout()
        z_range_layout = QHBoxLayout()
        self.lbl_zmin = QLabel(tr("zmin_empty"))
        self.lbl_zmax = QLabel(tr("zmax_empty"))
        z_range_layout.addWidget(self.lbl_zmin)
        z_range_layout.addStretch(1)
        z_range_layout.addWidget(self.lbl_zmax)
        pv.addLayout(z_range_layout)

        self.render_preview = QLabel(tr("preview_placeholder"))
        self.render_preview.setAlignment(Qt.AlignCenter)
        self.render_preview.setMinimumHeight(240)
        self.render_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.render_preview.setStyleSheet("border: 1px solid #ccc; background: #f7f7f7;")
        pv.addWidget(self.render_preview, 1)
        self.preview_box.setLayout(pv)
        self.preview_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_box = QGroupBox(tr("log_group"))
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        log_layout.addWidget(self.log)
        self.log_box.setLayout(log_layout)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.preview_box)
        right_split.addWidget(self.log_box)
        right_split.setStretchFactor(0, 4)
        right_split.setStretchFactor(1, 1)
        right_split.setChildrenCollapsible(False)
        right.addWidget(right_split)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 860])
        splitter.setChildrenCollapsible(False)

        self.btn_lang = QPushButton("English" if self.lang == "zh" else "中文")
        self.btn_lang.setToolTip(tr("lang_tip"))
        self.btn_lang.setCursor(Qt.PointingHandCursor)
        self.btn_lang.clicked.connect(self.toggle_language)
        lang_bar = QHBoxLayout()
        lang_bar.setContentsMargins(10, 8, 10, 0)
        lang_bar.addStretch(1)
        lang_bar.addWidget(self.btn_lang)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(lang_bar)
        outer.addWidget(splitter)

        self._rendering = False
        self._cancel_requested = False
        self._quit_when_idle = False
        self._render_thread = None
        self._render_worker = None
        self._loading_seconds = 0
        self._build_loading_overlay()

        # 初始化色轴
        self.update_colorbar()
        self.update_label_color_widgets()
        self.bind_settings_signals()
        self.load_settings()

        # Mitsuba 变体提前设置一次
        self._mi_variant = "cuda_ad_rgb"
        try:
            mi.set_variant('cuda_ad_rgb')
        except Exception as e:
            # 如果没有 CUDA 插件，可改用 scalar_rgb 或 llvm_ad_rgb
            self.log.append(tr("variant_fail", error=e))
            try:
                mi.set_variant('scalar_rgb')
                self._mi_variant = "scalar_rgb"
            except Exception as e2:
                self.log.append(tr("variant_fail2", error=e2))
        self._mi_search_paths = collect_mitsuba_search_paths()

    def bind_settings_signals(self):
        self.batch_check.stateChanged.connect(self.save_settings)
        self.ball_size_box.valueChanged.connect(self.save_settings)
        self.noise_ball_size_box.valueChanged.connect(self.save_settings)
        self.floor_z_box.valueChanged.connect(self.save_settings)
        self.cam_x.valueChanged.connect(self.save_settings)
        self.cam_y.valueChanged.connect(self.save_settings)
        self.cam_z.valueChanged.connect(self.save_settings)
        self.cmap_combo.currentTextChanged.connect(self.save_settings)
        self.labels_in_cloud_check.stateChanged.connect(self.on_labels_in_cloud_changed)
        self.label_col_box.valueChanged.connect(self.on_label_col_changed)
        self.multiclass_check.stateChanged.connect(self.on_multiclass_changed)

    def load_settings(self):
        self.batch_check.setChecked(self.settings.value("batch_check", False, type=bool))
        self.labels_in_cloud_check.blockSignals(True)
        self.labels_in_cloud_check.setChecked(self.settings.value("labels_in_cloud", False, type=bool))
        self.labels_in_cloud_check.blockSignals(False)
        self.label_col_box.blockSignals(True)
        self.label_col_box.setValue(self.settings.value("label_col", 4, type=int))
        self.label_col_box.blockSignals(False)
        self.multiclass_check.blockSignals(True)
        self.multiclass_check.setChecked(self.settings.value("multiclass_labels", False, type=bool))
        self.multiclass_check.blockSignals(False)
        self.ball_size_box.setValue(self.settings.value("ball_size", 0.011, type=float))
        self.noise_ball_size_box.setValue(self.settings.value("noise_ball_size", 0.004, type=float))
        self.floor_z_box.setValue(self.settings.value("floor_z", -0.15, type=float))
        self.cam_x.setValue(self.settings.value("camera_x", -3.0, type=float))
        self.cam_y.setValue(self.settings.value("camera_y", -3.0, type=float))
        self.cam_z.setValue(self.settings.value("camera_z", 3.0, type=float))

        terrain_file = self.settings.value("terrain_file", "", type=str)
        terrain_dir = self.settings.value("terrain_dir", "", type=str)
        noise_file = self.settings.value("noise_file", "", type=str)
        noise_dir = self.settings.value("noise_dir", "", type=str)
        label_file = self.settings.value("label_file", "", type=str)
        output_xml = self.settings.value("output_xml", "", type=str)
        output_xml_manual = self.settings.value("output_xml_manual", False, type=bool)
        custom_cmap_path = self.settings.value("custom_cmap_path", "", type=str)
        cmap_name = self.settings.value("cmap_name", "viridis", type=str)

        label0 = self.settings.value("label_color_0", rgb_to_hex(self.label_colors[0]), type=str)
        label1 = self.settings.value("label_color_1", rgb_to_hex(self.label_colors[1]), type=str)
        self.label_colors = [
            qcolor_to_rgb_tuple(QColor(label0)),
            qcolor_to_rgb_tuple(QColor(label1)),
        ]
        self.update_label_color_widgets()

        if terrain_file and os.path.exists(terrain_file):
            self.terrain_file = terrain_file

        if terrain_dir and os.path.isdir(terrain_dir):
            self.terrain_dir = terrain_dir
        self.refresh_terrain_labels()

        if noise_file and os.path.exists(noise_file):
            self.noise_file = noise_file
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_dir = noise_dir
        self.refresh_noise_label()

        if label_file and os.path.exists(label_file):
            self.label_file = label_file

        if custom_cmap_path and os.path.exists(custom_cmap_path):
            try:
                self.cmap = load_custom_cmap_from_txt(custom_cmap_path)
                self.custom_cmap_path = custom_cmap_path
                self.cmap_combo.setCurrentIndex(self.cmap_combo.findData(CUSTOM_CMAP_TOKEN))
            except Exception:
                self.custom_cmap_path = None
                self._select_builtin_cmap(cmap_name if cmap_name in BUILTIN_CMAPS else "viridis")
        else:
            self._select_builtin_cmap(cmap_name if cmap_name in BUILTIN_CMAPS else "viridis")
            self.cmap = get_builtin_cmap(self.cmap_combo.currentText())
        self.refresh_cmap_label()

        if output_xml_manual and output_xml:
            self.output_xml = output_xml
            self.output_xml_manual = True
        else:
            self.output_xml_manual = False
            self.output_xml = ""

        self.refresh_output_label()
        self.update_colorbar()
        self.refresh_label_source_ui()
        self.refresh_label_mode_ui()

    def save_settings(self):
        self.settings.setValue("batch_check", self.batch_check.isChecked())
        self.settings.setValue("labels_in_cloud", self.labels_in_cloud_check.isChecked())
        self.settings.setValue("label_col", self.label_col_box.value())
        self.settings.setValue("multiclass_labels", self.multiclass_check.isChecked())
        self.settings.setValue("ball_size", self.ball_size_box.value())
        self.settings.setValue("noise_ball_size", self.noise_ball_size_box.value())
        self.settings.setValue("floor_z", self.floor_z_box.value())
        self.settings.setValue("camera_x", self.cam_x.value())
        self.settings.setValue("camera_y", self.cam_y.value())
        self.settings.setValue("camera_z", self.cam_z.value())
        self.settings.setValue("terrain_file", self.terrain_file)
        self.settings.setValue("terrain_dir", self.terrain_dir)
        self.settings.setValue("noise_file", self.noise_file)
        self.settings.setValue("noise_dir", self.noise_dir)
        self.settings.setValue("label_file", self.label_file)
        self.settings.setValue("output_xml", self.output_xml)
        self.settings.setValue("output_xml_manual", self.output_xml_manual)
        self.settings.setValue("cmap_name", self._cmap_setting_value())
        self.settings.setValue("custom_cmap_path", self.custom_cmap_path or "")
        self.settings.setValue("label_color_0", rgb_to_hex(self.label_colors[0]))
        self.settings.setValue("label_color_1", rgb_to_hex(self.label_colors[1]))
        self.settings.setValue("ui_lang", self.lang)
        self.settings.sync()

    def _cmap_setting_value(self):
        if self.cmap_combo.currentData() == CUSTOM_CMAP_TOKEN:
            return CUSTOM_CMAP_TOKEN
        return self.cmap_combo.currentText()

    def _select_builtin_cmap(self, name):
        if name not in BUILTIN_CMAPS:
            name = "viridis"
        index = self.cmap_combo.findData(name)
        if index < 0:
            index = 0
        self.cmap_combo.setCurrentIndex(index)

    def refresh_terrain_labels(self):
        if self.terrain_file:
            self.lbl_terrain.setText(tr("file_selected", path=self.terrain_file))
        else:
            self.lbl_terrain.setText(tr("file_none"))
        if self.terrain_dir:
            self.lbl_terrain_dir.setText(tr("folder_selected", path=self.terrain_dir))
        else:
            self.lbl_terrain_dir.setText(tr("folder_none"))

    def refresh_cmap_label(self):
        if self.cmap_combo.currentData() == CUSTOM_CMAP_TOKEN:
            if self.custom_cmap_path:
                self.lbl_custom_cmap.setText(
                    tr("cmap_custom_current", name=os.path.basename(self.custom_cmap_path))
                )
            else:
                self.lbl_custom_cmap.setText(tr("cmap_custom_empty"))
        else:
            self.lbl_custom_cmap.setText(tr("cmap_builtin", name=self.cmap_combo.currentText()))

    def toggle_language(self):
        self.lang = "en" if self.lang != "en" else "zh"
        set_ui_lang(self.lang)
        self.settings.setValue("ui_lang", self.lang)
        self.apply_language()

    def apply_language(self):
        self.btn_lang.setText("English" if self.lang == "zh" else "中文")
        self.btn_lang.setToolTip(tr("lang_tip"))
        self.cloud_box.setTitle(tr("cloud_group"))
        self.btn_terrain.setText(tr("btn_file"))
        self.btn_terrain_dir.setText(tr("btn_folder"))
        self.batch_check.setText(tr("batch_check"))
        self.label_box.setTitle(tr("label_group"))
        self.labels_in_cloud_check.setText(tr("labels_in_cloud"))
        self.labels_in_cloud_check.setToolTip(tr("labels_in_cloud_tip"))
        self.lbl_col.setText(tr("col"))
        self.label_col_box.setToolTip(tr("col_tip"))
        self.multiclass_check.setText(tr("multiclass"))
        self.multiclass_check.setToolTip(tr("multiclass_tip"))
        self.btn_label.setText(tr("btn_label_file"))
        self.btn_clear_label.setText(tr("btn_clear"))
        self.noise_box.setTitle(tr("noise_group"))
        self.btn_noise.setText(tr("btn_file"))
        self.btn_noise_dir.setText(tr("btn_folder"))
        self.btn_noise_dir.setToolTip(tr("noise_dir_tip"))
        self.btn_clear_noise.setText(tr("btn_clear"))
        self.param_box.setTitle(tr("param_group"))
        self.lbl_ball.setText(tr("ball_radius"))
        self.lbl_noise_radius.setText(tr("noise_radius"))
        self.lbl_floor.setText(tr("floor_z"))
        self.lbl_camera.setText(tr("camera_origin"))
        self.color_box.setTitle(tr("color_group"))
        self.lbl_cmap_name.setText(tr("colormap"))
        self.cmap_combo.blockSignals(True)
        custom_index = self.cmap_combo.findData(CUSTOM_CMAP_TOKEN)
        if custom_index >= 0:
            self.cmap_combo.setItemText(custom_index, tr("cmap_custom"))
        self.cmap_combo.blockSignals(False)
        self.btn_load_custom_cmap.setText(tr("btn_load_cmap"))
        self.btn_label0_color.setText(tr("class0"))
        self.btn_label1_color.setText(tr("class1"))
        self.output_box.setTitle(tr("output_group"))
        self.btn_output.setText(tr("btn_xml"))
        self.btn_clear_output.setText(tr("btn_auto"))
        self.btn_start.setText(tr("btn_start"))
        self.btn_reload_preview.setText(tr("btn_reload"))
        self.preview_box.setTitle(tr("preview_group"))
        self.log_box.setTitle(tr("log_group"))
        self.refresh_terrain_labels()
        self.refresh_noise_label()
        self.refresh_output_label()
        self.refresh_cmap_label()
        self.refresh_label_source_ui()
        self.refresh_label_mode_ui()
        if self.zmin is not None and self.zmax is not None:
            self.set_z_minmax_labels(self.zmin, self.zmax)
        else:
            self.lbl_zmin.setText(tr("zmin_empty"))
            self.lbl_zmax.setText(tr("zmax_empty"))
        if self._preview_pix is None:
            self.render_preview.setText(tr("preview_placeholder"))
        self.btn_cancel_render.setText(tr("cancelling") if self._cancel_requested else tr("cancel_render"))
        if self.loading_overlay.isVisible():
            if self._cancel_requested:
                self.loading_title.setText(tr("cancelling_title"))
                self.loading_status.setText(tr("stopping"))
            elif self._loading_seconds:
                self.loading_title.setText(tr("rendering_elapsed", seconds=self._loading_seconds))
            else:
                self.loading_title.setText(tr("rendering_wait"))

    def closeEvent(self, event):
        if self._rendering:
            if not self._cancel_requested:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Question)
                box.setWindowTitle(tr("dlg_rendering"))
                box.setText(tr("close_prompt"))
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                box.setDefaultButton(QMessageBox.No)
                box.button(QMessageBox.Yes).setText(tr("dlg_yes"))
                box.button(QMessageBox.No).setText(tr("dlg_no"))
                if box.exec_() != QMessageBox.Yes:
                    event.ignore()
                    return
                self._quit_when_idle = True
                self.cancel_render()
            else:
                self._quit_when_idle = True
            event.ignore()
            return
        self.save_settings()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._rendering:
            self.cancel_render()
            return
        super().keyPressEvent(event)

    # -------------------- 色轴相关 --------------------

    def on_cmap_changed(self, _text=None):
        if self.cmap_combo.currentData() == CUSTOM_CMAP_TOKEN:
            self.refresh_cmap_label()
        else:
            self.custom_cmap_path = None
            self.cmap = get_builtin_cmap(self.cmap_combo.currentText())
            self.refresh_cmap_label()
            self.update_colorbar()

    def load_custom_cmap(self):
        file, _ = QFileDialog.getOpenFileName(self, tr("choose_cmap"), "", tr("filter_txt"))
        if not file:
            return
        try:
            self.cmap = load_custom_cmap_from_txt(file)
            self.custom_cmap_path = file
            self.cmap_combo.setCurrentIndex(self.cmap_combo.findData(CUSTOM_CMAP_TOKEN))
            self.refresh_cmap_label()
            self.update_colorbar()
            self.log.append(tr("cmap_loaded", path=file))
        except Exception as e:
            QMessageBox.critical(self, tr("cmap_fail"), str(e))

    def update_colorbar(self):
        if self.multiclass_check.isChecked() and self._label_class_count > 0:
            _, palette = class_color_palette(
                np.arange(self._label_class_count),
                self.cmap,
                binary_colors=self.label_colors,
                multiclass=True,
            )
            pix = make_class_legend_pixmap(palette, width=360, height=24)
        else:
            pix = make_colorbar_pixmap(self.cmap, width=360, height=24)
        self.colorbar_label.setPixmap(pix)

    def update_label_color_widgets(self):
        for idx, (button, label) in enumerate(((self.btn_label0_color, self.label0_preview), (self.btn_label1_color, self.label1_preview))):
            color_hex = rgb_to_hex(self.label_colors[idx])
            button.setStyleSheet(f"background-color: {color_hex};")
            label.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #999;")

    def choose_label_color(self, index):
        initial = QColor(rgb_to_hex(self.label_colors[index]))
        color = QColorDialog.getColor(initial, self, tr("choose_class_color", index=index))
        if not color.isValid():
            return
        self.label_colors[index] = qcolor_to_rgb_tuple(color)
        self.update_label_color_widgets()
        self.save_settings()
        self.log.append(tr("color_updated", index=index, hex=rgb_to_hex(self.label_colors[index])))

    # -------------------- 公用更新 --------------------

    def set_z_minmax_labels(self, zmin, zmax):
        self.lbl_zmin.setText(tr("zmin", value=f"{zmin:.4f}"))
        self.lbl_zmax.setText(tr("zmax", value=f"{zmax:.4f}"))

    def show_image_on_label(self, rgb, label: QLabel):
        pix = numpy_rgb_to_qpixmap(rgb)
        if label is self.render_preview:
            self._preview_pix = pix
            self._update_preview_pixmap()
            return
        w = max(1, label.width())
        h = max(1, label.height())
        label.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_preview_pixmap(self):
        pix = self._preview_pix
        if pix is None or pix.isNull():
            return
        w = max(1, self.render_preview.width())
        h = max(1, self.render_preview.height())
        self.render_preview.setPixmap(pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()
        self._sync_loading_overlay()

    def get_default_output_xml(self, file_path):
        return os.path.splitext(file_path)[0] + ".xml"

    def get_output_xml_path(self, file_path):
        if self.output_xml_manual and self.output_xml:
            return self.output_xml
        return self.get_default_output_xml(file_path)

    def refresh_output_label(self):
        if self.output_xml_manual and self.output_xml:
            self.lbl_output.setText(tr("output_manual", path=self.output_xml))
        elif self.terrain_file:
            self.lbl_output.setText(tr("output_auto", path=self.get_default_output_xml(self.terrain_file)))
        else:
            self.lbl_output.setText(tr("output_default"))

    # -------------------- 选择文件/文件夹 --------------------

    def choose_terrain(self):
        file, _ = QFileDialog.getOpenFileName(self, tr("choose_cloud"), "", tr("filter_txt"))
        if file:
            self.terrain_file = file
            self.terrain_dir = ""
            self.batch_check.setChecked(False)
            self.refresh_terrain_labels()

            if not self.output_xml_manual:
                self.output_xml = ""
            self.refresh_output_label()

            # 读取并更新 z 范围
            try:
                terrain, embedded_labels = load_pointcloud(
                    self.terrain_file,
                    with_labels=self.labels_in_cloud_check.isChecked(),
                    multiclass=self.multiclass_check.isChecked(),
                    label_col=self.label_col_box.value(),
                )
                terrain = normalize_pointcloud(terrain)[0]
                self.zmin = float(terrain[:, 2].min())
                self.zmax = float(terrain[:, 2].max())
                self.set_z_minmax_labels(self.zmin, self.zmax)
                if embedded_labels is not None:
                    self.remember_label_classes(embedded_labels)
            except Exception as e:
                self.log.append(tr("read_cloud_fail", error=e))
            self.save_settings()

    def choose_terrain_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("choose_cloud_dir"), "")
        if d:
            self.terrain_dir = d
            self.terrain_file = ""
            self.refresh_terrain_labels()
            self.refresh_output_label()
            self.batch_check.setChecked(True)
            self.save_settings()

    def choose_noise(self):
        file, _ = QFileDialog.getOpenFileName(self, tr("choose_noise"), "", tr("filter_txt"))
        if file:
            self.noise_file = file
            self.refresh_noise_label()
            self.save_settings()

    def choose_noise_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("choose_noise_dir"), "")
        if d:
            self.noise_dir = d
            self.refresh_noise_label()
            self.save_settings()
            self.log.append(tr("batch_noise_hint"))

    def clear_noise(self):
        self.noise_file = ""
        self.noise_dir = ""
        self.refresh_noise_label()
        self.save_settings()
        self.log.append(tr("noise_cleared"))

    def refresh_noise_label(self):
        parts = []
        if self.noise_file:
            parts.append(tr("noise_file", path=self.noise_file))
        if self.noise_dir:
            parts.append(tr("noise_batch", path=self.noise_dir))
        self.lbl_noise.setText(tr("list_sep").join(parts) if parts else tr("label_none"))

    def find_matching_noise_file(self, clean_path):
        if not self.noise_dir or not os.path.isdir(self.noise_dir):
            return None
        clean_name = os.path.basename(clean_path)
        wanted = clean_name.lower()
        for name in os.listdir(self.noise_dir):
            if name.lower() == wanted:
                return os.path.join(self.noise_dir, name)
        return None

    def choose_label(self):
        file, _ = QFileDialog.getOpenFileName(self, tr("choose_label"), "", tr("filter_label"))
        if file:
            self.label_file = file
            self.lbl_label.setText(file)
            try:
                labels = load_labels_from_txt(file, multiclass=self.multiclass_check.isChecked())
                self.remember_label_classes(labels)
            except Exception as e:
                self.log.append(tr("label_read_fail", error=e))
            self.save_settings()

    def clear_label(self):
        self.label_file = ""
        self.lbl_label.setText(tr("label_none"))
        self.save_settings()
        self.log.append(tr("label_cleared"))

    def on_labels_in_cloud_changed(self):
        self.refresh_label_source_ui()
        self.save_settings()
        self.reload_embedded_labels_from_current_file()

    def on_label_col_changed(self):
        self.refresh_label_source_ui()
        self.save_settings()
        self.reload_embedded_labels_from_current_file()

    def on_multiclass_changed(self):
        self.refresh_label_mode_ui()
        self.save_settings()
        self.reload_embedded_labels_from_current_file()

    def reload_embedded_labels_from_current_file(self):
        if not (self.labels_in_cloud_check.isChecked() and self.terrain_file and os.path.exists(self.terrain_file)):
            return
        try:
            _, labels = load_pointcloud(
                self.terrain_file,
                with_labels=True,
                multiclass=self.multiclass_check.isChecked(),
                label_col=self.label_col_box.value(),
            )
            self.remember_label_classes(labels)
        except Exception as e:
            self.log.append(tr("label_reload_fail", error=e))

    def remember_label_classes(self, labels):
        if labels is None:
            self._label_class_count = 0
        else:
            self._label_class_count = int(np.unique(np.asarray(labels).reshape(-1)).size)
        self.refresh_label_mode_ui()

    def refresh_label_mode_ui(self):
        multi = self.multiclass_check.isChecked()
        self.btn_label0_color.setEnabled(not multi)
        self.btn_label1_color.setEnabled(not multi)
        self.label0_preview.setVisible(not multi)
        self.label1_preview.setVisible(not multi)
        if multi:
            if self._label_class_count > 0:
                self.lbl_colorbar_caption.setText(tr("caption_multi_n", n=self._label_class_count))
            else:
                self.lbl_colorbar_caption.setText(tr("caption_multi_empty"))
        else:
            self.lbl_colorbar_caption.setText(tr("caption_gradient"))
        self.update_colorbar()

    def refresh_label_source_ui(self):
        embedded = self.labels_in_cloud_check.isChecked()
        self.btn_label.setEnabled(not embedded)
        self.btn_clear_label.setEnabled(not embedded)
        self.label_col_box.setEnabled(embedded)
        if embedded:
            col = self.label_col_box.value()
            self.lbl_label.setText(tr("label_embedded", col=col))
        elif self.label_file:
            self.lbl_label.setText(self.label_file)
        else:
            self.lbl_label.setText(tr("label_none"))

    def choose_output_xml(self):
        default_path = self.output_xml if self.output_xml_manual and self.output_xml else self.get_default_output_xml(self.terrain_file) if self.terrain_file else "scene.xml"
        file, _ = QFileDialog.getSaveFileName(self, tr("choose_xml"), default_path, tr("filter_xml"))
        if file:
            if not file.lower().endswith(".xml"):
                file += ".xml"
            self.output_xml = file
            self.output_xml_manual = True
            self.refresh_output_label()
            self.save_settings()

    def clear_output_xml(self):
        self.output_xml = ""
        self.output_xml_manual = False
        self.refresh_output_label()
        self.save_settings()
        self.log.append(tr("output_reset"))

    # -------------------- 渲染 / Loading --------------------

    def _build_loading_overlay(self):
        self.loading_overlay = QWidget(self)
        self.loading_overlay.setObjectName("loadingOverlay")
        self.loading_overlay.setAttribute(Qt.WA_StyledBackground, True)
        self.loading_overlay.setStyleSheet("#loadingOverlay { background: rgba(20, 20, 20, 160); }")
        box = QVBoxLayout(self.loading_overlay)
        box.setAlignment(Qt.AlignCenter)
        box.setSpacing(12)
        self.loading_title = QLabel(tr("rendering_wait"))
        self.loading_title.setAlignment(Qt.AlignCenter)
        self.loading_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        self.loading_status = QLabel(tr("preparing"))
        self.loading_status.setAlignment(Qt.AlignCenter)
        self.loading_status.setStyleSheet("color: #eee; font-size: 13px; background: transparent;")
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setFixedWidth(280)
        self.loading_bar.setTextVisible(False)
        self.btn_cancel_render = QPushButton(tr("cancel_render"))
        self.btn_cancel_render.setFixedSize(160, 36)
        self.btn_cancel_render.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_render.setStyleSheet(
            "QPushButton { background: #d9534f; color: white; font-weight: bold; border: none; border-radius: 4px; }"
            "QPushButton:hover { background: #c9302c; }"
            "QPushButton:disabled { background: #777777; color: #eeeeee; }"
        )
        self.btn_cancel_render.clicked.connect(self.cancel_render)
        self._cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        self._cancel_shortcut.setContext(Qt.WindowShortcut)
        self._cancel_shortcut.activated.connect(self.cancel_render)
        box.addWidget(self.loading_title)
        box.addWidget(self.loading_status)
        box.addWidget(self.loading_bar, 0, Qt.AlignCenter)
        box.addWidget(self.btn_cancel_render, 0, Qt.AlignCenter)
        self.loading_overlay.hide()
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(1000)
        self._loading_timer.timeout.connect(self._tick_loading)

    def _sync_loading_overlay(self):
        overlay = getattr(self, "loading_overlay", None)
        if overlay is not None:
            overlay.setGeometry(self.rect())
            overlay.raise_()

    def _tick_loading(self):
        if self._cancel_requested:
            return
        self._loading_seconds += 1
        self.loading_title.setText(tr("rendering_elapsed", seconds=self._loading_seconds))

    def _set_loading(self, on, status=None):
        self._sync_loading_overlay()
        if on:
            self._loading_seconds = 0
            self._cancel_requested = False
            self.loading_title.setText(tr("rendering_wait"))
            self.loading_status.setText(status if status is not None else tr("preparing"))
            self.btn_cancel_render.setEnabled(True)
            self.btn_cancel_render.setText(tr("cancel_render"))
            self.loading_overlay.show()
            self.loading_overlay.raise_()
            self._loading_timer.start()
            self.btn_start.setEnabled(False)
            self.btn_reload_preview.setEnabled(False)
        else:
            self._loading_timer.stop()
            self.loading_overlay.hide()
            self.btn_start.setEnabled(True)
            self.btn_reload_preview.setEnabled(True)

    def cancel_render(self):
        if not self._rendering or self._cancel_requested:
            return
        worker = self._render_worker
        if worker is None:
            return
        self._cancel_requested = True
        self.btn_cancel_render.setEnabled(False)
        self.btn_cancel_render.setText(tr("cancelling"))
        self.loading_title.setText(tr("cancelling_title"))
        self.loading_status.setText(tr("stopping"))
        self.log.append(tr("cancel_requested"))
        worker.request_cancel()

    def _snapshot_render_job(self, file_path, preview=False, noise_path=None):
        return {
            "file_path": file_path,
            "preview": preview,
            "noise_path": noise_path,
            "use_embedded_labels": self.labels_in_cloud_check.isChecked(),
            "multiclass": self.multiclass_check.isChecked(),
            "label_col": self.label_col_box.value(),
            "label_file": "" if self.labels_in_cloud_check.isChecked() else self.label_file,
            "ball_size": self.ball_size_box.value(),
            "noise_ball_size": self.noise_ball_size_box.value(),
            "camera_origin": (self.cam_x.value(), self.cam_y.value(), self.cam_z.value()),
            "floor_z": self.floor_z_box.value(),
            "cmap": pack_cmap(self.cmap),
            "label_colors": list(self.label_colors),
            "xml_path": self.get_output_xml_path(file_path),
            "lang": self.lang,
        }

    def _start_render_jobs(self, jobs, done_message=""):
        if self._rendering:
            return
        self._rendering = True
        self._render_done_message = done_message
        self._set_loading(True, tr("queue_status", n=len(jobs)))

        thread = QThread(self)
        worker = RenderWorker(
            jobs,
            getattr(self, "_mi_variant", "scalar_rgb"),
            getattr(self, "_mi_search_paths", None),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_msg.connect(self.log.append)
        worker.status.connect(self.loading_status.setText)
        worker.z_range.connect(self.set_z_minmax_labels)
        worker.labels_ready.connect(self.remember_label_classes)
        worker.preview_ready.connect(self._on_preview_ready)
        worker.finished.connect(self._on_render_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._render_thread = thread
        self._render_worker = worker
        thread.start()

    def _on_preview_ready(self, img):
        self.show_image_on_label(img, self.render_preview)

    def _on_render_finished(self, cancelled=False):
        done_message = self._render_done_message
        self._rendering = False
        self._cancel_requested = False
        self._set_loading(False)
        if cancelled:
            self.log.append(tr("render_cancelled"))
        elif done_message:
            self.log.append(done_message)
        self._render_thread = None
        self._render_worker = None
        if self._quit_when_idle:
            self._quit_when_idle = False
            QTimer.singleShot(0, self.close)

    def render_scene(self):
        if self._rendering:
            QMessageBox.information(self, tr("dlg_wait"), tr("busy_stop"))
            return

        jobs = []
        done_message = ""
        if self.batch_check.isChecked():
            if not self.terrain_dir:
                QMessageBox.warning(self, tr("dlg_error"), tr("need_folder"))
                return
            if self.labels_in_cloud_check.isChecked():
                mode = tr("mode_multi") if self.multiclass_check.isChecked() else tr("mode_binary")
                col = self.label_col_box.value()
                self.log.append(tr("batch_labels", col=col, mode=mode))
            elif self.label_file:
                self.log.append(tr("batch_ignore_label"))

            if self.noise_dir:
                self.log.append(tr("batch_noise_dir", path=self.noise_dir))
            elif self.noise_file:
                self.log.append(tr("batch_ignore_noise"))

            txts = sorted([
                os.path.join(self.terrain_dir, f)
                for f in os.listdir(self.terrain_dir)
                if f.lower().endswith(".txt")
            ])
            if not txts:
                QMessageBox.information(self, tr("dlg_info"), tr("no_txt"))
                return

            self.log.append(tr("batch_list", n=len(txts)))
            for p in txts:
                self.log.append(f"- {os.path.basename(p)}")
                matched_noise = self.find_matching_noise_file(p) if self.noise_dir else None
                if self.noise_dir and matched_noise is None:
                    self.log.append(tr("noise_missing", name=os.path.basename(p)))
                jobs.append(self._snapshot_render_job(p, preview=False, noise_path=matched_noise))
            self.log.append(tr("batch_start"))
            done_message = tr("batch_done")
        else:
            if not self.terrain_file:
                QMessageBox.warning(self, tr("dlg_error"), tr("need_file"))
                return
            self.refresh_output_label()
            self.log.append(tr("single_start"))
            jobs.append(self._snapshot_render_job(
                self.terrain_file,
                preview=True,
                noise_path=self.noise_file or None,
            ))
            done_message = tr("single_done")

        self._start_render_jobs(jobs, done_message)

    def reload_tiff_preview(self):
        if self._rendering:
            QMessageBox.information(self, tr("dlg_wait"), tr("reload_busy"))
            return
        if not self.terrain_file:
            QMessageBox.information(self, tr("dlg_info"), tr("reload_need_file"))
            return
        tiff_path = self.get_output_xml_path(self.terrain_file).replace(".xml", ".tiff")
        if not os.path.exists(tiff_path):
            QMessageBox.information(self, tr("dlg_info"), tr("reload_missing"))
            return
        try:
            self.show_image_on_label(read_tiff(tiff_path), self.render_preview)
            self.log.append(tr("reload_ok"))
        except Exception as e:
            self.log.append(tr("reload_fail", error=e))


# =======================================
# 启动程序
# =======================================

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--render-job":
        run_render_job_cli(sys.argv[2], sys.argv[3])
        raise SystemExit(0)
    app = QApplication(sys.argv)
    ui = RenderUI()
    ui.show()
    sys.exit(app.exec_())
