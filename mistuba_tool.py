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
        raise ValueError(f"无法识别的图像形状：{array.shape}")
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
        raise ValueError("TIFF 需要 H×W×3 图像")
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
    raise ValueError(f"不支持的 TIFF 字段类型：{typ}")


def read_tiff(path):
    """读取本程序写出的未压缩 RGB TIFF（float32 或 uint8）。"""
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 8 or blob[:2] != b"II" or int.from_bytes(blob[2:4], "little") != 42:
        raise ValueError("只支持小端未压缩 TIFF")
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
                raise ValueError(f"TIFF 缺少字段 {tag}")
            return default
        typ, number, raw = tags[tag]
        return _tiff_values(blob, typ, number, raw)

    width = values(256)[0]
    height = values(257)[0]
    bits = values(258)
    if values(259)[0] != 1 or values(284, [1])[0] != 1:
        raise ValueError("只支持未压缩、逐像素排列的 TIFF")
    samples = values(277, [1])[0]
    sample_format = values(339, [1] * samples)
    strips = values(273)
    counts = values(279)
    if len(strips) != 1 or len(counts) != 1:
        raise ValueError("只支持单条带 TIFF")
    start = strips[0]
    raw = blob[start:start + counts[0]]
    if bits == [32, 32, 32] and sample_format[:3] == [3, 3, 3]:
        array = np.frombuffer(raw, dtype="<f4").reshape(height, width, 3)
    elif bits == [8, 8, 8]:
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    else:
        raise ValueError(f"不支持的 TIFF 像素格式：bits={bits} format={sample_format}")
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
        raise ValueError("标签为空。")
    if not multiclass and n > 2:
        raise ValueError(f"当前为二分类模式，检测到 {n} 个类别。请勾选“多分类标签”。")

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
            raise ValueError("标签数量与点云点数不一致。")
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
        raise ValueError("自定义色轴文件至少需要两行 RGB。")

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
        raise ValueError("标签为空。")
    unique_values = np.unique(labels)
    if not multiclass and unique_values.size > 2:
        raise ValueError(f"当前为二分类模式，检测到 {unique_values.size} 个类别。请勾选“多分类标签”。")
    return labels


def load_labels_from_txt(label_path, multiclass=False):
    return validate_labels(np.loadtxt(label_path), multiclass=multiclass)


def load_pointcloud(path, with_labels=False, multiclass=False, label_col=4):
    data = np.asarray(np.loadtxt(path))
    if data.size == 0:
        raise ValueError("点云文件为空。")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("点云至少需要 x y z 三列。")

    points = data[:, :3]
    labels = None
    if with_labels:
        col = int(label_col)
        if col < 1:
            raise ValueError("标签列必须从 1 开始计数。")
        idx = col - 1
        if data.shape[1] <= idx:
            raise ValueError(
                f"已勾选“点云含标签”，但文件只有 {data.shape[1]} 列，无法读取第 {col} 列。"
            )
        labels = validate_labels(data[:, idx], multiclass=multiclass)
        if labels.shape[0] != points.shape[0]:
            raise ValueError(f"标签数 {labels.shape[0]} 与点数 {points.shape[0]} 不一致。")
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
    log(f"📄 渲染文件：{file_path}")
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
            log(f"🌫️ 已叠加噪声：{noise_path}")
        except Exception as e:
            log(f"⚠️ 噪声点云读取失败：{e}")

    labels = None
    if job["use_embedded_labels"]:
        labels = embedded_labels
        n_cls = int(np.unique(labels).size) if labels is not None else 0
        log(f"🏷️ 已从点云第 {job['label_col']} 列读取标签（{n_cls} 类）")
    elif job.get("label_file"):
        try:
            labels = load_labels_from_txt(job["label_file"], multiclass=job["multiclass"])
            if labels.shape[0] != terrain.shape[0]:
                raise ValueError(f"标签数 {labels.shape[0]} 与点数 {terrain.shape[0]} 不一致。")
            n_cls = int(np.unique(labels).size)
            log(f"🏷️ 已载入标签文件：{job['label_file']}（{n_cls} 类）")
        except Exception as e:
            log(f"⚠️ 标签文件读取失败，将回退到渐变色轴：{e}")
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
    log("正在写入 TIFF...")
    write_float_tiff(tiff_path, rgb)
    log(f"✅ 渲染完成：\n{xml_path}\n{exr_path}\n{tiff_path}")

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
        apply_mitsuba_search_paths(collect_mitsuba_search_paths())
        total = len(jobs)
        for i, job in enumerate(jobs, 1):
            name = os.path.basename(job["file_path"])
            emit("STATUS", f"正在渲染 {i}/{total}：{name}")
            emit("LOG", f"—— [{i}/{total}] ——")
            try:
                result = execute_render_job(job, lambda message: emit("LOG", message))
            except Exception as e:
                emit("LOG", f"❌ 渲染失败：{job['file_path']} - {e}")
                continue
            emit("Z", f"{float(result['zmin'])}\t{float(result['zmax'])}")
            if result["labels"] is not None:
                label_path = f"{log_path}.labels{i}.npy"
                np.save(label_path, result["labels"])
                emit("LABELS", label_path)
            if result["preview_img"] is not None:
                emit("PREVIEW", job["xml_path"].replace(".xml", ".tiff"))
    except Exception as e:
        emit("LOG", f"❌ 渲染任务异常中止：{e}")
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
                    self.log_msg.emit("⏹️ 已取消，正在停止渲染进程。")
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
                        self.log_msg.emit(f"❌ 渲染进程异常退出（代码 {proc.returncode}）。{detail}")
                    break
                time.sleep(0.1)
        except Exception as e:
            if self._cancel_event.is_set():
                cancelled = True
                self.log_msg.emit("⏹️ 渲染已取消。")
            else:
                self.log_msg.emit(f"❌ 渲染任务异常中止：{e}")
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

        cloud_box = QGroupBox("1. 点云")
        cloud_layout = QVBoxLayout()
        cloud_btns = QHBoxLayout()
        btn_terrain = QPushButton("选择文件")
        btn_terrain.clicked.connect(self.choose_terrain)
        cloud_btns.addWidget(btn_terrain)
        btn_terrain_dir = QPushButton("选择文件夹")
        btn_terrain_dir.clicked.connect(self.choose_terrain_dir)
        cloud_btns.addWidget(btn_terrain_dir)
        self.batch_check = QCheckBox("批量渲染文件夹内全部 .txt")
        self.batch_check.setChecked(False)
        cloud_btns.addWidget(self.batch_check)
        cloud_btns.addStretch(1)
        cloud_layout.addLayout(cloud_btns)
        self.lbl_terrain = QLabel("单文件：未选择")
        self.lbl_terrain_dir = QLabel("文件夹：未选择")
        _style_path_label(self.lbl_terrain)
        _style_path_label(self.lbl_terrain_dir)
        cloud_layout.addWidget(self.lbl_terrain)
        cloud_layout.addWidget(self.lbl_terrain_dir)
        cloud_box.setLayout(cloud_layout)
        left.addWidget(cloud_box)

        label_box = QGroupBox("2. 标签")
        label_layout = QVBoxLayout()
        embed_row = QHBoxLayout()
        self.labels_in_cloud_check = QCheckBox("点云含标签")
        self.labels_in_cloud_check.setChecked(False)
        self.labels_in_cloud_check.setToolTip("打开后从点云文件指定列读取标签（单文件和批量都生效）。XYZ 仍使用前三列。")
        embed_row.addWidget(self.labels_in_cloud_check)
        embed_row.addWidget(QLabel("列"))
        self.label_col_box = QSpinBox()
        self.label_col_box.setRange(1, 64)
        self.label_col_box.setValue(4)
        self.label_col_box.setToolTip("点云文件中标签所在列，从 1 开始计数，默认第 4 列。批量时文件夹内所有文件使用同一列。")
        embed_row.addWidget(self.label_col_box)
        self.multiclass_check = QCheckBox("多分类")
        self.multiclass_check.setChecked(False)
        self.multiclass_check.setToolTip("打开后允许 2 个以上类别，颜色按当前色轴自动分配；关闭则仅支持二分类，使用手动配色。")
        embed_row.addWidget(self.multiclass_check)
        embed_row.addStretch(1)
        label_layout.addLayout(embed_row)

        label_file_row = QHBoxLayout()
        self.btn_label = QPushButton("选择单独标签文件")
        self.btn_label.clicked.connect(self.choose_label)
        label_file_row.addWidget(self.btn_label)
        self.btn_clear_label = QPushButton("清除")
        self.btn_clear_label.clicked.connect(self.clear_label)
        label_file_row.addWidget(self.btn_clear_label)
        label_file_row.addStretch(1)
        label_layout.addLayout(label_file_row)
        self.lbl_label = QLabel("未选择")
        _style_path_label(self.lbl_label)
        label_layout.addWidget(self.lbl_label)
        label_box.setLayout(label_layout)
        left.addWidget(label_box)

        noise_box = QGroupBox("3. 噪声（可选）")
        noise_layout = QVBoxLayout()
        noise_row = QHBoxLayout()
        btn_noise = QPushButton("选择文件")
        btn_noise.clicked.connect(self.choose_noise)
        noise_row.addWidget(btn_noise)
        btn_noise_dir = QPushButton("选择文件夹")
        btn_noise_dir.setToolTip("批量时噪声文件名必须与干净点云相同，例如 clean/a.txt 对应 noise/a.txt")
        btn_noise_dir.clicked.connect(self.choose_noise_dir)
        noise_row.addWidget(btn_noise_dir)
        btn_clear_noise = QPushButton("清除")
        btn_clear_noise.clicked.connect(self.clear_noise)
        noise_row.addWidget(btn_clear_noise)
        noise_row.addStretch(1)
        noise_layout.addLayout(noise_row)
        self.lbl_noise = QLabel("未选择")
        _style_path_label(self.lbl_noise)
        noise_layout.addWidget(self.lbl_noise)
        noise_box.setLayout(noise_layout)
        left.addWidget(noise_box)

        param_box = QGroupBox("4. 渲染参数")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ball_size_box = QDoubleSpinBox()
        self.ball_size_box.setDecimals(6)
        self.ball_size_box.setRange(0.0001, 1.0)
        self.ball_size_box.setSingleStep(0.001)
        self.ball_size_box.setValue(0.011)
        form.addRow("主点球半径", self.ball_size_box)

        self.noise_ball_size_box = QDoubleSpinBox()
        self.noise_ball_size_box.setDecimals(6)
        self.noise_ball_size_box.setRange(0.0001, 1.0)
        self.noise_ball_size_box.setSingleStep(0.001)
        self.noise_ball_size_box.setValue(0.004)
        form.addRow("噪声球半径", self.noise_ball_size_box)

        self.floor_z_box = QDoubleSpinBox()
        self.floor_z_box.setDecimals(4)
        self.floor_z_box.setRange(-1000.0, 1000.0)
        self.floor_z_box.setSingleStep(0.01)
        self.floor_z_box.setValue(-0.15)
        form.addRow("底板高度 z", self.floor_z_box)

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
        form.addRow("相机 origin", cam_wrap)
        param_box.setLayout(form)
        left.addWidget(param_box)

        color_box = QGroupBox("5. 着色")
        color_layout = QVBoxLayout()
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("色轴"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(BUILTIN_CMAPS + ["(自定义txt)"])
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        cmap_row.addWidget(self.cmap_combo, 1)
        self.btn_load_custom_cmap = QPushButton("载入 txt")
        self.btn_load_custom_cmap.clicked.connect(self.load_custom_cmap)
        cmap_row.addWidget(self.btn_load_custom_cmap)
        color_layout.addLayout(cmap_row)
        self.lbl_custom_cmap = QLabel("当前：内置 viridis")
        _style_path_label(self.lbl_custom_cmap)
        color_layout.addWidget(self.lbl_custom_cmap)

        self.lbl_colorbar_caption = QLabel("渐变色轴预览（无标签时使用）")
        color_layout.addWidget(self.lbl_colorbar_caption)
        self.colorbar_label = QLabel()
        self.colorbar_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.colorbar_label.setFixedHeight(28)
        color_layout.addWidget(self.colorbar_label)

        binary_row = QHBoxLayout()
        self.btn_label0_color = QPushButton("类别 0")
        self.btn_label0_color.clicked.connect(lambda: self.choose_label_color(0))
        binary_row.addWidget(self.btn_label0_color)
        self.label0_preview = QLabel("0")
        self.label0_preview.setAlignment(Qt.AlignCenter)
        self.label0_preview.setFixedSize(36, 24)
        binary_row.addWidget(self.label0_preview)
        self.btn_label1_color = QPushButton("类别 1")
        self.btn_label1_color.clicked.connect(lambda: self.choose_label_color(1))
        binary_row.addWidget(self.btn_label1_color)
        self.label1_preview = QLabel("1")
        self.label1_preview.setAlignment(Qt.AlignCenter)
        self.label1_preview.setFixedSize(36, 24)
        binary_row.addWidget(self.label1_preview)
        binary_row.addStretch(1)
        color_layout.addLayout(binary_row)
        color_box.setLayout(color_layout)
        left.addWidget(color_box)

        output_box = QGroupBox("6. 输出")
        output_layout = QVBoxLayout()
        output_row = QHBoxLayout()
        btn_output = QPushButton("指定 XML 路径")
        btn_output.clicked.connect(self.choose_output_xml)
        output_row.addWidget(btn_output)
        btn_clear_output = QPushButton("恢复自动")
        btn_clear_output.clicked.connect(self.clear_output_xml)
        output_row.addWidget(btn_clear_output)
        output_row.addStretch(1)
        output_layout.addLayout(output_row)
        self.lbl_output = QLabel("输出 XML 将自动生成（单文件模式）")
        _style_path_label(self.lbl_output)
        output_layout.addWidget(self.lbl_output)
        output_box.setLayout(output_layout)
        left.addWidget(output_box)

        left.addStretch(1)

        btn_start = QPushButton("开始渲染")
        btn_start.setObjectName("btnStart")
        btn_start.clicked.connect(self.render_scene)
        self.btn_start = btn_start
        left.addWidget(btn_start)
        btn_reload_preview = QPushButton("从 TIFF 载入预览")
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

        preview_box = QGroupBox("预览")
        pv = QVBoxLayout()
        z_range_layout = QHBoxLayout()
        self.lbl_zmin = QLabel("zmin: -")
        self.lbl_zmax = QLabel("zmax: -")
        z_range_layout.addWidget(self.lbl_zmin)
        z_range_layout.addStretch(1)
        z_range_layout.addWidget(self.lbl_zmax)
        pv.addLayout(z_range_layout)

        self.render_preview = QLabel("渲染结果将显示在这里")
        self.render_preview.setAlignment(Qt.AlignCenter)
        self.render_preview.setMinimumHeight(240)
        self.render_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.render_preview.setStyleSheet("border: 1px solid #ccc; background: #f7f7f7;")
        pv.addWidget(self.render_preview, 1)
        preview_box.setLayout(pv)
        preview_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        log_box = QGroupBox("日志")
        log_layout = QVBoxLayout()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(90)
        log_layout.addWidget(self.log)
        log_box.setLayout(log_layout)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(preview_box)
        right_split.addWidget(log_box)
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
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
            self.log.append(f"⚠️ 设置 Mitsuba 变体失败：{e}\n将尝试使用 'scalar_rgb'")
            try:
                mi.set_variant('scalar_rgb')
                self._mi_variant = "scalar_rgb"
            except Exception as e2:
                self.log.append(f"⚠️ 备用变体设置仍失败：{e2}")
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
            self.lbl_terrain.setText(f"单文件：{terrain_file}")

        if terrain_dir and os.path.isdir(terrain_dir):
            self.terrain_dir = terrain_dir
            self.lbl_terrain_dir.setText(f"文件夹：{terrain_dir}")

        if noise_file and os.path.exists(noise_file):
            self.noise_file = noise_file
        if noise_dir and os.path.isdir(noise_dir):
            self.noise_dir = noise_dir
        self.refresh_noise_label()

        if label_file and os.path.exists(label_file):
            self.label_file = label_file
            self.lbl_label.setText(label_file)

        if custom_cmap_path and os.path.exists(custom_cmap_path):
            try:
                self.cmap = load_custom_cmap_from_txt(custom_cmap_path)
                self.custom_cmap_path = custom_cmap_path
                self.cmap_combo.setCurrentText("(自定义txt)")
                self.lbl_custom_cmap.setText(f"当前：自定义 {os.path.basename(custom_cmap_path)}")
            except Exception:
                self.custom_cmap_path = None
                self.cmap_combo.setCurrentText(cmap_name if cmap_name in BUILTIN_CMAPS else "viridis")
        else:
            self.cmap_combo.setCurrentText(cmap_name if cmap_name in BUILTIN_CMAPS else "viridis")
            self.cmap = get_builtin_cmap(self.cmap_combo.currentText())
            self.lbl_custom_cmap.setText(f"当前：内置 {self.cmap_combo.currentText()}")

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
        self.settings.setValue("cmap_name", self.cmap_combo.currentText())
        self.settings.setValue("custom_cmap_path", self.custom_cmap_path or "")
        self.settings.setValue("label_color_0", rgb_to_hex(self.label_colors[0]))
        self.settings.setValue("label_color_1", rgb_to_hex(self.label_colors[1]))
        self.settings.sync()

    def closeEvent(self, event):
        if self._rendering:
            if not self._cancel_requested:
                reply = QMessageBox.question(
                    self,
                    "正在渲染",
                    "渲染尚未完成。要取消渲染并关闭窗口吗？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
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

    def on_cmap_changed(self, text):
        if text == "(自定义txt)":
            if self.custom_cmap_path:
                # 已有自定义
                self.lbl_custom_cmap.setText(f"当前：自定义 {os.path.basename(self.custom_cmap_path)}")
            else:
                self.lbl_custom_cmap.setText("当前：自定义（尚未载入）")
        else:
            self.custom_cmap_path = None
            self.cmap = get_builtin_cmap(text)
            self.lbl_custom_cmap.setText(f"当前：内置 {text}")
            self.update_colorbar()

    def load_custom_cmap(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择自定义色轴 txt", "", "TXT Files (*.txt)")
        if not file:
            return
        try:
            self.cmap = load_custom_cmap_from_txt(file)
            self.custom_cmap_path = file
            self.cmap_combo.setCurrentText("(自定义txt)")
            self.lbl_custom_cmap.setText(f"当前：自定义 {os.path.basename(file)}")
            self.update_colorbar()
            self.log.append(f"🎨 已载入自定义色轴：{file}")
        except Exception as e:
            QMessageBox.critical(self, "色轴载入失败", str(e))

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
        color = QColorDialog.getColor(initial, self, f"选择类别 {index} 颜色")
        if not color.isValid():
            return
        self.label_colors[index] = qcolor_to_rgb_tuple(color)
        self.update_label_color_widgets()
        self.save_settings()
        self.log.append(f"🎨 已更新类别 {index} 配色：{rgb_to_hex(self.label_colors[index])}")

    # -------------------- 公用更新 --------------------

    def set_z_minmax_labels(self, zmin, zmax):
        self.lbl_zmin.setText(f"zmin: {zmin:.4f}")
        self.lbl_zmax.setText(f"zmax: {zmax:.4f}")

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
            self.lbl_output.setText(f"手动：{self.output_xml}")
        elif self.terrain_file:
            self.lbl_output.setText(f"自动：{self.get_default_output_xml(self.terrain_file)}")
        else:
            self.lbl_output.setText("单文件模式将自动在点云同目录生成 XML")

    # -------------------- 选择文件/文件夹 --------------------

    def choose_terrain(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择点云文件", "", "TXT Files (*.txt)")
        if file:
            self.terrain_file = file
            self.lbl_terrain.setText(f"单文件：{file}")
            self.terrain_dir = ""  # 若选择了单文件，清空文件夹
            self.lbl_terrain_dir.setText("文件夹：未选择")
            self.batch_check.setChecked(False)

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
                self.log.append(f"⚠️ 读取点云失败，无法更新 z 范围：{e}")
            self.save_settings()

    def choose_terrain_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择点云文件夹", "")
        if d:
            self.terrain_dir = d
            self.lbl_terrain_dir.setText(f"文件夹：{d}")
            self.terrain_file = ""   # 若选择文件夹，清空单文件
            self.lbl_terrain.setText("单文件：未选择")
            self.refresh_output_label()
            self.batch_check.setChecked(True)
            self.save_settings()

    def choose_noise(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择噪声点云文件", "", "TXT Files (*.txt)")
        if file:
            self.noise_file = file
            self.refresh_noise_label()
            self.save_settings()

    def choose_noise_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择噪声点云文件夹", "")
        if d:
            self.noise_dir = d
            self.refresh_noise_label()
            self.save_settings()
            self.log.append("批量噪声将按文件名与干净点云一一对应，例如 clean/a.txt 对应 noise/a.txt")

    def clear_noise(self):
        self.noise_file = ""
        self.noise_dir = ""
        self.refresh_noise_label()
        self.save_settings()
        self.log.append("已取消噪声点云文件/文件夹")

    def refresh_noise_label(self):
        parts = []
        if self.noise_file:
            parts.append(f"单文件：{self.noise_file}")
        if self.noise_dir:
            parts.append(f"批量文件夹：{self.noise_dir}（文件名需与干净点云相同）")
        self.lbl_noise.setText("；".join(parts) if parts else "未选择")

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
        file, _ = QFileDialog.getOpenFileName(self, "选择标签文件", "", "TXT Files (*.txt);;CSV Files (*.csv);;All Files (*)")
        if file:
            self.label_file = file
            self.lbl_label.setText(file)
            try:
                labels = load_labels_from_txt(file, multiclass=self.multiclass_check.isChecked())
                self.remember_label_classes(labels)
            except Exception as e:
                self.log.append(f"⚠️ 标签文件读取失败：{e}")
            self.save_settings()

    def clear_label(self):
        self.label_file = ""
        self.lbl_label.setText("未选择")
        self.save_settings()
        self.log.append("已取消标签文件")

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
            self.log.append(f"⚠️ 按当前标签设置重新读取失败：{e}")

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
                self.lbl_colorbar_caption.setText(
                    f"多分类色轴预览（{self._label_class_count} 类，按当前色轴自动分配）"
                )
            else:
                self.lbl_colorbar_caption.setText("多分类色轴预览（载入标签后按当前色轴自动分配）")
        else:
            self.lbl_colorbar_caption.setText("渐变色轴预览（无标签时使用）")
        self.update_colorbar()

    def refresh_label_source_ui(self):
        embedded = self.labels_in_cloud_check.isChecked()
        self.btn_label.setEnabled(not embedded)
        self.btn_clear_label.setEnabled(not embedded)
        self.label_col_box.setEnabled(embedded)
        if embedded:
            col = self.label_col_box.value()
            self.lbl_label.setText(f"每个点云文件都使用第 {col} 列作为标签")
        elif self.label_file:
            self.lbl_label.setText(self.label_file)
        else:
            self.lbl_label.setText("未选择")

    def choose_output_xml(self):
        default_path = self.output_xml if self.output_xml_manual and self.output_xml else self.get_default_output_xml(self.terrain_file) if self.terrain_file else "scene.xml"
        file, _ = QFileDialog.getSaveFileName(self, "选择输出 XML 路径", default_path, "XML Files (*.xml)")
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
        self.log.append("已取消手动输出路径，恢复为自动生成")

    # -------------------- 渲染 / Loading --------------------

    def _build_loading_overlay(self):
        self.loading_overlay = QWidget(self)
        self.loading_overlay.setObjectName("loadingOverlay")
        self.loading_overlay.setAttribute(Qt.WA_StyledBackground, True)
        self.loading_overlay.setStyleSheet("#loadingOverlay { background: rgba(20, 20, 20, 160); }")
        box = QVBoxLayout(self.loading_overlay)
        box.setAlignment(Qt.AlignCenter)
        box.setSpacing(12)
        self.loading_title = QLabel("正在渲染，请稍候…")
        self.loading_title.setAlignment(Qt.AlignCenter)
        self.loading_title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; background: transparent;")
        self.loading_status = QLabel("准备中")
        self.loading_status.setAlignment(Qt.AlignCenter)
        self.loading_status.setStyleSheet("color: #eee; font-size: 13px; background: transparent;")
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setFixedWidth(280)
        self.loading_bar.setTextVisible(False)
        self.btn_cancel_render = QPushButton("取消渲染")
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
        self.loading_title.setText(f"正在渲染，请稍候… 已用时 {self._loading_seconds} 秒")

    def _set_loading(self, on, status="准备中"):
        self._sync_loading_overlay()
        if on:
            self._loading_seconds = 0
            self._cancel_requested = False
            self.loading_title.setText("正在渲染，请稍候…")
            self.loading_status.setText(status)
            self.btn_cancel_render.setEnabled(True)
            self.btn_cancel_render.setText("取消渲染")
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
        self.btn_cancel_render.setText("正在取消…")
        self.loading_title.setText("正在取消渲染，请稍候…")
        self.loading_status.setText("正在停止当前任务")
        self.log.append("⏹️ 已请求取消渲染")
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
        }

    def _start_render_jobs(self, jobs, done_message=""):
        if self._rendering:
            return
        self._rendering = True
        self._render_done_message = done_message
        self._set_loading(True, f"队列中共 {len(jobs)} 个任务")

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
            self.log.append("⏹️ 渲染已取消。未完成的图像不会保存。")
        elif done_message:
            self.log.append(done_message)
        self._render_thread = None
        self._render_worker = None
        if self._quit_when_idle:
            self._quit_when_idle = False
            QTimer.singleShot(0, self.close)

    def render_scene(self):
        if self._rendering:
            QMessageBox.information(self, "请稍候", "当前正在渲染。如需停止，请点击“取消渲染”或按 Esc。")
            return

        jobs = []
        done_message = ""
        if self.batch_check.isChecked():
            if not self.terrain_dir:
                QMessageBox.warning(self, "错误", "请先选择点云文件夹，或取消批量模式。")
                return
            if self.labels_in_cloud_check.isChecked():
                mode = "多分类" if self.multiclass_check.isChecked() else "二分类"
                col = self.label_col_box.value()
                self.log.append(f"🏷️ 批量模式将从每个点云文件的第 {col} 列读取{mode}标签。")
            elif self.label_file:
                self.log.append("⚠️ 当前批量模式会忽略单个标签文件，请勾选“点云含标签”并指定标签列。")

            if self.noise_dir:
                self.log.append(f"🌫️ 批量噪声文件夹：{self.noise_dir}。噪声文件名必须与对应干净点云相同。")
            elif self.noise_file:
                self.log.append("⚠️ 批量模式不会使用单个噪声文件，请选择噪声文件夹，并保证文件名与干净点云一一对应。")

            txts = sorted([
                os.path.join(self.terrain_dir, f)
                for f in os.listdir(self.terrain_dir)
                if f.lower().endswith(".txt")
            ])
            if not txts:
                QMessageBox.information(self, "提示", "该文件夹下未找到 .txt 点云文件。")
                return

            self.log.append(f"📁 将批量渲染 {len(txts)} 个文件：")
            for p in txts:
                self.log.append(f"- {os.path.basename(p)}")
                matched_noise = self.find_matching_noise_file(p) if self.noise_dir else None
                if self.noise_dir and matched_noise is None:
                    self.log.append(
                        f"⚠️ 未找到与 {os.path.basename(p)} 对应的噪声文件（文件名需相同），本文件不叠加噪声。"
                    )
                jobs.append(self._snapshot_render_job(p, preview=False, noise_path=matched_noise))
            self.log.append("开始批量渲染...")
            done_message = "🎉 批量渲染完成！如需查看单个结果，请切换到单文件模式并载入 TIFF 预览。"
        else:
            if not self.terrain_file:
                QMessageBox.warning(self, "错误", "请先选择点云文件（或启用批量并选择文件夹）。")
                return
            self.refresh_output_label()
            self.log.append("开始渲染（单文件）...")
            jobs.append(self._snapshot_render_job(
                self.terrain_file,
                preview=True,
                noise_path=self.noise_file or None,
            ))
            done_message = "🎉 单文件渲染完成。"

        self._start_render_jobs(jobs, done_message)

    def reload_tiff_preview(self):
        if self._rendering:
            QMessageBox.information(self, "请稍候", "正在渲染。请等待完成，或先取消当前渲染。")
            return
        if not self.terrain_file:
            QMessageBox.information(self, "提示", "请先在单文件模式下选择点云并完成一次渲染。")
            return
        tiff_path = self.get_output_xml_path(self.terrain_file).replace(".xml", ".tiff")
        if not os.path.exists(tiff_path):
            QMessageBox.information(self, "提示", "未找到TIFF文件，请先渲染或确认路径。")
            return
        try:
            self.show_image_on_label(read_tiff(tiff_path), self.render_preview)
            self.log.append("🔄 已从TIFF重新载入预览")
        except Exception as e:
            self.log.append(f"⚠️ 预览载入失败：{e}")


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
