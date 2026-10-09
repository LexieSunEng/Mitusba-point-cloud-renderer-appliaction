# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

# Dr.Jit's native extension does ``import drjit.llvm`` (and cuda/scalar) from C++
# via nanobind. PyInstaller cannot see those imports by scanning Python sources.
drjit_datas, drjit_binaries, drjit_hidden = collect_all(
    'drjit',
    include_py_files=True,
    exclude_datas=['include', 'cmake', '*.pyi', '*.lib'],
)

# Do NOT use collect_all / collect_submodules on top-level ``mitsuba``.
# ``mitsuba.mitsuba_stubs`` recurses infinitely (mitsuba3#1487), and importing
# every compiled variant re-registers nanobind types until the build hangs.
mitsuba_datas = collect_data_files(
    'mitsuba',
    include_py_files=True,
    excludes=['mitsuba_stubs', 'include', 'cmake', '*.pyi', '*.h', '*.hpp', '*.cmake', '*.lib'],
)
mitsuba_binaries = collect_dynamic_libs('mitsuba')

hiddenimports = list(dict.fromkeys(
    list(drjit_hidden) + [
        'drjit._drjit_ext',
        'drjit.llvm',
        'drjit.llvm.ad',
        'drjit.cuda',
        'drjit.cuda.ad',
        'drjit.scalar',
        'drjit.auto',
        'drjit.auto.ad',
        'drjit.detail',
        'drjit.config',
        'typing_extensions',
        'mitsuba.config',
        'mitsuba.mitsuba_ext',
        'mitsuba.mitsuba_alias',
        'mitsuba.mitsuba_scalar_rgb_ext',
        'mitsuba.mitsuba_cuda_ad_rgb_ext',
        'mitsuba.mitsuba_llvm_ad_rgb_ext',
        'mitsuba.mitsuba_scalar_spectral_ext',
        'mitsuba.python',
        'mitsuba.python.util',
        'Imath',
    ]
))

# Variant modules are loaded by name at runtime (mitsuba.mitsuba_<variant>_ext).
# PyInstaller does not see that import, so add the extension binaries directly.
import importlib.util
_mitsuba_root = os.path.dirname(importlib.util.find_spec('mitsuba').origin)
_mitsuba_exts = [
    (os.path.join(_mitsuba_root, name), 'mitsuba')
    for name in os.listdir(_mitsuba_root)
    if name.endswith('.pyd')
]

def _setuptools_backports_datas():
    """pkg_resources on Python 3.8 imports ``backports.tarfile`` from setuptools/_vendor."""
    vendor = os.path.join(os.path.dirname(importlib.util.find_spec('setuptools').origin), '_vendor', 'backports')
    items = []
    for dirpath, _, filenames in os.walk(vendor):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            rel_dir = os.path.relpath(dirpath, vendor)
            dest = os.path.join('setuptools', '_vendor', 'backports')
            if rel_dir != '.':
                dest = os.path.join(dest, rel_dir)
            items.append((os.path.join(dirpath, filename), dest))
    return items


datas = drjit_datas + mitsuba_datas + _setuptools_backports_datas()
binaries = drjit_binaries + mitsuba_binaries + _mitsuba_exts

a = Analysis(
    ['mistuba_tool.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/pyi_rth_native_dlls.py'],
    excludes=[
        'mitsuba.mitsuba_stubs',
        # This environment also contains training libraries. The renderer does not import them.
        'torch',
        'torchvision',
        'torchaudio',
        'ultralytics',
        'tensorboard',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='mistuba_tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[r'E:\Windshear_packed_new\utils\myinstaller\myicon.ico'] if os.path.isfile(r'E:\Windshear_packed_new\utils\myinstaller\myicon.ico') else None,
)
