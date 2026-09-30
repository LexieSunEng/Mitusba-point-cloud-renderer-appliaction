# -*- mode: python ; coding: utf-8 -*-

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
        'mitsuba.python',
        'mitsuba.python.util',
        'mitsuba.scalar_rgb',
        'mitsuba.cuda_ad_rgb',
        'mitsuba.llvm_ad_rgb',
        'Imath',
    ]
))

datas = drjit_datas + mitsuba_datas
binaries = drjit_binaries + mitsuba_binaries

a = Analysis(
    ['mistuba_tool.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['hooks/pyi_rth_native_dlls.py'],
    excludes=['mitsuba.mitsuba_stubs'],
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
    icon=['E:\\Windshear_packed_new\\utils\\myinstaller\\myicon.ico'],
)
