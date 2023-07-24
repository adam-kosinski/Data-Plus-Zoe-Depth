# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files


block_cipher = None

datas = [('gui.ui', '.'), ('hubconf.py', '.'), ('zoedepth/models', './zoedepth/models'), ('megadetector_weights', 'megadetector_weights')]
datas += collect_data_files('timm', include_py_files=True)
datas += collect_data_files('ultralytics')

a = Analysis(
    ['gui.py'],
    pathex=['MegaDetector', 'yolov5'],
    binaries=[],
    datas=datas,
    hiddenimports=['timm'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Wildlife Depth Estimation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Wildlife Depth Estimation',
)
