"""Light extraction stubs (milestones 4–5).

These will be filled out after the viewer + DA² subprocess pipeline lands.
Public surface, sketched:

    extract.crop_rect(hdr, lr) -> np.ndarray
    extract.unproject(lr, distance) -> np.ndarray   # (N, 3) world points
    fitplane.fit(points) -> RectFit                 # center, normal, u, v, w, h
    inpaint.dome_residual(hdr, masks) -> np.ndarray
    photometry.intensity(crop) -> float

For now this module is a placeholder so the package imports cleanly.
"""
