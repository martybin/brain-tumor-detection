"""Local MRI review dashboard.

The page can show an uploaded study and, when a trained checkpoint exists,
the research model's region mask. It never presents that mask as a diagnosis.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from dashboard.render import flat_png, slice_png
from dashboard.sessions import (
    checkpoint_path,
    load_original,
    new_session,
    read_meta,
    save_flat_image,
    save_nifti_upload,
    volume_shape,
    write_meta,
    write_sample,
)
from preprocessing.io_utils import CANONICAL_MODALITIES

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Brain tumor research viewer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

DISCLAIMER = "این خروجی پژوهشی است و تشخیص پزشکی نیست. تصمیم بالینی با پزشک و رادیولوژیست است."


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/font/{name}")
def font(name: str) -> FileResponse:
    allowed = {
        "NotoSansArabic-Regular.ttf": "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "NotoSansArabic-Bold.ttf": "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    }
    path = allowed.get(name)
    if path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="Font not found.")
    return FileResponse(path)


@app.get("/api/health")
def health() -> dict:
    checkpoint = checkpoint_path()
    return {
        "clinical_validity": False,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_ready": checkpoint is not None,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/sessions/sample")
def sample_session() -> dict:
    folder = new_session()
    meta = write_sample(folder)
    meta["clinical_validity"] = False
    meta["disclaimer"] = DISCLAIMER
    return _public_meta(meta)


@app.post("/api/sessions")
async def create_session(
    t1: UploadFile | None = File(default=None),
    t1ce: UploadFile | None = File(default=None),
    t2: UploadFile | None = File(default=None),
    flair: UploadFile | None = File(default=None),
    picture: UploadFile | None = File(default=None),
) -> dict:
    folder = new_session()
    meta = {"id": folder.name, "modalities": {}, "image": None, "result": None, "sample": False}
    uploads = {"t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}
    try:
        for modality, upload in uploads.items():
            if upload is None or not upload.filename:
                continue
            payload = await upload.read()
            if not payload:
                continue
            save_nifti_upload(folder, modality, upload.filename, payload)
            suffix = ".nii.gz" if upload.filename.lower().endswith(".nii.gz") else ".nii"
            meta["modalities"][modality] = f"{folder.name}_{modality}{suffix}"
        if picture is not None and picture.filename:
            payload = await picture.read()
            if payload:
                save_flat_image(folder, picture.filename, payload)
                meta["image"] = "flat_image"
        if not meta["modalities"] and not meta["image"]:
            raise HTTPException(status_code=400, detail="فایلی دریافت نشد.")
        shape = volume_shape(folder, meta)
        meta["shape"] = shape
        write_meta(folder, meta)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _public_meta(meta)


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    try:
        _, meta = read_meta(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="مطالعه پیدا نشد.") from exc
    return _public_meta(meta)


@app.get("/api/sessions/{session_id}/slice")
def session_slice(
    session_id: str,
    modality: str = "flair",
    axis: str = "axial",
    index: int = 0,
    overlay: int = 0,
    space: str = "original",
) -> Response:
    try:
        folder, meta = read_meta(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="مطالعه پیدا نشد.") from exc
    if meta.get("image") and not meta.get("modalities"):
        payload = (folder / "original" / "flat_image").read_bytes()
        return Response(content=flat_png(payload), media_type="image/png")
    if modality not in CANONICAL_MODALITIES:
        raise HTTPException(status_code=400, detail="سکانس ناشناخته است.")
    try:
        volume = load_original(folder, meta, modality, space=space)
        mask = _mask_volume(folder, meta) if overlay else None
        png = slice_png(volume, axis, index, overlay=mask)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="این سکانس بارگذاری نشده.") from exc
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.post("/api/sessions/{session_id}/predict")
def predict_session(session_id: str) -> dict:
    try:
        folder, meta = read_meta(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="مطالعه پیدا نشد.") from exc
    if meta.get("image") and not meta.get("modalities"):
        result = _unavailable(
            "عکس دوبعدی برای این مدل کافی نیست. چهار حجم NIfTI یعنی T1، T1ce، T2 و FLAIR لازم است."
        )
        meta["result"] = result
        write_meta(folder, meta)
        return result
    missing = [name for name in CANONICAL_MODALITIES if name not in meta.get("modalities", {})]
    if missing:
        result = _unavailable(f"سکانس‌های ناقص را نمی‌توان نادیده گرفت: {', '.join(missing)}")
        meta["result"] = result
        write_meta(folder, meta)
        return result
    checkpoint = checkpoint_path()
    if checkpoint is None:
        result = _unavailable(
            "هنوز چک‌پوینت آموزش‌دیده‌ای وجود ندارد، پس مدل نمی‌تواند وجود تومور را بررسی کند."
        )
        meta["result"] = result
        write_meta(folder, meta)
        return result
    try:
        from dashboard.infer import run_case
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"وابستگی مدل نصب نیست: {exc}") from exc
    try:
        result = run_case(folder, meta, checkpoint)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    meta["result"] = {key: value for key, value in result.items() if key != "mask"}
    write_meta(folder, meta)
    return meta["result"]


def _mask_volume(folder: Path, meta: dict):
    mask_name = (meta.get("result") or {}).get("mask_file")
    if not mask_name:
        return None
    path = folder / mask_name
    if not path.is_file():
        return None
    from preprocessing.io_utils import load_nifti

    data, _ = load_nifti(path, expected_ndim=(3,))
    return data


def _unavailable(message: str) -> dict:
    return {
        "available": False,
        "marked": None,
        "clinical_validity": False,
        "regions": {},
        "message": message,
        "disclaimer": DISCLAIMER,
    }


def _public_meta(meta: dict) -> dict:
    return {
        "id": meta["id"],
        "modalities": sorted(meta.get("modalities") or []),
        "image": bool(meta.get("image")),
        "sample": bool(meta.get("sample")),
        "shape": meta.get("shape"),
        "result": meta.get("result"),
        "clinical_validity": False,
        "disclaimer": DISCLAIMER,
    }
