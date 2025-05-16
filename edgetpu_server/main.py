from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from filelock import FileLock

from pycoral.utils import edgetpu
from pycoral.adapters import common, segment, classify

SEG_PATH = Path.getenv("SEG_MODEL", "/models/seg.tflite")
CLS_PATH = Path.getenv("CLS_MODEL", "/models/cls.tflite")
LOCK_PATH = Path("/tmp/edgetpu.lock")          # 컨테이너 내부 전역 락

# ─── Interpreter preload ──────────────────────────────────
seg_interp = edgetpu.make_interpreter(str(SEG_PATH), device="usb")
cls_interp = edgetpu.make_interpreter(str(CLS_PATH), device="usb")
seg_interp.allocate_tensors()
cls_interp.allocate_tensors()
seg_w, seg_h = common.input_size(seg_interp)
cls_w, cls_h = common.input_size(cls_interp)

# ─── FastAPI 라우터 ────────────────────────────────────────
app = FastAPI(title="Edge-TPU Gateway", version="1.0")


r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, decode_responses=False)

class ImgReq(BaseModel):
    redis_key: str          # seg_app / cls_app 이 SET 한 key


def _load_image_from_redis(key: str, target_size):
    raw = r.get(key)
    if raw is None:
        raise HTTPException(404, "image key expired or not found")
    pil = Image.open(io.BytesIO(raw)).convert("RGB").resize(target_size)
    return np.asarray(pil).astype(np.uint8)

@app.post("/segment")
def segment_image(req: ImgReq):
    np_img = _load_image_from_redis(req.redis_key, (seg_w, seg_h))
    with FileLock("/tmp/edgetpu.lock"):
        common.set_input(seg_interp, np_img)
        seg_interp.invoke()
        mask = segment.get_output(seg_interp)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    r.delete(req.redis_key)                      # 메모리 회수
    return {"mask_shape": mask.shape,
            "unique_labels": np.unique(mask).tolist()}

@app.post("/classify")
def classify_image(req: ImgReq):
    np_img = _load_image_from_redis(req.redis_key, (cls_w, cls_h))
    with FileLock("/tmp/edgetpu.lock"):
        common.set_input(cls_interp, np_img)
        cls_interp.invoke()
        pred = classify.get_classes(cls_interp, top_k=1)[0]
    r.delete(req.redis_key)
    return {"id": pred.id, "score": float(pred.score)}
