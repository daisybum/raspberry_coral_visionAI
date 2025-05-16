import os, time, uuid, logging, redis
from pathlib import Path

REDIS = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)
INTERVAL = int(os.getenv("INTERVAL", 30))
CHANNEL  = "img_channel"
CAP_CMD  = "libcamera-still -n -o {dst} --width 1640 --height 1232"
TMP_IMG  = Path("/tmp/cap.jpg")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [CAP] %(message)s")

while True:
    os.system(CAP_CMD.format(dst=TMP_IMG))
    if not TMP_IMG.exists():
        logging.warning("capture failed"); time.sleep(INTERVAL); continue

    raw = TMP_IMG.read_bytes()
    key = f"img:{uuid.uuid4().hex}"
    REDIS.setex(key, 60, raw)            # 60 초 TTL
    REDIS.publish(CHANNEL, key)          # 구독자에게 알림
    TMP_IMG.unlink(missing_ok=True)

    logging.info(f"published {key}")
    time.sleep(INTERVAL)
