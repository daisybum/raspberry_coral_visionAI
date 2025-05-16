import os, logging, redis, requests
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [SEG] %(message)s")

r     = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)
p     = r.pubsub(); p.subscribe("img_channel")
ENDPT = os.getenv("TPU_ENDPOINT", "http://tpu_server:8080")

for msg in p.listen():
    if msg["type"] != "message": continue
    key = msg["data"].decode()
    res = requests.post(f"{ENDPT}/segment", json={"redis_key": key}, timeout=15)
    logging.info(res.json())
