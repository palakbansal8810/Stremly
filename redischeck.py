import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
r.set('test', 'Hello from Memurai!')
print(r.get('test'))
import joblib
chunks = joblib.load("embeddings\data_76780302ded643719a0fa7e65b559d66.pkl")
print(type(chunks), type(chunks[0]))