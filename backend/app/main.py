from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Recycle AI Server Running"
    }
    