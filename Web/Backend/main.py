from fastapi import FastAPI
from app.routers import auth, projects, upload

app = FastAPI()

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(upload.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the ECG Story Generator API"}
