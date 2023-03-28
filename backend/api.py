from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from url_controller import router as url_router
from users_controllers import router as users_router
from model_loader import ModelLoader

app = FastAPI()

@app.on_event("startup")
def load_model():
    # app.state.model = ModelLoader(
    #     path="models/rfc/rfc.sav", name="RandomForest", model_dir="sklearn_rfc"
    # )
    # app.state.model = ModelLoader(
    #     path="models/svc/svc.sav", name="SVC", model_dir="sklearn_svc"
    # )
    # app.state.model = ModelLoader(
    #     path="models/clf/clf.sav", name="MultinomialNB", model_dir="sklearn_clf"
    # )
    app.state.model = ModelLoader(
        path="models/m1/model.sav", name="Best_Model", model_dir="sklearn_m1"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(
    url_router,
    prefix="/url",
    tags=["url"],
    responses={404: {"description": "Not found"}},
)

app.include_router(
    users_router,
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@app.get("/hi")
def root():
    return {"message": "Hello World from FastAPI and Docker Deployment course"}
