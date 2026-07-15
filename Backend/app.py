from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import time

from classifiers.regex_classifier import predict_regex
from classifiers.statistical_ml_classifier import predict_ml
from classifiers.llm_classifier import predict_llm

app = FastAPI(title="AI Email Spam Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=3)

from pydantic import BaseModel, field_validator

class EmailRequest(BaseModel):
    email_text: str

    @field_validator("email_text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("email_text cannot be empty")
        if len(v) > 5000:
            raise ValueError("email_text exceeds 5000 character limit")
        return v

@app.get("/")
def home():
    return {"message": "Backend running"}

@app.post("/analyze")
def analyze_email(request: EmailRequest):
    start_time = time.time()

    future_regex = executor.submit(predict_regex, request.email_text)
    future_ml = executor.submit(predict_ml, request.email_text)
    future_llm = executor.submit(predict_llm, request.email_text)

    regex_result = future_regex.result()
    ml_result = future_ml.result()
    llm_result = future_llm.result()

    end_time = time.time()

    votes = [regex_result["prediction"], ml_result["prediction"], llm_result["prediction"]]
    spam_votes = votes.count("spam")
    ham_votes = votes.count("ham")

    if spam_votes > ham_votes:
        final_verdict = "spam"
        agreement = f"{spam_votes}/3"
    else:
        final_verdict = "ham"
        agreement = f"{ham_votes}/3"

    return {
        "regex": regex_result,
        "ml": ml_result,
        "llm": llm_result,
        "final_verdict": final_verdict,
        "agreement": agreement,
        "processing_time": round(end_time - start_time, 2)
    }

#Old code(SEQUENTIAL EXECUTION)
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from classifiers.regex_classifier import predict_regex
# from classifiers.statistical_ml_classifier import predict_ml
# from classifiers.llm_classifier import predict_llm
# import time

# # Initialises FastAPI
# app = FastAPI(title="AI Email Spam Classifier")

# # Enables cross-origin requests (allows frontend on a different port/domain to call this API)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request schema - expects a single email_text string
# class EmailRequest(BaseModel):
#     email_text: str

# # Health check endpoint
# @app.get("/")
# def home():
#     return {"message": "Backend running"}

# # Analyzes the email based on teh output of the 3 classifiers
# @app.post("/analyze")
# def analyze_email(request: EmailRequest):
#     start_time = time.time()

#     regex_result = predict_regex(request.email_text)
#     ml_result = predict_ml(request.email_text)
#     llm_result = predict_llm(request.email_text)

#     end_time = time.time()

#     votes = [
#         regex_result["prediction"],
#         ml_result["prediction"],
#         llm_result["prediction"]
#     ]

#     # Majority vote across 3 classifiers to determine final verdict
#     spam_votes = votes.count("spam")
#     ham_votes = votes.count("ham")

#     if spam_votes > ham_votes:
#         final_verdict = "spam"
#         agreement = f"{spam_votes}/3"
#     else:
#         final_verdict = "ham"
#         agreement = f"{ham_votes}/3"

#     return {
#         "regex": regex_result,
#         "ml": ml_result,
#         "llm": llm_result,
#         "final_verdict": final_verdict,
#         "agreement": agreement,
#         "processing_time": round(end_time - start_time, 2)
#     }