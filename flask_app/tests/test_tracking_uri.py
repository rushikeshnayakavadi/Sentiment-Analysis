import mlflow
import dagshub

# ✅ Set tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/rushikeshnayakavadi/Sentiment-Analysis.mlflow")
dagshub.init(repo_owner="rushikeshnayakavadi", repo_name="Sentiment-Analysis", mlflow=True)

def check_tracking_uri():
    """
    Print and confirm the MLflow tracking URI.
    """
    tracking_uri = mlflow.get_tracking_uri()
    print(f"🔍 Current MLflow Tracking URI: {tracking_uri}")

if __name__ == "__main__":
    check_tracking_uri()
