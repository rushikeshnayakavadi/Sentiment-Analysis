import mlflow
import dagshub

# ✅ Set tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/rushikeshnayakavadi/Sentiment-Analysis.mlflow")
dagshub.init(repo_owner="rushikeshnayakavadi", repo_name="Sentiment-Analysis", mlflow=True)

def get_latest_model_version(model_name):
    """
    Fetches the latest registered version of the model from MLflow.
    """
    client = mlflow.MlflowClient()
    
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"✅ Model '{model_name}' found!")
        
        # Fetch all available versions
        latest_versions = sorted(
            [int(v.version) for v in client.search_model_versions(f"name='{model_name}'")], 
            reverse=True
        )
        
        print(f"📌 Available Versions (sorted): {latest_versions}")
        return latest_versions[0] if latest_versions else None  # Returns the highest version
    except Exception as e:
        print(f"❌ Model '{model_name}' not found. Error: {e}")
        return None

if __name__ == "__main__":
    model_name = "my_model"
    model_version = get_latest_model_version(model_name)
    
    if model_version:
        print(f"✅ Latest Version: {model_version}")
    else:
        print(f"❌ Model '{model_name}' not found in MLflow.")
