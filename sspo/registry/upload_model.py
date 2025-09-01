import os
from pathlib import Path
from xgboost import XGBRegressor
import pickle
from google.cloud import storage
from sspo.registry.load_model import load_xgb_reg

def save_xgb_reg(xgb_reg: XGBRegressor, filename: str) -> None:
    """Saves the trained model into the cloud."""
    print("\n--- Saving Model ---")

    BUCKET_NAME = os.environ["BUCKET_NAME"]
    storage_filename = f"models/{filename}.pkl"
    local_filename = "model.pkl"

    file_path = Path(local_filename)
    pickle.dump(xgb_reg, open(file_path, "wb"))

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_filename(local_filename)

    print(f"💾 Model in '{file_path}' successfully saved to Google Cloud as {filename}.pkl")


if __name__ == "__main__":
    print("=============================================")
    print("            Starting load_model.py           ")
    print("=============================================")

    save_xgb_reg(load_xgb_reg("xgb_reg_24_54"), "xgb_reg_24_54_gc_test")

    print("\n=============================================")
    print("          Script finished successfully       ")
    print("=============================================")
