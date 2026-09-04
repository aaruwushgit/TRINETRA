import shutil
from huggingface_hub import hf_hub_download

# This downloads the best model weights from the cloud
model_path = hf_hub_download(repo_id="Babblu2821/alpr-plate-detector", filename="best.pt")

# Copy the file to the current directory so ALPR can find it
shutil.copy(model_path, "best.pt")
print(f"Model downloaded and copied to current directory: best.pt")
