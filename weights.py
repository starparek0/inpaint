import subprocess
import os

def download_weights(url, dest):
    command = f"pget {url} -o {dest}"
    try:
        subprocess.check_call(command, shell=True)
        print("Weights downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print("Error downloading weights:", e)
        raise

if __name__ == "__main__":
    # Podmień poniższy URL na właściwy adres do wag
    weights_url = "https://example.com/weights.bin"
    destination = "./weights/weights.bin"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    download_weights(weights_url, destination)
