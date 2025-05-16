# LKH-3
import os
import tarfile

from robust_downloader import download


def download_lkh():
    """
    Download and extract the LKH-3.0.13 tarball from the given URL.
    """
    url = "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.13.tgz"
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        download(url, ".", filename)
        print("Download complete.")

    # Extract the tarball
    if not os.path.exists("LKH-3.0.13"):
        print(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("Extraction complete.")

    # remove the tarball
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Removed {filename}.")
    else:
        print(f"{filename} not found, skipping removal.")


if __name__ == "__main__":
    download_lkh()
