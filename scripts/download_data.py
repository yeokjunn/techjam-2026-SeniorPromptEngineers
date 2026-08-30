import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    archive_path = data_dir / "KuaiRand-Pure.tar.gz"
    url = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
    expected_md5 = "0820331067a3784d9691136f772b35a7"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading KuaiRand-Pure (~280 MB)...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as e:
        print(f"Failed to download: {e}")
        return

    print("Verifying checksum...")
    md5 = hashlib.md5()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    actual_md5 = md5.hexdigest()

    if actual_md5 != expected_md5:
        print(f"Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return

    print("Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    print("Cleaning up archive...")
    archive_path.unlink()

    print("Done! Data extracted to data/KuaiRand-Pure/data")

if __name__ == "__main__":
    main()

