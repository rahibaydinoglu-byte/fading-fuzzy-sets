"""Download the public SNAP Bitcoin OTC and Bitcoin Alpha trust networks into ./data/."""
import gzip, shutil, urllib.request, pathlib
URLS = {"soc-sign-bitcoinotc.csv": "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz",
        "soc-sign-bitcoinalpha.csv": "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"}
pathlib.Path("data").mkdir(exist_ok=True)
for name, url in URLS.items():
    gz = pathlib.Path("data") / (name + ".gz")
    urllib.request.urlretrieve(url, gz)
    with gzip.open(gz, "rb") as f, open(pathlib.Path("data") / name, "wb") as g: shutil.copyfileobj(f, g)
    gz.unlink(); print("saved data/" + name)
