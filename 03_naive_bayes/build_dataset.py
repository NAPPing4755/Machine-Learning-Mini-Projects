# Builds _datasets/enron_aueb/enron_spam_aueb.csv from the ORIGINAL Enron-Spam archives (Metsis, Androutsopoulos
# and Paliouras, 2006). Why not the popular Kaggle CSV? Its 17,171 "spam" rows contain copies of genuine Enron
# e-mails instead of spam (see README), so it cannot be used.
import re                                         # find the date in each file name
import tarfile                                    # read .tar.gz archives without extracting them to disk
import urllib.request                             # download the archives
from pathlib import Path                          # file paths

import pandas as pd                               # build and save the table

OUT = Path(__file__).resolve().parent.parent / "_datasets" / "enron_aueb"   # big data lives outside the repo
URL = "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron{}.tar.gz"   # the authors' own copies


def main():
    OUT.mkdir(parents=True, exist_ok=True)        # create the folder if needed
    rows = []                                     # one dict per e-mail
    for i in range(1, 7):                         # six mailboxes: enron1 ... enron6
        path = OUT / f"enron{i}.tar.gz"           # local copy of this archive
        if not path.exists():                     # download only once
            urllib.request.urlretrieve(URL.format(i), path)
        with tarfile.open(path) as tf:            # open the archive
            for m in tf.getmembers():             # every file inside
                label = "spam" if "/spam/" in m.name else "ham" if "/ham/" in m.name else None   # folder = label
                if not m.isfile() or label is None:   # skip folders and anything else
                    continue
                raw = tf.extractfile(m).read().decode("latin-1")   # read in memory (never written to disk)
                first, _, body = raw.partition("\n")   # line 1 is "Subject: ...", the rest is the body
                subject = first[8:].strip() if first.lower().startswith("subject:") else ""   # drop "Subject:"
                date = re.search(r"\.(\d{4}-\d{2}-\d{2})\.", m.name)   # e.g. 0001.1999-12-10.kaminski.ham.txt
                rows.append({"mailbox": f"enron{i}", "subject": subject, "body": body.strip(),
                             "label": label, "date": date.group(1) if date else ""})
    df = pd.DataFrame(rows)                       # 33,716 rows
    df.to_csv(OUT / "enron_spam_aueb.csv", index=False)   # the file naive_bayes.py reads
    print(pd.crosstab(df["mailbox"], df["label"]))   # ham/spam per mailbox


if __name__ == "__main__":
    main()
