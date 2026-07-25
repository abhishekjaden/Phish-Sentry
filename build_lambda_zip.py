#!/usr/bin/env python3
"""
Build the Lambda zip for the lookalike detector, then VERIFY the packaged code
gives identical verdicts to the local version.

The verification matters: tldextract's public-suffix handling differs between
environments, and an unnoticed difference would mean the deployed endpoint
behaves unlike everything that was tested. The detector is written to be
parse-independent, but this checks that rather than assuming it.

Usage:
  python build_lambda_zip.py
Produces: lookalike_lambda.zip
"""
import os
import shutil
import subprocess
import sys
import zipfile

BUILD = "build_lambda"
ZIP = "lookalike_lambda.zip"
FILES = ["lookalike_detector.py", "lambda_handler_lookalike.py"]
DEPS = ["tldextract"]

# Same cases as the detector self-test -- packaged code must agree exactly.
CHECKS = [
    ("https://www.paypal.com/login", False),
    ("https://paypal.com/signin", False),
    ("https://microsoft.github.io/monaco-editor/", False),
    ("https://google.github.io/styleguide/", False),
    ("https://johnsmith.github.io/portfolio/", False),
    ("https://docs.pages.dev/", False),
    ("https://myportfolio.pages.dev/", False),
    ("https://stackoverflow.com/questions/tagged/python", False),
    ("https://www.startups.com/", False),
    ("http://paypa1-secure.com/account/verify", True),
    ("https://paypalsupport.com/", True),
    ("https://verify-dhl.com/signin", True),
    ("https://ups-tracking-update.com/", True),
    ("https://instagram-cmd.github.io/accounts/index.html", True),
    ("https://paypal-verify.pages.dev/", True),
    ("https://www.roblox.com.am/users/378768172502/profile", True),
]


def main():
    for f in FILES:
        if not os.path.exists(f):
            print(f"Missing {f}")
            return 1

    if os.path.exists(BUILD):
        shutil.rmtree(BUILD)
    os.makedirs(BUILD)

    print("installing dependencies for linux/x86_64 ...")
    rc = subprocess.call([
        sys.executable, "-m", "pip", "install",
        "--platform", "manylinux2014_x86_64",
        "--target", BUILD,
        "--implementation", "cp",
        "--python-version", "3.11",
        "--only-binary=:all:",
        "--upgrade", "--quiet",
    ] + DEPS)
    if rc != 0:
        print("platform-specific install failed; falling back to plain install")
        print("(fine here -- tldextract is pure Python)")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--target", BUILD, "--upgrade", "--quiet"] + DEPS)

    for f in FILES:
        shutil.copy(f, BUILD)

    # verify the packaged copy behaves identically
    print("\nverifying packaged code ...")
    sys.path.insert(0, os.path.abspath(BUILD))
    for mod in ("lookalike_detector", "lambda_handler_lookalike"):
        sys.modules.pop(mod, None)
    try:
        from lookalike_detector import analyze
    except Exception as e:
        print(f"  import failed: {type(e).__name__}: {e}")
        return 1

    bad = 0
    for url, want in CHECKS:
        got = analyze(url)["is_lookalike"]
        if got != want:
            bad += 1
            print(f"  MISMATCH  want={want} got={got}  {url}")
    if bad:
        print(f"\n{bad} mismatch(es) -- do NOT deploy this zip.")
        return 1
    print(f"  all {len(CHECKS)} checks passed")

    if os.path.exists(ZIP):
        os.remove(ZIP)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(BUILD):
            for name in files:
                if name.endswith((".pyc",)) or "__pycache__" in root:
                    continue
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, BUILD))

    size = os.path.getsize(ZIP) / 1024 / 1024
    print(f"\nwrote {ZIP} ({size:.1f} MB)")
    print("For comparison, the model-based container image was ~1 GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
