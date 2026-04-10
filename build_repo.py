import os
import shutil
from pathlib import Path

def setup_repo():
    src_dir = Path("/Users/sumeetdas/Antigravity_NSE_Data")
    dest_dir = Path("/Users/sumeetdas/Documents/NSE-Scanners-Automated")
    
    # Create dirs
    scripts_dir = dest_dir / "scripts"
    gha_dir = dest_dir / ".github" / "workflows"
    data_dir = dest_dir / "data"
    
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(gha_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Copy and modify lineage scripts
    def copy_and_fix(filename):
        text = (src_dir / filename).read_text()
        # Replace absolute paths with relative paths
        text = text.replace('"/Users/sumeetdas/Antigravity_NSE_Data/nse_', '"data/nse_')
        text = text.replace("'/Users/sumeetdas/Antigravity_NSE_Data/nse_", "'data/nse_")
        
        # In Camarilla: OUTPUT_DIR = Path("/Users/sumeetdas/Desktop") -> Path(".")
        text = text.replace('OUTPUT_DIR   = Path("/Users/sumeetdas/Desktop")', 'OUTPUT_DIR = Path("results")')
        text = text.replace('output_path = Path("~/Desktop/linearity scan").expanduser()', 'output_path = Path("results")')
        
        (scripts_dir / filename).write_text(text)

    for script in ["linearity_scanner.py", "export_strict_camarilla.py", "nse_daily_update_service.py"]:
        copy_and_fix(script)
        
    # Build send_email.py
    email_script = """import smtplib
import os
from email.message import EmailMessage
import mimetypes

def send_email():
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("EMAIL_APP_PASSWORD")
    receiver = os.environ.get("RECEIVER_EMAIL")

    if not sender or not password or not receiver:
        print("Email credentials not set. Skipping email.")
        return

    msg = EmailMessage()
    msg['Subject'] = 'NSE Daily Scanners Report'
    msg['From'] = sender
    msg['To'] = receiver
    msg.set_content("Attached are the latest automated runs for the Camarilla and Linearity Scanners.")

    files = ["results/Linearity Scans.xlsx", "results/Camarilla Scans/Camarilla Scans.xlsx"]
    for fpath in files:
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                file_data = f.read()
                file_name = os.path.basename(fpath)
            orig_name = file_name
            msg.add_attachment(file_data, maintype='application', subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=orig_name)
        else:
            print(f"File not found: {fpath}")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == '__main__':
    send_email()
"""
    (scripts_dir / "send_email.py").write_text(email_script)
    
    # Build daily_scan.yml
    yml_script = """name: Daily NSE Scanners

on:
  schedule:
    - cron: '13 0 * * 1-5' # 13:00 UTC (18:30 IST) Monday-Friday
  workflow_dispatch:

jobs:
  run-scanners:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          pip install duckdb pandas yfinance xlsxwriter requests

      - name: Download Historical Data Asset
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          mkdir -p data
          # Download the zip from the 'data-latest' tag release
          gh release download data-latest --pattern nse_master.zip --dir data || echo "Release not found, fetching might fail"
          cd data
          unzip -o nse_master.zip || true
          rm nse_master.zip || true

      - name: Update Parquet Data
        run: |
          python scripts/nse_daily_update_service.py

      - name: Make results directory
        run: mkdir -p results

      - name: Run Linearity Scanner
        run: |
          python scripts/linearity_scanner.py

      - name: Run Camarilla Scanner
        run: |
          python scripts/export_strict_camarilla.py

      - name: Send Email
        env:
          SENDER_EMAIL: ${{ secrets.SENDER_EMAIL }}
          EMAIL_APP_PASSWORD: ${{ secrets.EMAIL_APP_PASSWORD }}
          RECEIVER_EMAIL: ${{ secrets.RECEIVER_EMAIL }}
        run: |
          python scripts/send_email.py

      - name: Re-zip and Upload Updated Parquet
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cd data
          zip -qr0 nse_master.zip *.parquet
          # Overwrite the release asset
          gh release upload data-latest nse_master.zip --clobber
"""
    (gha_dir / "daily_scan.yml").write_text(yml_script)

    # Build README.md
    readme = """# NSE Scanners Automated pipeline

This executes the Linearity and Camarilla Scanners daily at 18:30 IST.
It utilizes GitHub Releases to host the 370MB+ parquet datastore to avoid git history bloat.

**Deployment Checklist:**
1. Commit all files to a new GitHub repo.
2. Create an empty Tag & Release called `data-latest`.
3. Compress the three parquet directories into `nse_master.zip` and upload as an asset to `data-latest`.
4. Configure Action Secrets: `SENDER_EMAIL`, `EMAIL_APP_PASSWORD`, `RECEIVER_EMAIL`.
"""
    (dest_dir / "README.md").write_text(readme)
    print(f"Setup complete at {dest_dir}")

if __name__ == '__main__':
    setup_repo()
