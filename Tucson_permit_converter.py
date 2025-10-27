from PyPDF2 import PdfReader
import pandas as pd
import re
import os

filepath = "C:/Users/Owner/Documents/MADS/Capstone/Data/Tucson_permits/Residential"

dataframes = []

for file in os.listdir(filepath):
    full_path = os.path.join(filepath, file)
    reader = PdfReader(full_path )
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    permits = re.split(r"Residential Building\s*-\s*One or Two Family", text)
    permits = [p.strip() for p in permits if p.strip()]

    data = []

    for block in permits:
        permit_num = re.search(r"TC-[A-Z]{3}-\d{4}-\d+", block)
        workclass = re.search(r"Residential\s+([A-Za-z/\s]+?)\s+Permit", block)
        issue_date = re.search(r"\d{2}/\d{2}/\d{4}", block)
        address = re.search(r"\d+ [A-Z0-9\s\.]+, TUCSON, AZ \d{5}", block)
        desc = re.search(r"Description:\s*(.+)", block)
        contractor = re.search(r"Contractor:\s*(.+)", block)

        # Skip if Workclass is not 'Solar'
        if not workclass or workclass.group(1).strip() != "Solar":
            continue

        data.append({
            "Permit #": permit_num.group(0) if permit_num else None,
            "Workclass": workclass.group(1).strip() if workclass else None,
            "Issue Date": issue_date.group(0) if issue_date else None,
            "Address": address.group(0) if address else None,
            "Description": desc.group(1).strip() if desc else None,
            "Contractor": contractor.group(1).strip() if contractor else None
        })

    if data:
        df = pd.DataFrame(data)
        dataframes.append(df)

if dataframes:
    clean_permits = pd.concat(dataframes, ignore_index=True)
    clean_permits.to_csv("C:/Users/Owner/Documents/MADS/Capstone/Data/Tucson_solar_permits.csv", index=False)
    print(f"Saved {len(clean_permits)} solar permits.")
else:
    print("No solar permits found.")      