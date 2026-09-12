import json

filepath = r"c:\Users\Neha madhu shalini\OneDrive\Documents\Downloads\agentic-customer-support-ai-main\agentic-customer-support-ai-main\notebooks\agentic_customer_support.ipynb"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("Ashar Zeeshan", "Neha")
content = content.replace("asharzeeshan", "Neha")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
