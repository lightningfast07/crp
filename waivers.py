import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("monthly_service_charge_fees.csv")

# Display first few rows
print(df.head())

# Identify waiver columns
waiver_columns = ["Transaction Waiver", "Special Month Waiver", "Student Waiver"]
type_columns = ["Type of Transaction Waiver", "Type of Special Month Waiver", "Type of Student Waiver"]
# Filter rows where MSF was waived
waived_df = df[df["Msf Waived Status"] == "Yes"]

# Count how often each waiver was used
waiver_counts = waived_df[waiver_columns].apply(lambda x: (x == "Yes").sum()).sort_values(ascending=False)

# Display counts
print(waiver_counts)
# Dictionary to store waiver type counts
waiver_type_counts = {}

for waiver_col, type_col in zip(waiver_columns, type_columns):
    # Filter rows where the waiver was applied
    relevant_rows = waived_df[waived_df[waiver_col] == "Yes"]
    
    # Count occurrences of each type
    type_counts = relevant_rows[type_col].value_counts()
    
    # Store results
    waiver_type_counts[type_col] = type_counts

# Print waiver type breakdown
for key, value in waiver_type_counts.items():
    print(f"\n{key} Breakdown:\n{value}")
    plt.figure(figsize=(6, 6))
    plt.pie(value, labels=value.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(value)))
    plt.title(f"Distribution of {key}")
    plt.show()
plt.figure(figsize=(8, 6))
sns.barplot(x=waiver_counts.index, y=waiver_counts.values, palette="viridis")
plt.xticks(rotation=30, ha="right")
plt.xlabel("Waiver Type")
plt.ylabel("Number of Times Used")
plt.title("Most Common Waivers Used for MSF Waiver")
plt.show()
plt.figure(figsize=(8, 8))
plt.pie(waiver_counts, labels=waiver_counts.index, autopct='%1.1f%%', colors=sns.color_palette("viridis", len(waiver_counts)))
plt.title("Distribution of Waivers Used for MSF Waiver")
plt.show()
plt.figure(figsize=(12, 6))

# Combine all waiver type counts into a single DataFrame
waiver_type_df = pd.DataFrame(waiver_type_counts).fillna(0)

# Stacked bar chart
waiver_type_df.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 6))
plt.xlabel("Waiver Type Categories")
plt.ylabel("Number of Occurrences")
plt.title("Breakdown of Waiver Types Used")
plt.legend(title="Waiver Type")
plt.xticks(rotation=45, ha="right")
plt.show()
