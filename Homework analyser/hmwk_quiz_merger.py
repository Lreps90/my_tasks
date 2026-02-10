# import os
# import pandas as pd
#
# # Define the directory containing the Excel files.
# directory = r"C:\Users\Lreps\OneDrive - Heathside School\Computer Science\GCSE\2024-2025\Year 10 hmwk data"
#
# # Dictionary to accumulate scores for each student.
# # Structure: { student: { task_title: percentage, ... }, ... }
# student_scores = {}
#
# # Iterate through each file in the directory.
# for filename in os.listdir(directory):
#     # Process only Excel files (both .xlsx and .xls)
#     if filename.endswith(('.xlsx', '.xls')):
#         file_path = os.path.join(directory, filename)
#         try:
#             # Read the Excel file with headers on row 2 (i.e. header index=1)
#             df = pd.read_excel(file_path, header=1)
#
#             # Construct the task title using the first row's values from 'Assignments' and 'Due Date'
#             if not df.empty and 'Assignments' in df.columns and 'Due Date' in df.columns:
#                 assignment_text = str(df['Assignments'].iloc[0]).strip()
#                 due_date_text = str(df['Due Date'].iloc[0]).strip()
#                 task_title = f"{assignment_text}: {due_date_text}".strip()
#             else:
#                 # Fallback to using the filename if required columns are missing.
#                 task_title = os.path.splitext(filename)[0]
#
#             # Iterate through each row in the DataFrame.
#             for _, row in df.iterrows():
#                 student = str(row.get('Full Name', '')).strip()
#                 status = str(row.get('Status', '')).strip().lower()
#                 percentage = row.get('Percent', None)
#
#                 # Skip rows without a student name.
#                 if not student:
#                     continue
#
#                 # Initialise dictionary for the student if first encountered.
#                 if student not in student_scores:
#                     student_scores[student] = {}
#
#                 # If the homework was turned in, record the percentage (defaulting to 0 if missing).
#                 # Otherwise, assign a score of 0.
#                 if status == "turned in":
#                     student_scores[student][task_title] = percentage if pd.notnull(percentage) else 0
#                 else:
#                     student_scores[student][task_title] = 0
#
#         except Exception as e:
#             print(f"Error processing file {filename}: {e}")
#
# # Create a summary DataFrame from the accumulated scores.
# # Each row represents a student and each column represents a homework task.
# summary_df = pd.DataFrame.from_dict(student_scores, orient='index')
# summary_df.index.name = "Student"
# summary_df.reset_index(inplace=True)
#
# # Format percentage columns: multiply by 100, round to the nearest whole number, and add a '%' symbol.
# # This applies to every column except the "Student" column.
# for col in summary_df.columns:
#     if col != "Student":
#         summary_df[col] = summary_df[col].apply(lambda x: f"{round(x * 100)}%" if pd.notnull(x) else "")
#
# # Define the path for the summary CSV file.
# summary_csv_path = os.path.join(directory, "homework_summary.csv")
#
# # Write the summary DataFrame to CSV.
# summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
#
# print(f"Summary written to {summary_csv_path}")

import os
import pandas as pd

# Define the directory containing the Excel files.
directory = r"C:\Users\Lreps\OneDrive - Heathside School\Computer Science\GCSE\2024-2025\Year 10 hmwk data"

# Dictionary to accumulate scores for each student.
# Structure: { student: { task_title: percentage, ... }, ... }
student_scores = {}

# Iterate through each file in the directory.
for filename in os.listdir(directory):
    # Process only Excel files (both .xlsx and .xls)
    if filename.endswith(('.xlsx', '.xls')):
        file_path = os.path.join(directory, filename)
        try:
            # Read the Excel file with headers on row 2 (i.e. header index=1)
            df = pd.read_excel(file_path, header=1)

            # Construct the task title using the first row's values from 'Assignments' and 'Due Date'
            if not df.empty and 'Assignments' in df.columns and 'Due Date' in df.columns:
                assignment_text = str(df['Assignments'].iloc[0]).strip()
                due_date_text = str(df['Due Date'].iloc[0]).strip()
                task_title = f"{assignment_text}: {due_date_text}".strip()
            else:
                # Fallback to using the filename if required columns are missing.
                task_title = os.path.splitext(filename)[0]

            # Iterate through each row in the DataFrame.
            for _, row in df.iterrows():
                student = str(row.get('Full Name', '')).strip()
                status = str(row.get('Status', '')).strip().lower()
                percentage = row.get('Percent', None)

                # Skip rows without a student name.
                if not student:
                    continue

                # Initialise dictionary for the student if first encountered.
                if student not in student_scores:
                    student_scores[student] = {}

                # If the homework was turned in, record the percentage (defaulting to 0 if missing).
                # Otherwise, assign a score of 0.
                if status == "turned in":
                    student_scores[student][task_title] = percentage if pd.notnull(percentage) else 0
                else:
                    student_scores[student][task_title] = 0

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Create a summary DataFrame from the accumulated scores.
# Each row represents a student and each column represents a homework task.
summary_df = pd.DataFrame.from_dict(student_scores, orient='index')
summary_df.index.name = "Student"
summary_df.reset_index(inplace=True)

# Calculate overall homework completion percentage (i.e. percentage of tasks turned in).
# For each student, count tasks with a non-zero numeric value and divide by the total number of tasks.
# Note: This computation is done on the numeric data before formatting.
task_columns = [col for col in summary_df.columns if col not in ["Student"]]

def compute_completion(row):
    # Ensure missing values are treated as 0.
    tasks = row[task_columns].fillna(0)
    total_tasks = len(task_columns)
    # A task is considered completed on time if its numeric value is not 0.
    completed_count = (tasks != 0).sum()
    return f"{round((completed_count / total_tasks) * 100)}%" if total_tasks > 0 else "0%"

summary_df["Homework Completed On Time"] = summary_df.apply(compute_completion, axis=1)

# Format individual task percentage columns:
# Multiply each numeric value by 100, round to the nearest whole number, and append a '%' symbol.
# Skip the "Student" column and the newly added completion column.
for col in summary_df.columns:
    if col not in ["Student", "Homework Completed On Time"]:
        summary_df[col] = summary_df[col].apply(lambda x: f"{round(x * 100)}%" if pd.notnull(x) else "")

# Define the path for the summary CSV file.
summary_csv_path = os.path.join(directory, "homework_summary.csv")

# Write the summary DataFrame to CSV.
summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')

print(f"Summary written to {summary_csv_path}")
