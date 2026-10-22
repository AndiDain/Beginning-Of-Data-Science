from numpy.ma.extras import average
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data/student_dropout_risk_2026.csv')

df.head()
df.isna().sum()
df.duplicated().sum()

# df["gender"].value_counts()
# df["family_income_level"].value_counts()
# df["semester"].value_counts()

print("=" * 20)
test = df[["age", "attendance_rate", "sleep_hours", "stress_level", "satisfaction_score"]].describe()
print(test)

print("=" * 20)
dropout_counts = df["dropout"].value_counts()
dropout_percentage = df["dropout"].value_counts(normalize=True) * 100
print(dropout_counts)
print(dropout_percentage)

# print("=" * 20)
# dropout_percentage.plot(
#     kind="bar",
#     color=["steelblue", "orange"]
# )
# 
# plt.title("Student Dropout Distribution")
# plt.xlabel("Dropout (0 = No, 1 = Yes)")
# plt.ylabel("Percentage (%)")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

print("=" * 20)
dropout_stats = df.groupby("dropout")[
    [
        "age",
        "attendance_rate",
        "assignments_submitted_rate",
        "gpa_semester_1",
        "gpa_semester_2",
        "units_passed",
        "units_failed",
        "missed_deadlines",
        "weekly_work_hours",
        "sleep_hours",
        "stress_level"
    ]
].mean().round(2)

print(dropout_stats)

print("=" * 20)
numeric_df = df.select_dtypes(include=np.number)
correlations = numeric_df.corr()["dropout"].drop("dropout")
correlations = correlations.sort_values()
print(correlations)

study_mode_dropout = df.groupby("study_mode")["dropout"].mean().mul(100).sort_values(ascending=False)
course_type_dropout = df.groupby("course_type")["dropout"].mean().mul(100).round(2)
family_income_level_dropout = df.groupby("family_income_level")["dropout"].mean().mul(100).round(2)
gender_dropout = df.groupby("gender")["dropout"].mean().mul(100).round(2)

print(study_mode_dropout.round(2))
print(course_type_dropout.round(2))
print(family_income_level_dropout.round(2))
print(gender_dropout.round(2))

gpa_change = df["gpa_semester_2"].to_numpy() - df["gpa_semester_1"].to_numpy()
print("Average GPA change: ", np.mean(gpa_change).round(3))
print("Median GPA change: ", np.median(gpa_change).round(3))

df["gpa_change"] = (
    df["gpa_semester_2"] - df["gpa_semester_1"]
)

print(
    df.groupby("dropout")["gpa_change"]
      .agg(["mean", "median"])
      .round(3)
)

# df.boxplot(
#     column="attendance_rate",
#     by="dropout"
# )
# 
# plt.title("Attendance Rate by Dropout Status")
# plt.suptitle("")
# plt.xlabel("Dropout (0 = No, 1 = Yes)")
# plt.ylabel("Attendance Rate (%)")
# plt.tight_layout()
# plt.show()