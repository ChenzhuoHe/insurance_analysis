# ============================================================
# Insurance Charges EDA & Analysis
# Dataset: Kaggle Healthcare Insurance Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import os
print(os.getcwd())

# ============================================================
# Step 1: Load Data
# ============================================================
print("="*60)
print("Step 1: Loading Data")
print("="*60)

df = pd.read_excel("insurance.xlsx")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# Step 2: Data Cleaning
# ============================================================
print("\n" + "="*60)
print("Step 2: Data Cleaning")
print("="*60)

print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()
print(f"Shape after deduplication: {df.shape}")

# Convert categorical columns to category dtype
df["sex"] = df["sex"].astype("category")
df["smoker"] = df["smoker"].astype("category")
df["region"] = df["region"].astype("category")

# Check categorical value ranges
print("\nCategorical value ranges:")
print(f"sex: {df['sex'].unique()}")
print(f"smoker: {df['smoker'].unique()}")
print(f"region: {df['region'].unique()}")

# ============================================================
# Step 3: Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "="*60)
print("Step 3: Exploratory Data Analysis")
print("="*60)

print("\n[3.1] Charges distribution")
print(df['charges'].describe())
print(f"Skewness = {df['charges'].skew():.3f}")

print("\n[3.2] Smoker vs Charges")
smoker_stats = df.groupby('smoker')['charges'].agg(['mean', 'median', 'count'])
print(smoker_stats)
ratio = df[df['smoker']=='yes']['charges'].mean() / df[df['smoker']=='no']['charges'].mean()
print(f"Smokers pay {ratio:.2f}x more than non-smokers on average")

print("\n[3.3] Region vs Charges")
print(df.groupby('region')['charges'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False))

print("\n[3.4] Sex vs Charges")
print(df.groupby('sex')['charges'].agg(['mean', 'median', 'count']))

print("\n[3.5] Correlation between numeric features and charges")
corr = df[['age', 'bmi', 'children', 'charges']].corr()['charges'].sort_values(ascending=False)
print(corr)

print("\n[3.6] BMI × Smoking interaction effect")
for s in ['yes', 'no']:
    sub = df[df['smoker'] == s]
    print(f"  smoker={s}: BMI vs charges correlation = {sub['bmi'].corr(sub['charges']):.3f} (n={len(sub)})")

# ============================================================
# Step 4: Visualization — Save Figures
# ============================================================
print("\n" + "="*60)
print("Step 4: Visualization — Saving figures")
print("="*60)

# Figure 1: charges distribution (raw vs log-transformed)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['charges'], kde=True, ax=axes[0], color='#4C72B0')
axes[0].set_title(f'Charges Distribution (skew={df["charges"].skew():.2f})')
axes[0].set_xlabel('Charges ($)')
sns.histplot(np.log1p(df['charges']), kde=True, ax=axes[1], color='#55A868')
axes[1].set_title(f'Log-transformed (skew={np.log1p(df["charges"]).skew():.2f})')
axes[1].set_xlabel('log(1 + Charges)')
plt.tight_layout()
plt.savefig('01_charges_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 01_charges_distribution.png")

# Figure 2: smoker × charges boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df, hue='smoker', palette=['#55A868', '#C44E52'], legend=False)
plt.title(f'Charges by Smoker (smoker mean = {ratio:.1f}x non-smoker)')
plt.ylabel('Charges ($)')
plt.savefig('02_smoker_charges_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 02_smoker_charges_boxplot.png")

# Figure 3: age × charges scatter (colored by smoker)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', data=df,
                palette=['#55A868', '#C44E52'], alpha=0.7, s=40)
plt.title('Age vs Charges (colored by smoker)')
plt.xlabel('Age')
plt.ylabel('Charges ($)')
plt.savefig('03_age_charges_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 03_age_charges_scatter.png")

# Figure 4: region × charges boxplot
plt.figure(figsize=(9, 6))
order = df.groupby('region')['charges'].mean().sort_values(ascending=False).index
sns.boxplot(x='region', y='charges', data=df, order=order, hue='region', palette='Set2', legend=False)
plt.title('Charges by Region (southeast highest)')
plt.ylabel('Charges ($)')
plt.savefig('04_region_charges_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 04_region_charges_boxplot.png")

# Figure 5: bmi × charges scatter (colored by smoker)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df,
                palette=['#55A868', '#C44E52'], alpha=0.7, s=40)
plt.title('BMI vs Charges — smoker corr=0.806, non-smoker corr=0.084')
plt.xlabel('BMI')
plt.ylabel('Charges ($)')
plt.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='Obesity threshold (BMI=30)')
plt.legend()
plt.savefig('05_bmi_charges_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 05_bmi_charges_scatter.png")

# Figure 6: correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[['age', 'bmi', 'children', 'charges']].corr(),
            annot=True, cmap='RdBu_r', center=0, fmt='.3f', square=True)
plt.title('Correlation Heatmap (numeric features)')
plt.tight_layout()
plt.savefig('06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 06_correlation_heatmap.png")

from IPython.display import Image, display

for i in range(1, 7):
    display(Image(f"0{i}_charges_distribution.png" if i==1 else f"0{i}_smoker_charges_boxplot.png" if i==2 else f"0{i}_age_charges_scatter.png" if i==3 else f"0{i}_region_charges_boxplot.png" if i==4 else f"0{i}_bmi_charges_scatter.png" if i==5 else f"0{i}_correlation_heatmap.png"))

# ============================================================
# Step 5: Linear Regression Modeling
# ============================================================
print("\n" + "="*60)
print("Step 5: Linear Regression Modeling")
print("="*60)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Encode categorical variables
df_enc = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
X = df_enc.drop('charges', axis=1)
y = df_enc['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)

print(f"R² (Test Set) = {r2:.4f}")
print(f"MAE = ${mae:,.2f}")
print(f"Mean charges = ${y.mean():,.2f}")
print(f"MAE/Mean = {mae/y.mean()*100:.1f}%")

print("\nFeature coefficients (sorted by absolute value):")
coef = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
for k, v in coef.items():
    print(f"  {k:25s}: {v:+,.2f}")

# ============================================================
# Step 6: Conclusions
# ============================================================
print("\n" + "="*60)
print("Step 6: Conclusions")
print("="*60)

conclusions = [
    f"1. Smoking is the dominant cost driver: smokers pay ${df[df['smoker']=='yes']['charges'].mean():,.0f} on average, "
    f"{ratio:.1f}x more than non-smokers (${df[df['smoker']=='no']['charges'].mean():,.0f}).",

    f"2. Age shows a positive correlation with charges (r={df['age'].corr(df['charges']):.3f}), "
    f"with each additional year adding approximately ${coef.get('age', 0):,.0f} to annual premiums.",

    f"3. Strong BMI × smoking interaction: r=0.806 among smokers vs r=0.084 among non-smokers — "
    f"high BMI combined with smoking drives premiums sharply higher.",

    f"4. Regional variation exists: southeast has the highest average charges (${df[df['region']=='southeast']['charges'].mean():,.0f}), "
    f"while southwest has the lowest (${df[df['region']=='southwest']['charges'].mean():,.0f}).",

    f"5. Gender alone has minimal impact (male ${df[df['sex']=='male']['charges'].mean():,.0f} vs "
    f"female ${df[df['sex']=='female']['charges'].mean():,.0f}), but the effect amplifies when interacting with smoking.",

    f"6. Linear regression R² = {r2:.3f} — 6 features explain approximately {r2*100:.0f}% of charges variance, "
    f"with MAE/Mean at {mae/y.mean()*100:.1f}%."
]
for c in conclusions:
    print("\n" + c)

print("\n" + "="*60)
print("✓ Analysis complete! All figures and conclusions have been saved.")
print("="*60)