import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats


file_path = './res_doe_1.csv'
df = pd.read_csv(file_path)

df['Factor_Img_Size_Value'] = df['Factor_Img_Size_Value'].astype('category')
df['Factor_Batch_Size_Value'] = df['Factor_Batch_Size_Value'].astype('category')

print("--- Descriptive Statistics ---")
desc_stats = df.groupby(['Factor_Img_Size_Value', 'Factor_Batch_Size_Value'])['Response_Val_Accuracy'] \
               .agg(['mean', 'std', 'count']) \
               .reset_index()
print(desc_stats)
print("\n")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.set_style("whitegrid")

# Plot A: Boxplot for Image Size
sns.boxplot(x='Factor_Img_Size_Value', y='Response_Val_Accuracy', data=df, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title('Impact of Image Size on Accuracy', fontsize=14)
axes[0, 0].set_xlabel('Image Size (px)')
axes[0, 0].set_ylabel('Validation Accuracy')

# Plot B: Boxplot for Batch Size
sns.boxplot(x='Factor_Batch_Size_Value', y='Response_Val_Accuracy', data=df, ax=axes[0, 1], palette="Set3")
axes[0, 1].set_title('Impact of Batch Size on Accuracy', fontsize=14)
axes[0, 1].set_xlabel('Batch Size')
axes[0, 1].set_ylabel('Validation Accuracy')

# Plot C: Interaction Plot
sns.pointplot(x='Factor_Img_Size_Value', y='Response_Val_Accuracy', hue='Factor_Batch_Size_Value', 
              data=df, ax=axes[1, 0], markers=["o", "s", "^"], capsize=.1)
axes[1, 0].set_title('Interaction Plot: Image Size x Batch Size', fontsize=14)
axes[1, 0].set_xlabel('Image Size (px)')
axes[1, 0].set_ylabel('Mean Validation Accuracy')
axes[1, 0].legend(title='Batch Size')


print("--- ANOVA Table ---")
model = ols('Response_Val_Accuracy ~ C(Factor_Img_Size_Value) * C(Factor_Batch_Size_Value)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
print("\n")

residuals = model.resid

# Plot D: QQ Plot
sm.qqplot(residuals, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('Normal Probability Plot (QQ Plot)', fontsize=14)

plt.tight_layout()
plt.savefig('doe_analysis_results.png', dpi=300)
print("Graphs saved to 'doe_analysis_results.png'")

# Statistical Tests
print("--- Model Validation Tests ---")
# Shapiro-Wilk Test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: W={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  -> Residuals are Normally Distributed (Assumption met)")
else:
    print("  -> Residuals are NOT Normally Distributed (Assumption failed)")

# Levene's Test
groups = [group['Response_Val_Accuracy'].values for name, group in df.groupby(['Factor_Img_Size_Value', 'Factor_Batch_Size_Value'])]
levene_stat, levene_p = stats.levene(*groups)
print(f"Levene Test: W={levene_stat:.4f}, p-value={levene_p:.4f}")
if levene_p > 0.05:
    print("  -> Variances are Homogeneous (Assumption met)")
else:
    print("  -> Variances are NOT Homogeneous (Assumption failed)")