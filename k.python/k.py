
# 1. CREATE SAMPLE BUSINESS DATASET

#np.random.seed(42)

n = 300

platforms = ["Instagram", "YouTube", "Google Ads"]
segments = ["Students", "Young Adults", "Professionals"]

df = pd.DataFrame({
    "Advertising_Spend": np.random.randint(500, 10000, n),
    "Target_Segment": np.random.choice(segments, n),
    "Platform": np.random.choice(platforms, n),
    "Impressions": np.random.randint(1000, 50000, n),
    "Clicks": np.random.randint(100, 3000, n)
})

# Create sample conversions
df["Conversions"] = (
    df["Clicks"] * np.random.uniform(0.03, 0.15, n)
).astype(int)

# Create sample sales revenue
platform_effect = df["Platform"].map({
    "Instagram": 1.0,
    "YouTube": 1.15,
    "Google Ads": 1.30
})

segment_effect = df["Target_Segment"].map({
    "Students": 0.85,
    "Young Adults": 1.05,
    "Professionals": 1.25
})

df["Sales"] = (
    500
    + df["Advertising_Spend"] * 2.0
    + df["Conversions"] * 80
) * platform_effect * segment_effect

df["Sales"] += np.random.normal(0, 1000, n)
df["Sales"] = df["Sales"].clip(lower=0).round(2)

# Save original dataset
df.to_csv("sales_data.csv", index=False)

print("\nDATASET CREATED SUCCESSFULLY")
print(df.head())


# ----------------
# 2. DATA CLEANING
# ----------------

print("\n--- DATA CLEANING ---")

print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Convert numeric columns safely
numeric_cols = [
    "Advertising_Spend",
    "Impressions",
    "Clicks",
    "Conversions",
    "Sales"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid or missing data
df = df.dropna(
    subset=numeric_cols + ["Target_Segment", "Platform"]
).copy()

# Remove invalid negative values
for col in numeric_cols:
    df = df[df[col] >= 0]

# Remove rows where impressions are zero
df = df[df["Impressions"] > 0].copy()

# Keep clicks within impressions
df = df[df["Clicks"] <= df["Impressions"]].copy()

# Keep conversions within clicks
df = df[df["Conversions"] <= df["Clicks"]].copy()

print("\nCleaned dataset shape:", df.shape)
print(df.head())

 ----------------------------------------
# 3. FEATURE ENGINEERING ----------------------------------------

print("\n--- FEATURE ENGINEERING ---")

# Click-through rate in percentage
df["CTR"] = np.where(
    df["Impressions"] > 0,
    df["Clicks"] / df["Impressions"] * 100,
    0
)

# Conversion rate in percentage
df["Conversion_Rate"] = np.where(
    df["Clicks"] > 0,
    df["Conversions"] / df["Clicks"] * 100,
    0
)

# Cost per click
df["Cost_Per_Click"] = np.where(
    df["Clicks"] > 0,
    df["Advertising_Spend"] / df["Clicks"],
    0
)

print(df[
    ["CTR", "Conversion_Rate", "Cost_Per_Click"]
].head())


# ----------------------------------------
# 4. DATA TRANSFORMATION
# ----------------------------------------

# Features and target variable
# Do not use Sales or values derived from Sales as inputs.

features = [
    "Advertising_Spend",
    "Target_Segment",
    "Platform",
    "Impressions",
    "Clicks",
    "Conversions",
    "CTR",
    "Conversion_Rate",
    "Cost_Per_Click"
]

X = df[features]
y = df["Sales"]

numeric_features = [
    "Advertising_Spend",
    "Impressions",
    "Clicks",
    "Conversions",
    "CTR",
    "Conversion_Rate",
    "Cost_Per_Click"
]

categorical_features = [
    "Target_Segment",
    "Platform"
]

# Split before fitting transformations
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Encode text categories and scale numerical data
preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])


# ----------------------------------------
# 5. BUILD AND TRAIN MODEL
# ----------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            random_state=42
        )
    )
])

print("\n--- TRAINING MODEL ---")

model.fit(X_train, y_train)

print("Model training completed!")


# ----------------------------------------
# 6. SALES PREDICTION
# -----------------------------------------

y_pred = model.predict(X_test)

print("\n--- MODEL EVALUATION ---")

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Actual vs predicted sales
results = pd.DataFrame({
    "Actual_Sales": y_test.values,
    "Predicted_Sales": y_pred.round(2)
})

print("\nPrediction results:")
print(results.head(10))

results.to_csv("prediction_results.csv", index=False)


# -----------------------------------------------------
# 7. PLATFORM-WISE MARKETING ANALYSIS
# -----------------------------------------------------

print("\n--- PLATFORM-WISE ANALYSIS ---")

platform_analysis = df.groupby("Platform").agg(
    Total_Sales=("Sales", "sum"),
    Total_Advertising_Spend=("Advertising_Spend", "sum"),
    Total_Clicks=("Clicks", "sum"),
    Total_Conversions=("Conversions", "sum")
)

# Revenue divided by advertising spend
platform_analysis["ROAS"] = (
    platform_analysis["Total_Sales"]
    / platform_analysis["Total_Advertising_Spend"].replace(0, np.nan)
)

platform_analysis["Conversion_Rate"] = (
    platform_analysis["Total_Conversions"]
    / platform_analysis["Total_Clicks"].replace(0, np.nan)
    * 100
)

print(platform_analysis.round(2))

platform_analysis.to_csv("platform_analysis.csv")


# -----------------------------------------------------
# 8. TARGET SEGMENT ANALYSIS
# -----------------------------------------------------

print("\n--- CUSTOMER SEGMENT ANALYSIS ---")

segment_analysis = df.groupby("Target_Segment").agg(
    Total_Sales=("Sales", "sum"),
    Average_Sales=("Sales", "mean"),
    Advertising_Spend=("Advertising_Spend", "sum"),
    Total_Conversions=("Conversions", "sum")
)

segment_analysis["ROAS"] = (
    segment_analysis["Total_Sales"]
    / segment_analysis["Advertising_Spend"].replace(0, np.nan)
)

print(segment_analysis.round(2))


# -----------------------------------------------------
# 9. GRAPH 1: ACTUAL VS PREDICTED SALES
# -----------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot(
    [low, high],
    [low, high],
    linestyle="--"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.savefig("sales_prediction_graph.png")
plt.show()


# -----------------------------------------------------
# 10. GRAPH 2: PLATFORM-WISE SALES
# -----------------------------------------------------

platform_analysis["Total_Sales"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Platform-wise Total Sales")
plt.xlabel("Advertising Platform")
plt.ylabel("Total Sales Revenue")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("platform_sales_graph.png")
plt.show()


# -----------------------------------------------------
# 11. GRAPH 3: ADVERTISING SPEND VS SALES
# -----------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Advertising_Spend"],
    df["Sales"]
)

plt.xlabel("Advertising Spend")
plt.ylabel("Sales Revenue")
plt.title("Advertising Spend vs Sales")

plt.tight_layout()
plt.savefig("advertising_vs_sales.png")
plt.show()


# -----------------------------------------------------
# 12. USER INPUT FOR NEW SALES PREDICTION
# -----------------------------------------------------

print("\n--- PREDICT SALES FOR A NEW CAMPAIGN ---")

spend = float(input("Enter advertising spend: "))
segment = input(
    "Target segment (Students/Young Adults/Professionals): "
).strip()
platform = input(
    "Platform (Instagram/YouTube/Google Ads): "
).strip()

impressions = int(input("Enter expected impressions: "))
clicks = int(input("Enter expected clicks: "))
conversions = int(input("Enter expected conversions: "))

# Validate input
if (
    spend < 0
    or impressions <= 0
    or clicks < 0
    or clicks > impressions
    or conversions < 0
    or conversions > clicks
):
    raise ValueError("Please enter valid campaign data.")

if segment not in segments:
    raise ValueError("Invalid target segment.")

if platform not in platforms:
    raise ValueError("Invalid advertising platform.")

# Create features for new campaign
ctr = clicks / impressions * 100

conversion_rate = (
    conversions / clicks * 100 if clicks > 0 else 0
)

cost_per_click = spend / clicks if clicks > 0 else 0

new_campaign = pd.DataFrame([{
    "Advertising_Spend": spend,
    "Target_Segment": segment,
    "Platform": platform,
    "Impressions": impressions,
    "Clicks": clicks,
    "Conversions": conversions,
    "CTR": ctr,
    "Conversion_Rate": conversion_rate,
    "Cost_Per_Click": cost_per_click
}])

predicted_sales = model.predict(new_campaign)[0]

print("\n==============================")
print("PREDICTED SALES REVENUE")
print("==============================")
print("Predicted sales:", round(predicted_sales, 2))

print("\nProject completed successfully!")