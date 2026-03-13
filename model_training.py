from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model

model = RandomForestClassifier(random_state=42)

# Train model on training data

model.fit(X_train, y_train)

# Calculate feature importance from Random Forest model

importance = model.feature_importances_

# Create dataframe for feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

# Sort features by importance
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Display top 10 most important features
print("Top Features Influencing Loan Default:")
print(feature_importance.head(10))

# Plot top 10 important features

plt.figure(figsize=(8,5))

sns.barplot(
    x=feature_importance["Importance"][:10],
    y=feature_importance["Feature"][:10]
)

plt.title("Top 10 Important Features for Loan Default Prediction")

plt.show()

# Save trained model

pickle.dump(model, open("trained_model.pkl", "wb"))

# Predict class labels (0 or 1)

y_pred = model.predict(X_test)

# Predict probability of default

y_prob = model.predict_proba(X_test)[:,1]