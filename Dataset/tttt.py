import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Tạo dữ liệu giả lập
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(20))

# Huấn luyện hai thuật toán
lr = LinearRegression().fit(X, y)
dt = DecisionTreeRegressor(max_depth=2).fit(X, y)

# Lấy kết quả của hai thuật toán
y_lr = lr.predict(X)
y_dt = dt.predict(X)

# Vẽ biểu đồ scatter plot
plt.scatter(y_lr, y_dt)
plt.xlabel("Linear Regression")
plt.ylabel("Decision Tree")
plt.title("Comparison of Linear Regression and Decision Tree")
plt.show()
