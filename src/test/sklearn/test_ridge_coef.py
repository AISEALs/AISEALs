from sklearn import linear_model


reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg.coef_)

print(reg.intercept_)

reg2 = linear_model.LinearRegression()
reg2.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg2.coef_)

print(reg2.intercept_)
