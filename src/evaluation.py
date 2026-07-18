from sklearn.metrics import mean_squared_error, r2_score


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {"r2": round(r2, 3), "rmse": round(rmse, 4)}


def print_metrics(name, metrics):
    print(f"{name}: R2 = {metrics['r2']}, RMSE = {metrics['rmse']}")