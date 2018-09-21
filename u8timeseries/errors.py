def mean_absolute_scaled_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = np.sum(np.abs(y_true - y_pred))
    t = y_true.size
    scale = t/(t-1) * np.sum(np.abs(np.diff(y_true)))
    return errors / scale

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def overall_percentage_error(y_true, y_pred):
    y_true_sum, y_pred_sum = np.sum(np.array(y_true)), np.sum(np.array(y_pred))
    return np.abs((y_true_sum - y_pred_sum) / y_true_sum) * 100
