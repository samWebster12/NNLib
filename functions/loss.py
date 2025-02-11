def least_squares(y_pred, y_true):
    loss = 0
    for i in range(len(y_pred)):
        loss += (y_true[i] - y_pred[i]) ** 2
    
    return loss

def least_squares_deriv(y_pred, y_true):
    loss_der = 0
    for i in range(len(y_pred)):
        loss_der += -2 * (y_true[i] - y_pred[i])
    
    return loss_der