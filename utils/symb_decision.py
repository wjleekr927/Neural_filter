# If correct, return 1 / else return 0
def decision_16QAM(pred_symb, target_symb, norm_cof, filter_type = None):
    pred_label = []

    if pred_symb[0] >= 0:
        if pred_symb[0] >= 2 * norm_cof:
            pred_label.append(3*norm_cof)
        else:
            pred_label.append(norm_cof)
    else:
        if pred_symb[0] <= -2 * norm_cof:
            pred_label.append(-3*norm_cof)
        else:
            pred_label.append(-norm_cof)

    if pred_symb[1] >= 0:
        if pred_symb[1] >= 2 * norm_cof:
            pred_label.append(3*norm_cof)
        else:
            pred_label.append(norm_cof)
    else:
        if pred_symb[1] <= -2 * norm_cof:
            pred_label.append(-3*norm_cof)
        else:
            pred_label.append(-norm_cof)
    
    if filter_type == 'NN': # To convert target_symb from tensor
        if round((abs(pred_label[0] - target_symb[0].item()) + abs(pred_label[1] - target_symb[1].item())), 4) < 0.0001:
            return 1
        else:
            return 0
    else:
        if round((abs(pred_label[0] - target_symb[0]) + abs(pred_label[1] - target_symb[1])), 4) < 0.0001:
            return 1
        else:
            return 0