import numpy as np


def cal_miap(metric, general=False):

    predicts = metric['predicts_gzsl'] if general else metric['predicts_zsl']
    gts = metric['gts_gzsl'] if general else metric['gts_zsl']

    print(gts)
    aaa

    iAPs = []
    for predict, gt in zip(predicts, gts):
        gt = np.where(gt == 1)[0]

        idx = np.array([i for i, p in enumerate(np.argsort(-predict), 1) if p in gt])
        num_hits = np.cumsum([1 for p in np.argsort(-predict) if p in gt])
        scores = num_hits / idx

        iap = sum(scores) / (len(scores) + np.finfo(float).eps)
        iAPs.append(iap)
    miAP = np.mean(iAPs)

    return iAPs, miAP


def bool_arr(predicts, top_num):
    top = np.zeros(predicts.shape)
    x_ind = np.array([[i] * top_num for i in range(predicts.shape[0])]).reshape(-1)
    y_ind = np.argsort(-predicts, axis=1)[:, :top_num].reshape(-1)
    top[[x_ind, y_ind]] = 1
    return top


def cal_prf1(tops, gts):
    tp_per_class = np.logical_and(tops, gts).sum(axis=0)
    p_per_class = tops.sum(axis=0)
    g_per_class = gts.sum(axis=0)

    c_p = np.nan_to_num(tp_per_class / p_per_class).sum() / tp_per_class.shape[0]
    c_r = np.nan_to_num(tp_per_class / g_per_class).sum() / tp_per_class.shape[0]
    c_f1 = np.nan_to_num(2 * c_p * c_r / (c_p + c_r))

    o_p = np.nan_to_num(tp_per_class.sum() / p_per_class.sum())
    o_r = np.nan_to_num(tp_per_class.sum() / g_per_class.sum())
    o_f1 = np.nan_to_num(2 * o_p * o_r / (o_p + o_r))

    return {'c_p': c_p, 'c_r': c_r, 'c_f1': c_f1, 'o_p': o_p, 'o_r': o_r, 'o_f1': o_f1}


def cal_top(metric, general=False):
    predicts = metric['predicts_gzsl'] if general else metric['predicts_zsl']
    gts = metric['gts_gzsl'] if general else metric['gts_zsl']

    top3_prf1 = cal_prf1(bool_arr(predicts, 3), gts)
    top5_prf1 = cal_prf1(bool_arr(predicts, 10), gts)

    return top3_prf1, top5_prf1


def write_table(prf1_dict):
    text = '| Tables            |  CP  |  CR  |  CF  |  OP  |  OR  |  OF  |\n\
                | :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |\n'

    for topk, prf1 in prf1_dict.items():
        text += '| ' + str(topk) + ' | ' + \
            '{:.3f}'.format(prf1['c_p']) + ' | ' + '{:.3f}'.format(prf1['c_r']) + ' | ' + \
            '{:.3f}'.format(prf1['c_f1']) + ' | ' + '{:.3f}'.format(prf1['o_p']) + ' | ' + \
            '{:.3f}'.format(prf1['o_r']) + ' | ' + '{:.3f}'.format(prf1['o_f1']) + ' |\n'

    return text
