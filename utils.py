import numpy as np


def cal_h_acc(record, general=False):
    tr = record['test_seen_g']['acc'] if general else record['test_seen']['acc']
    ts = record['test_unseen_g']['acc'] if general else record['test_unseen']['acc']

    return 2 * tr * ts / (tr + ts)


def cal_acc(metric, general=False):

    labels = metric['correct_g'] if general else metric['correct']
    totals = metric['total_g'] if general else metric['total']

    labels = np.asarray(labels)
    totals = np.asarray(totals)

    tp = np.logical_and(labels, totals).sum(axis=0).reshape(-1)
    # fp = np.logical_not(np.logical_or(labels, totals)).sum(axis=0).reshape(-1)

    totals = totals.sum(axis=0).reshape(-1)

    classes = [np.nan_to_num(l / t) for l, t in zip(tp, totals)]
    acc = sum(classes) / len(np.asarray(metric['total']).sum(axis=0).reshape(-1))

    return classes, acc


def cal_miap(metric, general=False):

    predicts = metric['predicts_gzsl'] if general else metric['predicts_zsl']
    gts = metric['gts_gzsl'] if general else metric['gts_zsl']

    iAPs = []
    for predict, gt in zip(np.asarray(predicts), np.asarray(gts)):

        predict = predict.reshape(-1)
        gt = gt.reshape(-1)

        gt = np.where(gt == 1)[0]
        predict_list = np.argsort(-predict)

        idx = np.array(sorted([np.nonzero(predict_list == g)[0][0] + 1 for g in gt]))
        num_hits = np.cumsum([1 for p in np.argsort(-predict) if p in gt])
        scores = num_hits / idx

        iap = sum(scores) / (len(scores) + np.finfo(float).eps)
        iAPs.append(iap)

    miAP = np.mean(iAPs)

    return miAP


def cal_map(metric, general=False):

    predicts = metric['predicts_gzsl'] if general else metric['predicts_zsl']
    gts = metric['gts_gzsl'] if general else metric['gts_zsl']
    predicts = np.squeeze(np.asarray(predicts))
    gts = np.squeeze(np.asarray(gts))

    mAPs = []
    for col in range(gts.shape[1]):
        predict = predicts[:, col]
        ground_truth = gts[:, col]

        tp = np.zeros(len(ground_truth))
        fp = np.zeros(len(ground_truth))

        tp[ground_truth[np.argsort(-predict)] == 1] = 1
        fp[ground_truth[np.argsort(-predict)] == 0] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / (sum(ground_truth == 1) + np.finfo(float).eps)
        precision = tp / (fp + tp + np.finfo(float).eps)

        ap = sum([0 if len(precision[recall >= (t * 0.1)]) == 0 else max(precision[recall >= (t * 0.1)]) for t in range(11)]) / 11
        mAPs.append(ap)

    return mAPs


def cal_prf1(metric, general=False):

    predicts = metric['predicts_gzsl'] if general else metric['predicts_zsl']
    gts = metric['gts_gzsl'] if general else metric['gts_zsl']

    tops = np.asarray(predicts)
    gts = np.asarray(gts)

    tp_per_class = np.logical_and(tops, gts).sum(axis=0)
    p_per_class = tops.sum(axis=0)
    g_per_class = gts.sum(axis=0)

    c_p = np.nan_to_num(tp_per_class / p_per_class).sum() / tp_per_class.shape[0]
    c_r = np.nan_to_num(tp_per_class / (g_per_class)).sum() / tp_per_class.shape[0]
    c_f1 = np.nan_to_num(2 * c_p * c_r / (c_p + c_r))

    o_p = np.nan_to_num(tp_per_class.sum() / p_per_class.sum())
    o_r = np.nan_to_num(tp_per_class.sum() / g_per_class.sum())
    o_f1 = np.nan_to_num(2 * o_p * o_r / (o_p + o_r))

    return {'c_p': c_p, 'c_r': c_r, 'c_f1': c_f1, 'o_p': o_p, 'o_r': o_r, 'o_f1': o_f1}


def write_table(prf1_dict):
    text = '| Tables            |  CP  |  CR  |  CF  |  OP  |  OR  |  OF  |\n\
                | :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |\n'

    for topk, prf1 in prf1_dict.items():
        text += '| ' + str(topk) + ' | ' + \
            '{:.2f}'.format(prf1['c_p'] * 100) + ' | ' + '{:.2f}'.format(prf1['c_r'] * 100) + ' | ' + \
            '{:.2f}'.format(prf1['c_f1'] * 100) + ' | ' + '{:.2f}'.format(prf1['o_p'] * 100) + ' | ' + \
            '{:.2f}'.format(prf1['o_r'] * 100) + ' | ' + '{:.2f}'.format(prf1['o_f1'] * 100) + ' |\n'

    return text
