
def cal_acc(metric, general=False):

    labels = metric['correct_g'] if general else metric['correct']
    classes = {c: labels[c] / (metric['total'][c] + 1e-10) for c in metric['total']}
    acc = sum(list(classes.values())) / len(metric['total'])

    return classes, acc
