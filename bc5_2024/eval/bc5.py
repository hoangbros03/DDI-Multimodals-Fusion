def __prf_bc5(tp, predict, golden):
    if tp == 0:
        return 0.0, 0.0, 0.0
    else:
        p = tp / predict
        r = tp / golden
        f1 = 2 * p * r / (p + r)
        return p, r, f1


def __evaluate_bc5(eval_data, eval_map):
    # count TP
    tp = 0
    for pmid, pair, rel in set(eval_data):
        arg1, arg2 = pair.split('_')

        if (pmid, arg1, arg2) in eval_map:
            tp += 1

    # count predict
    pred = len(set(eval_data))

    # count golden
    gold = len(eval_map)

    # calculate prf
    return __prf_bc5(tp, pred, gold)


def evaluate_bc5(answer, verbose=False):
    """
    :param list of (str, str, str) answer: pmid, pair, relation (n+1)
    :return:
    """
    # load intra annotation
    intra = []
    with open('bc5_2024/eval/bc5_intra.txt') as f:
        for l in f:
            pmid, pair = l.strip().split()
            e1, e2 = pair.split('_')
            intra.append((pmid, e1, e2))

    # load evaluate map
    full_eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)
    intra_eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)
    inter_eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)

    f = open('bc5_2024/eval/bc5_evaluate.txt')
    for line in f:
        pmid, rel, e1, e2 = line.strip().split()
        full_eval_map[(pmid, e1, e2)] = rel

        if (pmid, e1, e2) in intra:
            intra_eval_map[(pmid, e1, e2)] = rel
        else:
            inter_eval_map[(pmid, e1, e2)] = rel

    # split intra and inter answer evaluate map
    intra_answer = []
    inter_answer = []
    for rel in answer:
        pmid, pair, _ = rel
        e1, e2 = pair.split('_')
        if (pmid, e1, e2) in intra:
            intra_answer.append(rel)
        else:
            inter_answer.append(rel)

    if verbose:
        print('new model predict intra')
        for ans in intra_answer:
            print(ans[0], ans[1])

        print('new model predict inter')
        for ans in inter_answer:
            print(ans[0], ans[1])

    return __evaluate_bc5(answer, full_eval_map), __evaluate_bc5(intra_answer, intra_eval_map), __evaluate_bc5(inter_answer, inter_eval_map)


def evaluate_bc5_intra(eval_data):
    """
    :param list of (str, str, str) eval_data: pmid, pair, relation (n+1)
    :return:
    """
    # load evaluate map
    eval_map = {}  # dict (pmid, e1, e2) => relation (n + 1)
    f = open('bc5_2024/eval/bc5_evaluate_intra.txt')
    for line in f:
        pmid, rel, e1, e2 = line.strip().split()
        if rel.endswith('(r)'):
            eval_map[(pmid, e2, e1)] = rel[:-3]
        else:
            eval_map[(pmid, e1, e2)] = rel

    return __evaluate_bc5(eval_data, eval_map)
