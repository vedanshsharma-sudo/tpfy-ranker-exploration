import numpy as np
from operator import itemgetter
from common.offline_config import get_config
import random

from common.utils import is_premium


def diverse_by_source(results, evidences, watch_weights, watch_limit=20):
    if len(watch_weights) > watch_limit:
        trimmed_weights = {}
        for cid, weight in sorted(
            watch_weights.items(), key=itemgetter(1), reverse=True
        ):
            trimmed_weights[cid] = weight
            if len(trimmed_weights) == watch_limit:
                break
        watch_weights = trimmed_weights

    source_capacities = [c for _, c in watch_weights.items()]
    source_volumes = [1.0] * len(source_capacities)

    diversified_results = []
    for r, w in results:
        scores = [evidences.get((s, r), 0) for s in watch_weights]
        mi = np.argmax(scores)
        if scores[mi] <= 0:
            new_w = 0.0
        else:
            new_w = w * source_capacities[mi] / source_volumes[mi]
        source_volumes[mi] += 1.0
        diversified_results.append((r, new_w))

    diversified_results.sort(key=itemgetter(1), reverse=True)
    return diversified_results


def impression_discount(
    results, impressions, p_id, date_str, salt="home", discount_start=2, max_impres=10
):
    if impressions is None:
        return results

    config = get_config()
    content_count = {}
    for imp in impressions:
        if imp["tray_id"] == config["tpfy"]["tray_id"]["HOME_TPFY"]:
            content_count[imp["content_id"]] = (
                content_count.get(imp["content_id"], 0) + imp["count"]
            )
    results.sort(key=itemgetter(1), reverse=True)
    min_score = min(s for _, s in results)
    random.seed(p_id + date_str + salt)
    randomized_results = []
    for cid, w in results:
        if content_count.get(cid, 0) > discount_start:
            discount_weight = (content_count[cid] - discount_start) / (
                max_impres - discount_start
            )
            discount_weight = min(1.0, discount_weight)
            rw = (w - min_score) * (
                discount_weight * random.random() + (1 - discount_weight)
            ) + min_score
            randomized_results.append((cid, rw))
        else:
            randomized_results.append((cid, w))
    randomized_results.sort(key=itemgetter(1), reverse=True)
    final_results = []
    for (_, w), (cid, _) in zip(results, randomized_results):
        final_results.append((cid, w))
    return randomized_results


def random_by_user_date(
    results, p_id, date_str, salt="home", random_max=1.0, random_min=0.5
):
    results.sort(key=itemgetter(1), reverse=True)
    random.seed(p_id + date_str + salt)
    randomized_results = []
    for cid, w in results:
        if w > 0:
            rw = w * ((random_max - random_min) * random.random() + random_min)
            randomized_results.append((cid, rw))
        else:
            randomized_results.append((cid, w))
    randomized_results.sort(key=itemgetter(1), reverse=True)
    final_results = []
    for (_, w), (cid, _) in zip(results, randomized_results):
        final_results.append((cid, w))
    return randomized_results


def filter_premium_contents(results, watches, shows, movies, premium_decay_step=4):
    premium = any(is_premium(watch.content_id, movies, shows) for watch in watches)
    if premium:
        return results

    filtered_results = []
    premium_count = 0
    for (cid, lang), s in results:
        if is_premium(cid, shows, movies):
            s = s * (0.5 ** (premium_count // premium_decay_step))
            premium_count += 1
        filtered_results.append(((cid, lang), s))

    filtered_results.sort(key=itemgetter(1), reverse=True)
    return filtered_results


def combine_ent_ns_results(ns_weight, ent_results, ns_results):
    results = dict(ent_results)
    combined_results = []
    for c, s in ns_results:
        results[c] = results.get(c, 0.0) + s * ns_weight
        ns_weight = ns_weight * 0.8
    for c, s in results.items():
        combined_results.append((c, results[c]))
    combined_results.sort(key=itemgetter(1), reverse=True)
    return combined_results


def filter_by_language(ent_watches, results, min_lang_len=600):
    if len(results) == 0:
        return results

    ent_langs = {}
    for watch in ent_watches:
        ent_langs[watch.language] = ent_langs.get(watch.language, 0) + watch.watch_len
    valid_langs = set(
        lid for lid, watch_len in ent_langs.items() if watch_len >= min_lang_len
    )
    results = sorted(results, key=itemgetter(1), reverse=True)
    min_score = results[-1][1]
    filtered_results, waiting_results = [], []
    for (cid, lid), score in results:
        if len(valid_langs) == 0 or lid in valid_langs:
            filtered_results.append(((cid, lid), score))
        else:
            waiting_results.append(
                ((cid, lid), min_score - 1 + 1 / (len(waiting_results) + 1))
            )
    filtered_results.extend(waiting_results)
    return filtered_results
