from common.feature_utils import recency_decay
from common.cms_esknol_utils import get_sub_name_weight


class NSTokenBasedRetriever:
    HalfTimeDecay = 14 * 24 * 3600

    CONTENT_TYPE_WEIGHT = {
        100: 1.0,  # 'SERIES',
        102: 1.0,  # 'EPISODE',
        105: 1.0,  # 'SHOW_LIVE',
        200: 1.0,  # 'MOVIE',
        300: 0.1,  # 'SPORT',
        301: 1.0,  # 'SPORT_LIVE',
        303: 1.0,  # 'SPORT_REPLAY',
        304: 0.1,  # 'SPORT_MATCH_HIGHLIGHTS',
        401: 0.1,  # 'NEWS_CLIPS'
    }

    def __init__(self, relevances, ns_sub_names):
        self.relevances = relevances
        self.ns_sub_names = ns_sub_names

    def retrieve(self, ent_watches, ns_watches, current_ts, tv_shows, clips, matches, n=200, last_n=20, type_min=100):
        scores = {}
        watched = set([watch.content_id for watch in ent_watches])
        for watch in sorted(ns_watches, key=lambda t: t.last_watch, reverse=True)[0:last_n]:
            time_decay = recency_decay(watch.last_watch, current_ts,
                                       half_decay_time=self.HalfTimeDecay)
            if watch.watch_len <= 600:
                time_decay = time_decay * 0.1

            if watch.content_id in clips:
                watched_video = clips[watch.content_id]
            elif watch.content_id in matches:
                watched_video = matches[watch.content_id]
            else:
                continue

            content_type_id = watched_video.content_type_id
            content_type_weight = self.CONTENT_TYPE_WEIGHT.get(content_type_id, 0.0)
            # FixME(shenglong): hard-code
            len_decay = 1.0
            if watched_video.content_type_id == 301:
                if watch.watch_len < 600:
                    len_decay = 0.1
            elif watch.watch_len < 30 and \
                    (watched_video.duration <= 0 or watch.watch_len / watched_video.duration < 0.5):
                len_decay = 0.1

            watch_weight = content_type_weight * time_decay * len_decay
            ns_sub_weights = get_sub_name_weight(watched_video, watch.language, self.ns_sub_names)
            for name, weight in ns_sub_weights.items():
                for dst_id, s in self.relevances.get(name, []):
                    score = s * watch_weight * weight
                    scores[dst_id] = scores.get(dst_id, 0) + score

        results = []
        movie_count, show_count = 0, 0
        for dst_key, score in sorted(scores.items(), key=lambda t: 0 - t[1]):
            cid = dst_key[0]
            if cid in watched:
                continue

            is_show = cid in tv_shows
            if len(results) < n:
                results.append((dst_key, score))
                if is_show:
                    show_count += 1
                else:
                    movie_count += 1
            else:
                if show_count < type_min and is_show:
                    results.append((dst_key, score))
                    show_count += 1
                elif movie_count < type_min and not is_show:
                    results.append((dst_key, score))
                    movie_count += 1

                if show_count >= type_min and movie_count >= type_min:
                    break

        return results
