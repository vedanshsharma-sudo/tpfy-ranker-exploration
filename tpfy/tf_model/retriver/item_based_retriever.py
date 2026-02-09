from math import pow
from common.feature_utils import recency_decay


class ItemBasedRetriever:
    HalfTimeDecay = 14 * 24 * 3600

    def __init__(self, relevances):
        self.relevances = relevances

    @staticmethod
    def time_decay(last_time, current_time):
        return pow(1.0 / 2, (current_time - last_time) * 1.0 / ItemBasedRetriever.HalfTimeDecay)

    def retrieve(self, watches, current_ts, tv_shows, movies, n=200, type_min=50, watch_len_thres=600):
        scores, evidences = {}, {}
        watch_weights, watched = {}, set()
        for watch in watches:
            if watch.watch_len < watch_len_thres:
                continue
            key = (watch.content_id, watch.language)
            watched.add(watch.content_id)
            decay = recency_decay(watch.last_watch, current_ts,
                                  half_decay_time=self.HalfTimeDecay)
            watch_weights[key] = decay
            for dst_key, score in self.relevances.get(key, []):
                scores[dst_key] = scores.get(dst_key, 0) + score * decay
                evidences[(key, dst_key)] = score

        if len(scores) == 0:
            for watch in watches:
                key = (watch.content_id, watch.language)
                watched.add(watch.content_id)
                decay = recency_decay(watch.last_watch, current_ts,
                                      half_decay_time=self.HalfTimeDecay)
                watch_weights[key] = decay
                for dst_key, score in self.relevances.get(key, []):
                    scores[dst_key] = scores.get(dst_key, 0) + score * decay
                    evidences[(key, dst_key)] = score

        retrieved = []
        movie_count, show_count, disneyplus_count = 0, 0, 0
        for dst_key, score in sorted(scores.items(), key=lambda t: 0-t[1]):
            cid, lang = dst_key
            is_show = cid in tv_shows
            if cid in watched:
                continue
            if len(retrieved) < n:
                retrieved.append((dst_key, score))
                if is_show:
                    show_count += 1
                else:
                    movie_count += 1
                if cid in movies and movies[cid].is_disneyplus:
                    disneyplus_count += 1
                elif cid in tv_shows and tv_shows[cid].is_disneyplus:
                    disneyplus_count += 1
            else:
                appended = False
                if show_count < type_min and is_show:
                    if cid in tv_shows and tv_shows[cid].is_disneyplus:
                        disneyplus_count += 1
                        appended = True
                    retrieved.append((dst_key, score))
                    show_count += 1
                elif movie_count < type_min and not is_show:
                    if cid in movies and movies[cid].is_disneyplus:
                        disneyplus_count += 1
                        appended = True
                    retrieved.append((dst_key, score))
                    movie_count += 1
                if not appended and disneyplus_count < type_min \
                        and ((cid in movies and movies[cid].is_disneyplus)
                             or (cid in tv_shows and tv_shows[cid].is_disneyplus)):
                        disneyplus_count += 1
                        retrieved.append((dst_key, score))

                if show_count >= type_min and movie_count >= type_min and disneyplus_count >= type_min:
                    break
        return retrieved, evidences, watch_weights
