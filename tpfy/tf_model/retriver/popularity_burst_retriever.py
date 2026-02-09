from collections import defaultdict
from operator import itemgetter


class PopularityBurstRetriever:
    POPULARITY_THRESHOLD = 0.7
    BREAKING_THRESHOLD = 0.2
    LOOK_BACK_WINDOW = 7

    @staticmethod
    def get_breaking_content(day_pv, now):
        end_day = now // 86400 - 1
        start_day = end_day - PopularityBurstRetriever.LOOK_BACK_WINDOW
        print(end_day, start_day)
        breaking_candidates = defaultdict(dict)
        for key, pop_end in day_pv[end_day].items():
            pop_start = day_pv[start_day].get(key, 0.0)
            if pop_end > PopularityBurstRetriever.POPULARITY_THRESHOLD and \
                    pop_end - pop_start > PopularityBurstRetriever.BREAKING_THRESHOLD:
                breaking_candidates[key[0]][key[1]] = pop_end

        print("==== breaking candidates ====")
        print(len(breaking_candidates))
        for cid, langs in breaking_candidates.items():
            for lid in langs:
                key = (cid, lid)
                print(key, day_pv[end_day][key], day_pv[start_day].get(key, 0.0))

        return breaking_candidates

    @staticmethod
    def retrieve(ent_watches, breaking_candidates):
        ent_languages = {}
        for watch in ent_watches:
            ent_languages[watch.language] = watch.watch_len + ent_languages.get(watch.language, 0)
        ent_languages = {lang: value for lang, value in ent_languages.items() if value > 3600}

        candidates = []
        for content_id, langs in breaking_candidates.items():
            if len(ent_languages) == 0:
                lid, _ = max(langs.items(), key=itemgetter(1))
            else:
                lid, _ = max(langs.items(), key=lambda t: ent_languages.get(t[0], 0.0))
                if lid not in ent_languages:
                    lid = None

            if lid is not None:
                candidates.append((content_id, lid))
        return candidates
