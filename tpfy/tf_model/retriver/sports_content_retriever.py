from collections import defaultdict
from common.offline_config import USER_DEFAULT_LANGUAGES

_SPORTS_GENRE_IDS = {19, 21, 25, 26, 27, 29, 30, 31, 32, 33, 34, 39, 44, 45, 49, 50, 54, 55}


class SportsContentRetriever:

    @staticmethod
    def get_sports_content(movies, shows, now, lookback_days=90):
        end_day = now - lookback_days * 86400
        sports_content = defaultdict(list)
        for cid, movie in movies.items():
            if movie.content_type_id != 200:
                continue
            if len(movie.genre_ids.intersection(_SPORTS_GENRE_IDS)) == 0:
                continue
            if movie.start_dt < end_day or movie.start_dt > now:
                continue
            for lid in movie.language_ids:
                sports_content[lid].append(cid)

        for cid, show in shows.items():
            if len(show.genre_ids.intersection(_SPORTS_GENRE_IDS)) == 0:
                continue
            if show.last_broadcast_date < end_day:
                continue
            if show.first_broadcast_date > now:
                continue
            for lid in show.language_ids:
                sports_content[lid].append(cid)

        print("==== sports candidates ====")
        for lid, contents in sports_content.items():
            print(lid, contents)

        return sports_content

    @staticmethod
    def retrieve(ent_watches, ns_watches, sports_content):
        languages = {}
        for watch in ent_watches:
            if watch.watch_len > 0:
                languages[watch.language] = watch.watch_len + languages.get(watch.language, 0)
        for watch in ns_watches:
            if watch.watch_len > 0:
                languages[watch.language] = watch.watch_len + languages.get(watch.language, 0)
        if len(languages) == 0:
            languages = USER_DEFAULT_LANGUAGES

        candidates = []
        for lid, weight in sorted(languages.items(), key=lambda t: -t[1])[0:2]:
            for i, cid in enumerate(sports_content.get(lid, [])):
                candidates.append(((cid, lid), (i + 0.5) / weight))
        candidates.sort(key=lambda t: t[1])

        return [k for k, _ in candidates[0:20]]
