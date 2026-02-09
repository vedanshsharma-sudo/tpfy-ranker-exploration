from collections import defaultdict
from common.offline_config import USER_DEFAULT_LANGUAGES


class PopularityRetriever:

    @staticmethod
    def get_popular_content(day_pv, now, movies, tv_shows, n=40, type_min=20):
        end_day = now // 86400 - 1
        popular_candidates = defaultdict(list)
        movie_count, show_count, disneyplus_count = {}, {}, {}
        for (cid, lid), pop_end in sorted(day_pv[end_day].items(), key=lambda t: -t[1]):
            if cid not in tv_shows and cid not in movies:
                continue
            is_show = cid in tv_shows
            content = tv_shows[cid] if is_show else movies[cid]
            if len(popular_candidates[lid]) < n:
                popular_candidates[lid].append(cid)
                if is_show:
                    show_count[lid] = show_count.get(lid, 0) + 1
                else:
                    movie_count[lid] = movie_count.get(lid, 0) + 1
                if content.is_disneyplus:
                    disneyplus_count[lid] = disneyplus_count.get(lid, 0) + 1
            else:
                appended = False
                if show_count.get(lid, 0) < type_min and is_show:
                    popular_candidates[lid].append(cid)
                    show_count[lid] = show_count.get(lid, 0) + 1
                    appended = True
                elif movie_count.get(lid, 0) < type_min and not is_show:
                    popular_candidates[lid].append(cid)
                    movie_count[lid] = movie_count.get(lid, 0) + 1
                    appended = True
                if content.is_disneyplus and (appended or disneyplus_count.get(lid, 0) < type_min):
                    disneyplus_count[lid] = disneyplus_count.get(lid, 0) + 1
                    if not appended:
                        popular_candidates[lid].append(cid)

        print("==== popular candidates ====")
        print(len(popular_candidates))
        for lid, contents in popular_candidates.items():
            print(lid, [(cid, day_pv[end_day][(cid, lid)]) for cid in contents])

        return popular_candidates

    @staticmethod
    def retrieve(ent_watches, ns_watches, popular_candidates, movies, tv_shows, n=40, type_min=20):
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
            for i, cid in enumerate(popular_candidates.get(lid, [])):
                candidates.append(((cid, lid), (i + 0.5) / weight))

        results = []
        for (cid, lid), _ in sorted(candidates, key=lambda t: t[1]):
            movie_count, show_count, disneyplus_count = 0, 0, 0
            is_show = cid in tv_shows
            content = tv_shows[cid] if is_show else movies[cid]
            if len(results) < n:
                results.append((cid, lid))
                if is_show:
                    show_count += 1
                else:
                    movie_count += 1
                if content.is_disneyplus:
                    disneyplus_count += 1
            else:
                appended = False
                if show_count < type_min and is_show:
                    results.append((cid, lid))
                    show_count += 1
                    appended = True
                elif movie_count < type_min and not is_show:
                    results.append((cid, lid))
                    movie_count += 1
                    appended = True
                if content.is_disneyplus and (appended or disneyplus_count < type_min):
                    disneyplus_count += 1
                    if not appended:
                        results.append((cid, lid))
        return results
