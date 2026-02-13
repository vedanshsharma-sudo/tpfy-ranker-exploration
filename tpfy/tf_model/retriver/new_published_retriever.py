from operator import itemgetter


class NewPublishedRetriever:
    NEW_PUBLISHED_DAYS_THRESHOLD = 7

    @staticmethod
    def get_new_published_content(movies, shows, now):
        days = NewPublishedRetriever.NEW_PUBLISHED_DAYS_THRESHOLD
        new_published_movies = [
            movie for movie in movies.values()
            if movie.content_type_id == 200 and 86400 + now > movie.start_dt > now - 86400 * days
        ]
        new_published_shows = [
            show for show in shows.values()
            if show.content_type_id == 100 and 86400 + now > show.episode_start_dt > now - 86400 * days
        ]
        print("==== new published candidates ====")
        print(len(new_published_movies), len(new_published_shows))
        for movie in new_published_movies:
            print(movie.content_id, movie.title, movie.start_dt)
        for show in new_published_shows:
            print(show.content_id, show.title, show.episode_start_dt)

        return new_published_movies, new_published_shows

    @staticmethod
    def retrieve(ent_watches, new_movies, new_shows):
        ent_languages = {}
        for watch in ent_watches:
            ent_languages[watch.language] = watch.watch_len + ent_languages.get(watch.language, 0)
        ent_languages = {lang: value for lang, value in ent_languages.items() if value > 3600}

        candidates = []
        for contents in [new_movies, new_shows]:
            for content in contents:
                if len(ent_languages) == 0:
                    lid = content.primary_language_id
                    candidates.append((content.content_id, lid))
                elif len(content.language_ids) > 0:
                    lid, _ = max([(lang, ent_languages.get(lang, 0)) for lang in content.language_ids],
                                 key=itemgetter(1))
                    if lid in ent_languages:
                        candidates.append((content.content_id, lid))

        return candidates
