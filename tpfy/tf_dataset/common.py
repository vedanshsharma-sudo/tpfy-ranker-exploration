def filter_sports_and_news_watches(
    ns_watches,
    ln_watches,
    sports,
    news,
    ts_filter_fn=None,
    watch_len_thres=600,
    clip_len_thres=30,
    clip_len_ratio=0.5,
):
    sports_watches, news_clips_watches, live_news_watches = [], [], []
    for watch in ns_watches:
        cid = watch.content_id
        if ts_filter_fn and not ts_filter_fn(watch.last_watch):
            continue

        if cid in sports:
            video = sports[cid]

            if video.content_type_id == 301 or video.content_type_id == 303:
                if watch.watch_len < watch_len_thres:
                    continue
            else:
                if watch.watch_len < clip_len_thres and (
                    video.duration <= 0
                    or watch.watch_len / video.duration < clip_len_ratio
                ):
                    continue
            sports_watches.append(watch)

        if cid in news:
            video = news[cid]

            if not video.content_type_id == 401:
                continue
            if watch.watch_len < clip_len_thres and (
                video.duration <= 0 or watch.watch_len / video.duration < clip_len_ratio
            ):
                continue
            news_clips_watches.append(watch)

    for watch in ln_watches:
        cid = watch.content_id
        if ts_filter_fn and not ts_filter_fn(watch.first_watch):
            continue

        if cid in news:
            video = news[cid]
            if not video.is_live_news or watch.watch_len < watch_len_thres:
                continue
            live_news_watches.append(watch)
    return sports_watches, news_clips_watches, live_news_watches
