from common.feature_utils import Bucketizer
from tpfy.cum_watch.metadata import CumWatchMetadata
from tpfy.tf_dataset.feature import schema


def normalize_vec_dict(data):
    total = sum(data.values())
    if total == 0:
        return
    for k in data:
        data[k] = data[k] / total


release_time_bucketier = Bucketizer([7, 30, 180, 365, 365 * 2, 365 * 3, 365 * 4])

past_watch_time_bucketizer = Bucketizer(
    [0, 600, 1800, 3600, 3600 * 2, 3600 * 3, 3600 * 10, 3600 * 20, 3600 * 40]
)
last_watch_to_now_bucketizer = Bucketizer(
    [86400 * d for d in [0, 3, 7, 14, 30, 60, 180]]
)


def extract_feature_using_fd(
    metadata: CumWatchMetadata,
    dw_p_id,
    content_id: int,
    langauge_id,
    timestamp,
    past_wt_offline,
    events,
    country,
):
    if events is None or len(events) == 0:
        return None

    assert isinstance(content_id, int)
    movies = metadata.cms_data.movies[country]
    tv_shows = metadata.cms_data.tv_shows[country]
    movie = movies.get(content_id, None)
    show = tv_shows.get(content_id, None)
    if movie is not None:
        content = movie
        content_type = 1
    elif show is not None:
        content = show
        content_type = 0
    else:
        return None

    last_event = None
    for e in events:
        if len(e["payload"]) > 0 and e.timestamp < timestamp:
            if last_event is None or last_event["timestamp"] < e["timestamp"]:
                last_event = e
    if last_event is None:
        return None

    from tpfy.proto.tpfy_predict_event_pb2 import TpfyPredictPayload

    payload = TpfyPredictPayload()
    payload.ParseFromString(last_event["payload"])
    user_fids = list(payload.user_feature.fids)
    user_weighted_fids = list(payload.user_feature.weighted_fids)
    user_weighted_fid_weights = list(payload.user_feature.fid_weights)

    raw_feature = payload.raw_feature

    # target item
    fids = []

    fids.append(schema.target_id.hash(str(content_id)))
    fids.append(schema.target_language_id.hash(str(langauge_id)))
    fids.append(schema.target_content_type_id.hash(str(content.content_type_id)))
    if len(content.genre_ids) == 0:
        genre_ids = ["empty"]
    else:
        genre_ids = content.genre_ids
    for genre_id in genre_ids:
        fids.append(schema.target_genre_id.hash(str(genre_id)))

    fids.append(schema.target_entitlement.hash(content.entitlement))
    fids.append(schema.target_parental_rating_id.hash(str(content.parental_rating)))

    past_wt_online = 0
    last_watch_ts = 0
    str_content_id = str(content_id)
    for w in raw_feature.ent_watches:
        if w.content_id == str_content_id:
            past_wt_online += w.watch_time
            last_watch_ts = max(last_watch_ts, w.timestamp)

    past_wt_bucket = past_watch_time_bucketizer.get_bucket(past_wt_online)
    fids.append(schema.target_past_watched_time_cumwt.hash(past_wt_bucket))

    if last_watch_ts > 0:
        last_watch_to_now = timestamp - last_watch_ts
        last_watch_to_now_bucket = (
            last_watch_to_now_bucketizer.get_bucket(last_watch_to_now) + 1
        )
    else:
        last_watch_to_now = 0
        last_watch_to_now_bucket = 0
    fids.append(schema.target_last_watch_to_now_cumwt.hash(last_watch_to_now_bucket))

    fids.append(schema.target_studio_id.hash(str(content.studio_id)))
    if show is not None:
        fids.append(schema.target_channel_id.hash(str(show.channel_id)))

        # if show.btv is None:
        #     show_btv = "unknown"
        # elif show.btv:
        #     show_btv = "true"
        # else:
        #     show_btv = "false"
        # fids.append(schema.target_show_btv.hash(show_btv))
        #
        # if show.show_type is None:
        #     show_type = "unknown"
        # else:
        #     show_type = show.show_type.lower()
        # fids.append(schema.target_show_type.hash(show_type))

    if content.start_dt < timestamp:
        release_days = (timestamp - content.start_dt) / 86400
    else:
        release_days = 0
    release_time_bucket = str(release_time_bucketier.get_bucket(release_days))
    fids.append(schema.target_release_time_bucket.hash(release_time_bucket))

    return {
        "user_hash": hash(dw_p_id),
        "dw_p_id": dw_p_id,
        "content_id": str(content_id),
        "language_id": langauge_id,
        "timestamp": timestamp,
        "user_fids": user_fids,
        "user_weighted_fids": user_weighted_fids,
        "user_weighted_fid_weights": user_weighted_fid_weights,
        "fids": fids,
        "weighted_fids": [0],
        "weighted_fid_weights": [0.0],
        "raw": [past_wt_offline, past_wt_online, last_watch_to_now, content_type],
    }
