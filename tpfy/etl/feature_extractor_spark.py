import os
from typing import List
import datetime
import random
from common.data.behavior.watch import Watch2
from tpfy.tf_dataset.metadata import Metadata, PopularityValue
from tpfy.tf_dataset.feature import FeatureSchema, ScalarIndex
from tpfy.etl.emr_train_dataset import ImprExample, MtlExample
from tpfy.etl.schema import TpfyDatasetSchema, TpfyMtlDatasetSchema
from tpfy.common import TpfyDataPath
from common.config.utils import data_path
from common.s3_utils import is_s3_file_exist, download_single_file, upload_file
from common.feature_utils import log_recency_decay
from model.content_meta_utils import _year_bucket
from model.emr_utils import dict_to_spark_row
import pickle


def emr_load_metadata(sql_context, date, tenant, countries, use_cached_metadata=True):
    metadata_path_template = TpfyDataPath.S3_TPFY_IMPR_V3_DAILY_METADATA_CACHE_CMS3
    s3_cache_path = data_path(metadata_path_template, tenant) % date
    local_filename = "metadata.pkl"
    print(f"metadata cache path {s3_cache_path}; countries {countries}")
    if is_s3_file_exist(s3_cache_path) and use_cached_metadata:
        print("load cached metadata", s3_cache_path)
        download_single_file(s3_cache_path, local_filename, from_folder=False)
        with open(local_filename, "rb") as f:
            metadata = pickle.load(f)
        os.remove(local_filename)
        return metadata
    else:
        metadata = Metadata(date=date, countries=countries)
        metadata.load(sql_context=sql_context)
        with open(local_filename, "wb") as f:
            pickle.dump(metadata, f)
        print("upload metadata cache", s3_cache_path)
        upload_file(s3_cache_path, local_filename)
        os.remove(local_filename)
        return metadata


class SerializedFeatureKey:
    netAcuityGeoCity = 1
    netAcuityGeoState = 2
    adsGender = 3
    adsAge = 4
    discoveryPopularityVersion = 5
    stateNoLangDailyPopularityVersion = 6
    state = 7
    region = 8


def deserialize_string_feature(serialized_features, key):
    if key not in serialized_features:
        return None
    return serialized_features[key].decode("utf8")


def _coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None


class UserState:
    def __init__(self, ts, payload, cms3):
        self.cms3 = cms3

        self.user_fids = list(payload.user_feature.fids)
        self.user_weighted_fids = list(payload.user_feature.weighted_fids)
        self.user_fid_weights = list(payload.user_feature.fid_weights)

        self.studio_weights = {}
        self.year_bucket_weights = {}
        self.parental_rating_weights = {}
        self.genre_weights = {}
        self.production_house_weights = {}
        self.content_type_weights = {}

        self.cur_year = datetime.datetime.utcfromtimestamp(ts).year

        self.all_weights_dict = [
            self.studio_weights,
            self.year_bucket_weights,
            self.parental_rating_weights,
            self.genre_weights,
            self.production_house_weights,
            self.content_type_weights,
        ]

        serialized_features = payload.raw_feature.serialized_features
        self.discover_popularity_version = deserialize_string_feature(
            serialized_features, SerializedFeatureKey.discoveryPopularityVersion
        )
        self.state_nolang_pop_version = deserialize_string_feature(
            serialized_features, SerializedFeatureKey.stateNoLangDailyPopularityVersion
        )
        self.net_acuity_geo_state = deserialize_string_feature(
            serialized_features, SerializedFeatureKey.netAcuityGeoState
        )
        self.region = deserialize_string_feature(
            serialized_features, SerializedFeatureKey.region
        )
        self.state = deserialize_string_feature(
            serialized_features, SerializedFeatureKey.state
        )

    def normalize(self):
        for ws in self.all_weights_dict:
            if len(ws) == 0:
                continue
            normalizer = sum(ws.values())
            if normalizer == 0:
                # FIXME: shouldn't happen but exists; 2024-12-05
                ws.clear()
            else:
                for k, v in ws.items():
                    ws[k] = v / normalizer


def incr_weight(d, key, incr):
    d[key] = d.get(key, 0.0) + incr


def gen_state(event, metadata, country):
    from tpfy.proto.tpfy_predict_event_pb2 import TpfyPredictPayload

    ts = event["timestamp"]
    payload = TpfyPredictPayload()
    payload.ParseFromString(event["payload"])
    watches = payload.raw_feature.ent_watches
    watch_weights = payload.raw_feature.ent_watch_weights

    state = UserState(ts, payload, cms3=payload.cms3)

    if len(watch_weights) != len(watches):
        raise Exception("watch and weight count mismatch")
    for watch, weight in zip(watches, watch_weights):
        int_cid = int(watch.content_id)
        movie = metadata.movies[country].get(int_cid, None)
        show = metadata.tv_shows[country].get(int_cid, None)
        if movie is not None:
            content = movie
        elif show is not None:
            content = show
        else:
            continue
        incr_weight(state.studio_weights, content.studio_id, weight)
        incr_weight(state.parental_rating_weights, content.parental_rating, weight)
        for genre_id in content.genre_ids:
            incr_weight(state.genre_weights, genre_id, weight)

        incr_weight(state.content_type_weights, content.content_type_id, weight)
        incr_weight(
            state.year_bucket_weights,
            _year_bucket(content.year, state.cur_year),
            weight,
        )
        incr_weight(state.production_house_weights, content.production_house_id, weight)
    state.normalize()
    return state


class ExtractConfig:
    def __init__(self):
        pass


def extract_features_from_online(
    dw_p_id,
    examples,
    events,
    metadata: Metadata,
    country,
    task,
    extract_config: ExtractConfig,
    mtl=False,
):
    if examples is None or len(examples) == 0:
        return []
    if events is None or len(events) == 0:
        return []

    examples = sorted(examples, key=lambda ex: ex["timestamp"])
    event_cursor = 0
    positives = []
    negatives = []
    user_state = None
    schema = FeatureSchema
    for ex in examples:
        if events[event_cursor]["timestamp"] >= ex["timestamp"]:
            continue
        while (
            event_cursor + 1 < len(events)
            and events[event_cursor + 1]["timestamp"] < ex["timestamp"]
        ):
            event_cursor += 1
            user_state = None
        if event_cursor >= len(events):
            break

        event = events[event_cursor]
        if user_state is None:
            user_state = gen_state(event, metadata, country)

        user_fids = user_state.user_fids
        user_weighted_fids = user_state.user_weighted_fids
        user_weighted_fid_weights = user_state.user_fid_weights
        ts = event["timestamp"]

        content_id = ex["content_id"]
        str_content_id = str(content_id)
        movie = metadata.movies[country].get(content_id, None)
        show = metadata.tv_shows[country].get(content_id, None)
        if movie is not None:
            content = movie
        elif show is not None:
            content = show
        else:
            continue

        genre_ids = [str(gid) for gid in content.genre_ids]
        if len(genre_ids) == 0:
            genre_ids = ["empty"]
        year_bucket = _year_bucket(content.year, user_state.cur_year)
        if show is not None:
            start_dt = show.episode_start_dt
        else:
            start_dt = movie.start_dt
        fids = [
            schema.target_id.hash(str(content_id)),
            *(schema.target_genre_id.hash(str(genre_id)) for genre_id in genre_ids),
            schema.target_entitlement.hash(str(content.entitlement)),
            schema.target_content_type_id.hash(str(content.content_type_id)),
            schema.target_parental_rating_id.hash(str(content.parental_rating)),
            schema.target_studio_id.hash(str(content.studio_id)),
            schema.target_production_house_id.hash(str(content.production_house_id)),
            schema.target_year_bucket.hash(str(year_bucket)),
            schema.target_priority.hash(content.priority),
        ]

        genre_weight = sum(
            user_state.genre_weights.get(gid, 0.0) for gid in content.genre_ids
        )

        geo_state = _coalesce(
            user_state.net_acuity_geo_state, user_state.state, user_state.region, "mh"
        )
        if user_state.state_nolang_pop_version is not None:
            state_pop_repo = metadata.state_nolang_popularity[
                country
            ].get_or_close_version(user_state.state_nolang_pop_version)
            geo_state = geo_state.lower()
            key = (geo_state, str_content_id)
            state_pv = state_pop_repo.get(key, PopularityValue())
        else:
            state_pv = PopularityValue()

        if user_state.discover_popularity_version is not None:
            pop_repo = metadata.discover_popularity[country].get_or_close_version(
                user_state.discover_popularity_version
            )
            if len(content.language_ids) != 0:
                primary_lang_id = content.language_ids[0]
            else:
                primary_lang_id = -1

            key = (str_content_id, primary_lang_id)
            pv = pop_repo.get(key, PopularityValue())
        else:
            pv = PopularityValue()

        scalar_features = (
            (ScalarIndex.genre_weight, genre_weight),
            (
                ScalarIndex.parental_rating_weight,
                user_state.parental_rating_weights.get(content.parental_rating, 0.0),
            ),
            (
                ScalarIndex.content_type_weight,
                user_state.content_type_weights.get(content.content_type_id, 0.0),
            ),
            (
                ScalarIndex.studio_weight,
                user_state.studio_weights.get(content.studio_id, 0.0),
            ),
            (
                ScalarIndex.year_bucket_weight,
                user_state.year_bucket_weights.get(year_bucket, 0.0),
            ),
            (ScalarIndex.log_recency_weight, log_recency_decay(ts, start_dt)),
            (ScalarIndex.release_3d, 1.0 if ts - start_dt < 86400 * 3 else 0.0),
            (ScalarIndex.release_7d, 1.0 if ts - start_dt < 86400 * 7 else 0.0),
            (ScalarIndex.release_30d, 1.0 if ts - start_dt < 86400 * 30 else 0.0),
            (ScalarIndex.stateNoLangPopularityDV, state_pv.dv),
            (ScalarIndex.stateNoLangPopularityWV, state_pv.wv),
            (ScalarIndex.primaryLangPopularityDV, pv.dv),
            (ScalarIndex.primaryLangPopularityWV, pv.wv),
        )

        if not mtl:
            label = ex["label"]
            labels = {"label": [label]}
        else:
            label = ex["click"]
            labels = {
                "click": [ex["click"]],
                "watch": [ex["watch"]],
                "watch_time": [ex["add_watchlist"]],
                "paywall_view": [ex["paywall_view"]],
                "add_watchlist": [ex["add_watchlist"]],
            }

        row = {
            "country": country,
            "dw_p_id": dw_p_id,
            "content_id": content_id,
            "timestamp": ts,
            "task": [task],
            "flag": task,
            "user_fids": user_fids,
            "user_weighted_fids": user_weighted_fids,
            "user_weighted_fid_weights": user_weighted_fid_weights,
            "fids": fids,
            "weighted_fids": [0],
            "weighted_fid_weights": [0.0],
            "sparse_indices": [p[0] for p in scalar_features],
            "sparse_values": [p[1] for p in scalar_features],
            "secs_start_dt": ts - start_dt,
            **labels,
        }
        if label > 0:
            positives.append(row)
        else:
            negatives.append(row)

    if not mtl:
        if len(positives) > 0 and len(negatives) > 0:
            return positives + negatives
        else:
            return []
    else:
        return positives + negatives


def generate_random_negatives(
    country, metadata: Metadata, positive, candidate_ids, visited, num
):
    timestamp = positive["timestamp"]
    full_examples = [positive]
    touched = {positive["content_id"]}
    for i in range(num):
        candidate_id = random.choice(candidate_ids)
        if candidate_id in visited or candidate_id in touched:
            continue
        if candidate_id in metadata.movies[country]:
            content = metadata.movies[country][candidate_id]
            if content.start_dt > timestamp:
                continue
        elif candidate_id in metadata.tv_shows[country]:
            content = metadata.tv_shows[country][candidate_id]
            if content.episode_start_dt > timestamp:
                continue
        else:
            raise Exception(f"illegal sampled content id {candidate_id}")
        touched.add(candidate_id)
        full_examples.append(
            ImprExample(
                tray_id="",
                content_id=candidate_id,
                timestamp=positive["timestamp"],
                label=0,
                watch_time=0,
                is_inter_tray=False,
            )._asdict()
        )
    return full_examples


def extract_features(
    dw_p_id,
    tpfy_examples,
    lw_examples,
    events,
    raw_complete_watches,
    metadata,
    candidate_ids,
    country,
    lw_sample_rate,
    num_random_neg,
    extract_config: ExtractConfig,
):
    if events is None or len(events) == 0:
        return []

    events = sorted(events, key=lambda e: e["timestamp"])

    tpfy_examples_with_feature = extract_features_from_online(
        dw_p_id,
        tpfy_examples,
        events,
        metadata,
        country,
        task=0,
        extract_config=extract_config,
    )

    lastwatch_examples_with_feature = []
    if random.random() < lw_sample_rate and lw_examples:
        complete_watches = Watch2.parse_watches_fn()(raw_complete_watches)
        touched_content_ids = {w.content_id for w in complete_watches}
        # TODO: random sampling
        full_lw_examples = generate_random_negatives(
            country=country,
            metadata=metadata,
            candidate_ids=candidate_ids,
            positive=lw_examples[0],
            visited=touched_content_ids,
            num=num_random_neg,
        )

        lastwatch_examples_with_feature = extract_features_from_online(
            dw_p_id,
            full_lw_examples,
            events,
            metadata,
            country,
            task=1,
            extract_config=extract_config,
        )
    rows = [
        dict_to_spark_row(TpfyDatasetSchema, row)
        for row in tpfy_examples_with_feature + lastwatch_examples_with_feature
    ]

    return rows


def mtl_extract_features(
    dw_p_id,
    tpfy_mtl_examples,
    lw_examples,
    events,
    raw_complete_watches,
    metadata,
    candidate_ids,
    country,
    lw_sample_rate,
    num_random_neg,
    extract_config: ExtractConfig,
):
    if events is None or len(events) == 0:
        return []

    events = sorted(events, key=lambda e: e["timestamp"])

    tpfy_examples_with_feature = extract_features_from_online(
        dw_p_id,
        tpfy_mtl_examples,
        events,
        metadata,
        country,
        task=0,
        extract_config=extract_config,
        mtl=True,
    )

    lastwatch_examples_with_feature = []
    if random.random() < lw_sample_rate and lw_examples:
        complete_watches = Watch2.parse_watches_fn()(raw_complete_watches)
        touched_content_ids = {w.content_id for w in complete_watches}
        # TODO: random sampling
        full_lw_examples: List[dict] = generate_random_negatives(
            country=country,
            metadata=metadata,
            candidate_ids=candidate_ids,
            positive=lw_examples[0],
            visited=touched_content_ids,
            num=num_random_neg,
        )
        full_lw_mtl_examples = [
            MtlExample(
                tray_id="",
                content_id=ex["content_id"],
                timestamp=ex["timestamp"],
                click=ex["label"],
                watch=ex["label"],
                watch_time=ex["watch_time"],
                paywall_view=0,
                add_watchlist=0,
            )._asdict()
            for ex in full_lw_examples
        ]

        lastwatch_examples_with_feature = extract_features_from_online(
            dw_p_id,
            full_lw_mtl_examples,
            events,
            metadata,
            country,
            task=1,
            extract_config=extract_config,
            mtl=True,
        )
    rows = [
        dict_to_spark_row(TpfyMtlDatasetSchema, row)
        for row in tpfy_examples_with_feature + lastwatch_examples_with_feature
    ]

    return rows
