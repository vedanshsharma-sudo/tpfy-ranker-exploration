from collections import defaultdict

from common.config import TENANT
from common.config.constants import DataPath
from common.config.utils import data_path
from common.s3_utils import download_all_files, download_single_file
from common.data.behavior.watch import Watch2
from tpfy.common import DeviceInfo
from tpfy.tf_dataset.feature import XDeepFMFeatureExtractor
from tpfy.tf_model.retriver.item_based_retriever import ItemBasedRetriever
from tpfy.tf_model.retriver.new_published_retriever import NewPublishedRetriever
from tpfy.tf_model.retriver.popularity_burst_retriever import PopularityBurstRetriever
from tpfy.tf_model.retriver.popularity_retriever import PopularityRetriever

import os
import json
import csv
import boto3

OUTPUT = XDeepFMFeatureExtractor.output


def download_model_if_needed(model_name):
    if ":" not in model_name:
        return "tmp/model/export/%s" % model_name
    model_name, ts = model_name.split(":")
    model_path = "tmp/model/export/%s_%s" % (model_name, ts)
    if not os.path.exists(model_path):
        s3_model_path = os.path.join(
            data_path(DataPath.S3_TPFY_MODEL_EXPORT, TENANT, "") % model_name,
            "export",
            ts,
        )
        download_all_files(s3_model_path, model_path, reserve_structure=True)
    return model_path


def read_relevance_table(date):
    relevance_s3_path = data_path(DataPath.S3_ITEM_RELEVANCES, TENANT)
    relevance_path = "tmp/debug/relevances.csv"
    download_single_file(relevance_s3_path, relevance_path, ".csv")

    relevances = {}
    file = open(relevance_path, "r", encoding="utf-8")
    reader = csv.reader(file, delimiter=",")
    for cid, lang, dst_str in reader:
        cid, lang = int(cid), int(lang)
        if len(dst_str.strip()) == 0:
            continue
        tks = dst_str.split(",")
        dsts = []
        for i in range(0, len(tks), 3):
            dsts.append(((int(tks[i]), int(tks[i + 1])), float(tks[2 + i])))
        relevances[(cid, lang)] = dsts
    print("Relevance Count", len(relevances))
    return relevances


def fetch_behavior_from_ubs(pid, now):
    url = (
        "http://apubsvc-prod-in-apse1-v2.persona.prod.hotstar.corp/v2/user/%s/watch?types=tv_show,"
        "movie,sports&start_time=%s&end_time=%s" % (pid, now - 86400 * 180, now)
    )

    import urllib.request
    import urllib.parse

    f = urllib.request.urlopen(url)
    response = json.loads(f.read().decode("utf-8"))
    watches, ns_watches = [], []
    for behavior in response:
        watch = Watch2(
            content_id=int(behavior["content_id"]),
            language=behavior["language_id"],
            watch_len=behavior["watch_time"],
            first_watch=behavior["first_timestamp"],
            last_watch=behavior["timestamp"],
        )
        if watch.first_watch > now:
            continue
        watch_type = behavior["watch_type"]
        if watch_type in {2, 4}:
            watches.append(watch)
        else:
            ns_watches.append(watch)
    return watches, ns_watches


class UserInfo:
    def __init__(self):
        self.result = []
        self.age = None
        self.gender = None
        self.joined_on = -1
        self.sub_info = None

        # Note: this is always empty
        self.installed_app = []
        self.device_info = None

    def pretty(self):
        return """
            age:            %s
            gender:         %s
            joined_on:      %s
            sub_info:       %s
            platform:       %s
            country:        %s
            state:          %s
            model:          %s
            carrier:        %s
            asn_number:     %s
            manufacturer:   %s
            screen_height:  %s
        """


def fetch_info_from_dynamo(pid):
    client = boto3.client("dynamodb", region_name="ap-southeast-1")
    user_info = UserInfo()

    response = client.get_item(
        TableName="tpfy-off-ret-ddb-prod-in-apse1", Key={"p_id": {"S": pid}}
    )
    row = response.get("Item")
    if row:
        raw_result = row.get("result", {}).get("L", [])
        for cell in raw_result:
            user_info.result.append(int(cell["S"]))

        if "age" in row:
            user_info.age = row["age"]["S"]
        if "gender" in row:
            user_info.gender = row["gender"]["S"]
        if "joined_on" in row:
            user_info.joined_on = int(row["joined_on"]["N"])
        if "sub_info" in row:
            user_info.sub_info = row["sub_info"]["S"]

    response = client.get_item(
        TableName="p13n.prod.trays.ui.features", Key={"p_id": {"S": pid}}
    )
    row = response.get("Item", {})
    features = row["features"]["M"] if "features" in row else None
    updated_at = row["updated_at"]["N"] if "updated_at" in row else None
    if features:
        user_info.device_info = DeviceInfo(
            ts=updated_at,
            platform=features["platform"]["S"],
            country=features["country"]["S"],
            state=features["state"]["S"],
            city=None,
            carrier=features["carrier"]["S"],
            asn_number=features["asnNumber"]["S"],
            manufacturer=features["manufacturer"]["S"],
            screen_height=features["screenHeight"]["S"],
            model=features["device"]["S"],
        )
    return user_info


def bootstrap_retrievers(metadata, relevances, predict_ts):
    als_retriever = ItemBasedRetriever(relevances)
    new_contents = NewPublishedRetriever.get_new_published_content(
        metadata.movies, metadata.tv_shows, predict_ts
    )
    breaking_contents = PopularityBurstRetriever.get_breaking_content(
        metadata.day_pvs, predict_ts
    )
    popular_contents = PopularityRetriever.get_popular_content(
        metadata.day_pvs, predict_ts, metadata.movies, metadata.tv_shows
    )
    return als_retriever, new_contents, breaking_contents, popular_contents


def retrieve(watches, ns_watches, predict_ts, metadata, retrieve_capsule):
    cid_sources = defaultdict(list)
    candidates = set()
    als_retriever, new_contents, breaking_contents, popular_contents = retrieve_capsule
    item_based_candidates, evidences, weights = als_retriever.retrieve(
        watches, predict_ts, metadata.tv_shows, metadata.movies
    )
    item_based_candidates = [k for k, _ in item_based_candidates]
    new_published_candidates = NewPublishedRetriever.retrieve(watches, *new_contents)
    breaking_candidates = PopularityBurstRetriever.retrieve(watches, breaking_contents)
    popular_candidates = PopularityRetriever.retrieve(
        watches, ns_watches, popular_contents, metadata.movies, metadata.tv_shows
    )

    for name, results in zip(
        ["als", "new", "breaking", "popular"],
        [
            item_based_candidates,
            new_published_candidates,
            breaking_candidates,
            popular_candidates,
        ],
    ):
        for result in results:
            if result[0] not in metadata.tv_shows and result[0] not in metadata.movies:
                continue
            candidates.add(result)
            cid_sources[result].append(name)
    candidates = list(candidates)
    return candidates, cid_sources


def extract_features(watches, ns_watches, user_info, predict_ts, candidates, metadata):
    meta_feature = ([0],)
    (
        user_feature,
        sub_plan,
        aggregated_values,
    ) = XDeepFMFeatureExtractor.generate_user_features(
        watches,
        ns_watches,
        user_info.age,
        user_info.gender,
        user_info.joined_on,
        user_info.sub_info,
        user_info.installed_app,
        user_info.device_info,
        predict_ts,
        metadata.wv,
        metadata.tv_shows,
        metadata.movies,
        metadata.sports,
    )

    data = []
    for candidate_id, language_id in candidates:
        target_feature = XDeepFMFeatureExtractor.generate_target_feature(
            candidate_id,
            0.0,
            predict_ts,
            metadata.tv_shows,
            metadata.movies,
            language_id,
        )
        wide_feature = XDeepFMFeatureExtractor.generate_wide_features(
            candidate_id,
            aggregated_values,
            sub_plan,
            predict_ts,
            metadata.pv,
            metadata.tv_shows,
            metadata.movies,
            language_id,
        )
        data.append(meta_feature + user_feature + target_feature + wide_feature)

    padded_batch = []
    for i in range(len(OUTPUT)):
        max_len = max(len(point[i]) for point in data)
        batch_cell = []
        pad = [OUTPUT[i][3]]
        for point in data:
            batch_cell.append(point[i] + pad * (max_len - len(point[i])))
        padded_batch.append(batch_cell)
    return padded_batch, meta_feature, user_feature
