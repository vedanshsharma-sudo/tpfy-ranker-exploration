import numpy as np
import s3fs
import os
from collections import defaultdict
import csv
import json
import argparse
from common.cms3.cms3_utils import get_cms_store
from common.s3_utils import download_single_file, set_s3_file_acl
from common.config import TENANT
from common.config.utils import data_path, tenant_countries
from common.config.constants import DataPath
from common.spark_utils import create_spark_context, create_sql_context, set_partitions

# from common.hive_utils import create_hive_connection
import bisect


def main(args):
    end_date = args.end_date
    day_range = args.day_range
    countries = tenant_countries(args.countries)

    spark_context = create_spark_context(
        "Watched cumulative Watch - %s" % end_date, enable_hive=True
    )
    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 1)

    movies, tv_shows = get_cms_store(
        country_list=countries,
        movie_fields=[],
        tv_show_fields=[],
        # hive_connector=create_hive_connection(),
        sql_context=sql_context,
    )
    content_type_map = {}
    for country in countries:
        for content_id, tv_show in tv_shows.all_entities_of_country(country).items():
            content_type_map[content_id] = "show"
        for content_id, movie in movies.all_entities_of_country(country).items():
            content_type_map[content_id] = "movie"

    watch_distribution_dict = defaultdict(
        lambda: defaultdict(
            lambda: {
                "watch_per_day": np.zeros([day_range], dtype=np.float64),
                "user_per_day": np.zeros([day_range], dtype=np.float64),
            }
        )
    )
    dimension_values = defaultdict(lambda: defaultdict(int))
    dimension_bins = {}
    dimension_dict = defaultdict(
        lambda: defaultdict(
            lambda: {
                "watch_per_day": np.zeros([day_range], dtype=np.float64),
                "user_per_day": np.zeros([day_range], dtype=np.float64),
            }
        )
    )
    file_path = data_path(DataPath.S3_CUMULATIVE_WATCH_BIN, TENANT) % end_date
    base_dir = "etl_tmp"
    local_path = os.path.join(base_dir, "cumulative_watch.csv")
    download_single_file(file_path, local_path, ".csv")

    with open(local_path) as f:
        reader = csv.DictReader(f)
        for _, row_dict in enumerate(reader):
            content_id = int(row_dict["content_id"])
            if content_id not in content_type_map:
                print(content_id)
                continue
            for i in range(args.max_bin_num):
                left_bin = row_dict[f"bin_{i}"]
                if left_bin.isdigit():
                    watch_distribution_dict[content_id][i]["bin"] = int(left_bin)

    file_path = data_path(DataPath.S3_WATCHED_WATCH_DISTRIBUTION, TENANT) % end_date
    base_dir = "etl_tmp"
    local_path = os.path.join(base_dir, "watch_distribution.csv")
    download_single_file(file_path, local_path, ".csv")

    with open(local_path) as f:
        reader = csv.DictReader(f)
        for _, row_dict in enumerate(reader):
            content_id = int(row_dict["content_id"])
            group = int(row_dict["group"])
            if content_id not in content_type_map:
                print(content_id)
                continue

            watch_per_day = np.array(
                [int(row_dict[f"sum_wt_{i}"]) for i in range(day_range)],
                dtype=np.float64,
            )
            user_per_day = np.array(
                [int(row_dict[f"sum_count_{i}"]) for i in range(day_range)],
                dtype=np.float64,
            )
            watch_distribution_dict[content_id][group]["watch_per_day"] += watch_per_day
            watch_distribution_dict[content_id][group]["user_per_day"] += user_per_day
            if "bin" not in watch_distribution_dict[content_id][group]:
                if group != 0:
                    print(f"content_id: {content_id}")
                    print(watch_distribution_dict[content_id])
                    print(f"group: {group}")
                    print(watch_distribution_dict[content_id][group])
                    raise Exception("group != 0 but no bin")
                if group == 0:
                    watch_distribution_dict[content_id][group]["bin"] = 0
            dimension_values[content_type_map[content_id]][
                watch_distribution_dict[content_id][group]["bin"]
            ] += int(row_dict["num"])

    for key in dimension_values:
        second_counts = []
        for second, count in dimension_values[key].items():
            if second <= 60:
                continue
            else:
                second_counts.append([second, count])
        second_counts = sorted(second_counts, key=lambda k: k[0])
        total = sum(count for second, count in second_counts)
        bin_num = args.max_bin_num - 2
        per_bin = total // bin_num
        bin_right_edges = [0, 60]
        current_bin_num = 0
        for second, count in second_counts:
            current_bin_num += count
            if current_bin_num >= per_bin:
                bin_right_edges.append(second)
                current_bin_num = 0
        dimension_bins[key] = bin_right_edges[: args.max_bin_num]

    for content_id in watch_distribution_dict:
        content_type = content_type_map[content_id]
        for group in watch_distribution_dict[content_id]:
            watch_distribution = watch_distribution_dict[content_id][group]
            cluster_group = min(
                bisect.bisect_left(
                    dimension_bins[content_type], watch_distribution["bin"]
                ),
                len(dimension_bins[content_type]) - 1,
            )
            dimension_dict[content_type][cluster_group][
                "watch_per_day"
            ] += watch_distribution["watch_per_day"]
            dimension_dict[content_type][cluster_group][
                "user_per_day"
            ] += watch_distribution["user_per_day"]

    for content_type in dimension_dict:
        for group in dimension_dict[content_type]:
            dimension_watch = dimension_dict[content_type][group]
            dimension_watch["watch_per_user"] = dimension_watch[
                "watch_per_day"
            ] / np.maximum(dimension_watch["user_per_day"], 1)
            dimension_watch["complete_watch"] = np.cumsum(
                dimension_watch["watch_per_user"]
            )

    for content_id in watch_distribution_dict:
        content_type = content_type_map[content_id]
        for group in watch_distribution_dict[content_id]:
            watch_distribution = watch_distribution_dict[content_id][group]
            cluster_group = min(
                bisect.bisect_left(
                    dimension_bins[content_type], watch_distribution["bin"]
                ),
                len(dimension_bins[content_type]) - 1,
            )
            default_watch_per_user = dimension_dict[content_type_map[content_id]][
                cluster_group
            ]["watch_per_user"]
            watch_per_day = watch_distribution["watch_per_day"]
            user_per_day = watch_distribution["user_per_day"]
            watch_per_user = []
            for day in range(day_range):
                if user_per_day[day] >= args.user_threshold:
                    cur_watch_per_user = watch_per_day[day] / max(user_per_day[day], 1)
                    watch_per_user.append(cur_watch_per_user)
                elif day == 0:
                    cur_watch_per_user = watch_per_day[day] / max(user_per_day[day], 1)
                    smoothed_factor = user_per_day[day] / args.user_threshold
                    smoothed_watch = (
                        smoothed_factor * cur_watch_per_user
                        + (1 - smoothed_factor) * default_watch_per_user[day]
                    )
                    watch_per_user.append(smoothed_watch)
                else:
                    past_complete_watch = sum(watch_per_user)
                    cur_watch_per_user = past_complete_watch * (
                        default_watch_per_user[day]
                        / np.sum(default_watch_per_user[:day])
                    )
                    watch_per_user.append(cur_watch_per_user)
            watch_per_user = np.array(watch_per_user, dtype=np.float64)
            watch_distribution["watch_per_user"] = watch_per_user
            watch_distribution["complete_watch"] = np.cumsum(watch_per_user)

    fs = s3fs.S3FileSystem(use_ssl=False)
    output_path = data_path(DataPath.S3_WATCHED_CUMULATIVE_WATCH, TENANT) % args.variant
    with fs.open(os.path.join(output_path, end_date, "data.json"), "w") as f:
        content_result = {}
        for content_id in watch_distribution_dict:
            content_result[content_id] = {"bin": [], "value": {}}
            for group in sorted(watch_distribution_dict[content_id].keys()):
                content_result[content_id]["value"][group] = watch_distribution_dict[
                    content_id
                ][group]["complete_watch"].tolist()
                content_result[content_id]["bin"].append(
                    watch_distribution_dict[content_id][group]["bin"]
                )
        cluster_result = {}
        for content_type in dimension_dict:
            cluster_result[content_type] = {"bin": [], "value": {}}
            for group in sorted(dimension_dict[content_type].keys()):
                cluster_result[content_type]["value"][group] = dimension_dict[
                    content_type
                ][group]["complete_watch"].tolist()
                cluster_result[content_type]["bin"].append(
                    dimension_bins[content_type][group]
                )
        result = {"content": content_result, "cluster": cluster_result}
        json.dump(result, f, sort_keys=True)
    with fs.open(os.path.join(output_path, "checkpoint"), "w") as f:
        f.write(end_date)
    set_s3_file_acl(os.path.join(output_path, end_date, "data.json"))
    set_s3_file_acl(os.path.join(output_path, "checkpoint"))
    print("finish upload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("end_date", type=str, help="end date, YYYY-mm-dd")
    parser.add_argument("--day_range", type=int, default=7)
    parser.add_argument("--variant", type=str, default="simple")
    parser.add_argument("--user_threshold", type=int, default=30)
    parser.add_argument("--max_bin_num", type=int, default=20)
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()

    main(args)
