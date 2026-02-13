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


def main(args):
    end_date = args.end_date
    day_range = args.day_range
    countries = tenant_countries(args.countries)

    content_dict = {}

    spark_context = create_spark_context(
        "Cumulative Watch - %s" % end_date, enable_hive=True
    )
    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 1)

    movies, tv_shows = get_cms_store(
        country_list=countries,
        movie_fields=[],
        tv_show_fields=[],
        sql_context=sql_context,
    )
    for country in countries:
        for content_id, tv_show in tv_shows.all_entities_of_country(country).items():
            content_dict[content_id] = tv_show
        for content_id, movie in movies.all_entities_of_country(country).items():
            content_dict[content_id] = movie

    watch_distribution_dict = defaultdict(
        lambda: {
            "watch_per_day": np.zeros([day_range + 1], dtype=np.float64),
            "user_per_day": np.zeros([day_range + 1], dtype=np.float64),
        }
    )
    dimension_dict = defaultdict(
        lambda: {
            "watch_per_day": np.zeros([day_range + 1], dtype=np.float64),
            "user_per_day": np.zeros([day_range + 1], dtype=np.float64),
        }
    )
    for country in countries:
        file_path = data_path(DataPath.S3_WATCH_DISTRIBUTION, country) % end_date
        base_dir = "etl_tmp"
        local_path = os.path.join(base_dir, "watch_distribution.csv")
        download_single_file(file_path, local_path, ".csv")

        with open(local_path) as f:
            reader = csv.DictReader(f)
            for i, row_dict in enumerate(reader):
                content_id = int(row_dict["content_id"])
                if content_id not in content_dict:
                    print(content_id)
                    continue

                content = content_dict[content_id]

                watch_per_day = np.array(
                    [int(row_dict[f"sum_wt_{i}"]) for i in range(day_range + 1)],
                    dtype=np.float64,
                )
                user_per_day = np.array(
                    [int(row_dict[f"sum_count_{i}"]) for i in range(day_range + 1)],
                    dtype=np.float64,
                )
                watch_distribution_dict[content_id]["watch_per_day"] += watch_per_day
                watch_distribution_dict[content_id]["user_per_day"] += user_per_day
                if content.content_type_id == 100:
                    dimension_dict["show"]["watch_per_day"] += watch_per_day
                    dimension_dict["show"]["user_per_day"] += user_per_day
                elif content.content_type_id == 200:
                    dimension_dict["movie"]["watch_per_day"] += watch_per_day
                    dimension_dict["movie"]["user_per_day"] += user_per_day

    for key in dimension_dict:
        dimension_dict[key]["watch_per_user"] = dimension_dict[key][
            "watch_per_day"
        ] / np.maximum(dimension_dict[key]["user_per_day"], 1)
        dimension_dict[key]["complete_watch"] = np.sum(
            dimension_dict[key]["watch_per_user"][: args.effect_day]
        )
    print(f"clustered watch distribution: {dimension_dict}")

    for content_id in watch_distribution_dict:
        content = content_dict[content_id]
        watch_per_day = watch_distribution_dict[content_id]["watch_per_day"]
        user_per_day = watch_distribution_dict[content_id]["user_per_day"]
        if content.content_type_id == 100:
            default_watch_per_user = dimension_dict["show"]["watch_per_user"]
        else:
            default_watch_per_user = dimension_dict["movie"]["watch_per_user"]
        watch_per_user = []
        for day in range(day_range + 1):
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
                    default_watch_per_user[day] / np.sum(default_watch_per_user[:day])
                )
                watch_per_user.append(cur_watch_per_user)
        watch_per_user = np.array(watch_per_user, dtype=np.float64)
        watch_distribution_dict[content_id]["watch_per_user"] = watch_per_user
        watch_distribution_dict[content_id]["complete_watch"] = np.sum(
            watch_per_user[: args.effect_day]
        )

    fs = s3fs.S3FileSystem(use_ssl=False)
    output_path = data_path(DataPath.S3_CUMULATIVE_WATCH, TENANT) % args.variant
    with fs.open(os.path.join(output_path, end_date, "data.json"), "w") as f:
        content_result = {
            cid: watch_distribution_dict[cid]["complete_watch"]
            for cid in watch_distribution_dict
        }
        cluster_result = {
            key: dimension_dict[key]["complete_watch"] for key in dimension_dict
        }
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
    parser.add_argument("--day_range", type=int, default=30)
    parser.add_argument("--effect_day", type=int, default=14)
    parser.add_argument("--variant", type=str, default="simple-14")
    parser.add_argument("--user_threshold", type=int, default=30)
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()

    main(args)
