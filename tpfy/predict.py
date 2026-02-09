from tpfy.emr_prediction.emr_prediction import run_prediction, run_upload
from common.spark_utils import create_spark_context, create_sql_context, set_partitions
from common.config.utils import tenant_countries
import argparse
import sys


def run(args):

    spark_context = create_spark_context(
        "Prediction at %s" % args.date_str, enable_hive=True
    )
    sql_context = create_sql_context(spark_context)
    set_partitions(sql_context, 1024, adjust=True)

    for country in tenant_countries(args.countries):
        if args.upload_dynamo:
            run_upload(sql_context, args, country)
        else:
            run_prediction(spark_context, sql_context, args, country)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPFY offline Prediction.")
    parser.add_argument("date_str", type=str)
    # parser.add_argument('--pid_subset', type=str, default=None)
    parser.add_argument(
        "--instances",
        type=int,
        default=200,
        help="number of concurrent predictor tasks",
    )
    parser.add_argument(
        "--result_count",
        type=int,
        default=40,
        help="number of contents in recommendation results",
    )
    parser.add_argument(
        "--exclude_sports_and_news",
        action="store_true",
        help="skip recommendation by sports and news",
    )
    parser.add_argument(
        "--export_to_ap", action="store_true", help="export tpfy results to AP team"
    )
    parser.add_argument("--dump_s3", action="store_true", help="write results to s3")
    parser.add_argument(
        "--upload_dynamo",
        action="store_true",
        help="upload results to dynamo, should use standalone after dump_s3",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="only predict results for last N days active users",
    )
    parser.add_argument(
        "--min_pay_ratio",
        type=float,
        default=0.0,
        help="minimum ratio of premium contents in recommendation results",
    )
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()

    print(sys.version)
    if args.dump_s3 or args.upload_dynamo:
        run(args)
    else:
        print("ERROR: No persistence of prediction results.")
        parser.print_help()
        sys.exit(1)
