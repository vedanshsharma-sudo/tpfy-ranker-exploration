import argparse
import os
import pickle

from contextlib import closing

import s3fs

from common.config.constants import DataPath
from common.config.utils import data_path, tenant_countries
from common.config import TENANT

from distribute.distribute_context import TFDistributeContext, SingleNodeCluster

from tpfy.tf_dataset.metadata import Metadata
from tpfy.tf_model.tpfy_distributed_trainer import TpfyDistributeTrainer
from tpfy.tpfy_config import get_tpfy_hparams, get_tpfy_hparams_spark

S3_TPFY_METADATA_CACHE_BASE = data_path(
    DataPath.S3_TPFY_METADATA_CACHE_BASE, TENANT, ""
)


def load_metadata(date, countries, sql_context=None, hive_connector_factory=None):
    base_dir = "dist_tmp/tpfy"
    os.makedirs(base_dir, exist_ok=True)
    metadata = Metadata(date, countries, base_path=base_dir + "/%s")
    metadata.load(
        sql_context=sql_context, hive_connector_factory=hive_connector_factory
    )
    return metadata


def load_metadata_cached(cache_key, cache_base_dir, metadata_factory):
    is_s3 = cache_base_dir.startswith("s3")
    if cache_key:
        cache_path = os.path.join(cache_base_dir, f"metadata_{cache_key}.pkl")
        if is_s3:
            fs = s3fs.S3FileSystem(anon=False, use_ssl=False)
            local_path = f"metadata_{cache_key}.pkl"
            if fs.exists(cache_path):
                fs.download(cache_path, local_path)  # direct pickling from s3 is slow
                with open(local_path, "rb") as f:
                    metadata = pickle.load(f)
            else:
                metadata = metadata_factory()
                with open(local_path, "wb") as f:
                    pickle.dump(metadata, f)
                fs.upload(local_path, cache_path)
            os.remove(local_path)
            return metadata
        else:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    metadata = pickle.load(f)
            else:
                metadata = metadata_factory()
                os.makedirs(cache_base_dir, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(metadata, f)
            return metadata
    else:
        return metadata_factory()


def init_spark(args, countries):
    from common.spark_utils import (
        create_spark_context,
        create_sql_context,
        set_partitions,
    )
    from distribute.spark_zmq_distribute_context import SparkZmqTFCluster

    sc = create_spark_context("dist_train_" + args.model_name, enable_hive=True)
    sql_context = create_sql_context(sc)
    set_partitions(sql_context, 1)

    metadata = load_metadata_cached(
        args.metadata_cache_key,
        S3_TPFY_METADATA_CACHE_BASE,
        lambda: load_metadata(args.date, countries, sql_context=sql_context),
    )

    hparams = get_tpfy_hparams_spark()

    spark_cluster = SparkZmqTFCluster(
        sc, args.num_ps, args.num_worker, args.driver_as_ps
    )

    return spark_cluster, metadata, hparams


def init_mpi(args, countries):
    from mpi4py import MPI
    from distribute.mpi_distribute_context import MpiCluster
    from common.hive_utils import create_hive_connection

    mpi_cluster = MpiCluster(args.num_ps, args.num_worker)

    hparams = get_tpfy_hparams()

    def metadata_factory():
        return load_metadata(
            args.date, countries, hive_connector_factory=create_hive_connection
        )

    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        metadata = load_metadata_cached(
            args.metadata_cache_key, "dist_tmp", metadata_factory
        )
    else:
        metadata = None
    metadata = comm.bcast(metadata)
    return mpi_cluster, metadata, hparams


def init_local(args, countries):
    from common.hive_utils import create_hive_connection

    single_node_cluster = SingleNodeCluster()

    hparams = get_tpfy_hparams()

    def metadata_factory():
        return load_metadata(
            args.date, countries, hive_connector_factory=create_hive_connection
        )

    metadata = load_metadata_cached(
        args.metadata_cache_key, "dist_tmp", metadata_factory
    )
    return single_node_cluster, metadata, hparams


def run(args):
    env = args.env
    countries = tenant_countries(args.countries)
    if env == "spark":
        cluster, metadata, hparams = init_spark(args, countries)
    elif env == "mpi":
        cluster, metadata, hparams = init_mpi(args, countries)
    elif env == "local":
        cluster, metadata, hparams = init_local(args, countries)
    else:
        raise Exception("Unknown env", env)
    hparams.countries = countries

    def node_func(distribute_ctx: TFDistributeContext):
        logger = distribute_ctx.logger

        print("metadata item count", len(metadata.item_index))
        logger.log("hparams", hparams)
        logger.log("start", distribute_ctx.task, distribute_ctx.index)

        trainer = TpfyDistributeTrainer(
            model_name=args.model_name,
            date=args.date,
            metadata=metadata,
            hparams=hparams,
            num_producer=args.num_producer_per_worker,
            distribute_ctx=distribute_ctx,
            workdir=f"dist_tmp",
            model_base_dir=args.model_base_dir,
            log_base_dir=args.log_base_dir,
            ps_inter=args.ps_inter,
            ps_intra=args.ps_intra,
            worker_inter=args.worker_inter,
            worker_intra=args.worker_intra,
            ns_weight=args.ns_weight,
            upload=args.upload,
            reload_local_model=args.reload_local_model,
            reload_s3_model=args.reload_s3_model,
            reload_clear_nn=args.clear_nn,
        )
        trainer.run()

    print("model name", args.model_name)
    cluster.run(node_func)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPFY offline Training.")
    parser.add_argument("env", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("date", type=str)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--num_producer_per_worker", type=int, default=0)
    parser.add_argument("--num_ps", type=int, default=1)
    parser.add_argument("--ps_inter", type=int, default=1)
    parser.add_argument("--ps_intra", type=int, default=2)
    parser.add_argument("--worker_inter", type=int, default=1)
    parser.add_argument("--worker_intra", type=int, default=1)
    parser.add_argument("--model_base_dir", type=str)
    parser.add_argument("--log_base_dir", type=str, default=None)
    parser.add_argument(
        "--ns_weight", type=float, default=1.0, help="weight of negative sampler"
    )
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="skip downloading training data from s3",
    )
    parser.add_argument("--upload", action="store_true", help="uploading model to s3")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="to pass non-empty params limit in data pipeline",
    )
    parser.add_argument("--enable_metrics", action="store_true")
    parser.add_argument("--metadata_cache_key", type=str, default=None)
    parser.add_argument("--reload_local_model", type=str, default=None)
    parser.add_argument("--reload_s3_model", type=str, default=None)
    parser.add_argument("--clear_nn", action="store_true")
    parser.add_argument("--driver_as_ps", action="store_true")
    parser.add_argument(
        "--countries",
        type=str,
        help="countries to run, separated with comma. "
        "default is None. fallback to region countries",
    )
    args = parser.parse_args()
    print("Start training")
    run(args)
