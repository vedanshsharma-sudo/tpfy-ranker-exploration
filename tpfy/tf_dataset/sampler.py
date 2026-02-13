import random
from abc import ABC, abstractmethod

from common.data.behavior.watch import Watch2
from common.cms_esknol_utils import Movie, Show


IMPR_SAMPLE_LIMIT = 1000


class ZeroEntMixIn:
    def is_valid(self, metadata, sample, timestamp):
        ent_watches = [
            watch for watch in sample.watches if watch.first_watch < timestamp
        ]
        valid = len(ent_watches) == 0
        return valid and super(ZeroEntMixIn, self).is_valid(metadata, sample, timestamp)


class SportsUserMixIn:
    def __init__(self, *args, **kwargs):
        self.filter_stats = [0, 0, 0, 0]
        super(SportsUserMixIn, self).__init__(*args, **kwargs)

    def is_valid(self, metadata, sample, timestamp):
        ent_watches = [
            watch.watch_len for watch in sample.watches if watch.first_watch < timestamp
        ]
        ent_length = sum(ent_watches)
        sports_watches = [
            watch.watch_len
            for watch in sample.ns_watches
            if watch.first_watch < timestamp
            and watch.content_id in metadata.sports[sample.country]
        ]
        sports_len = sum(sports_watches)

        i = 2 if ent_length < 240 * 60 else 0
        j = 1 if ent_length < sports_len else 0
        self.filter_stats[i + j] += 1

        valid = ent_length < 6000 and ent_length < sports_len
        return valid and super(SportsUserMixIn, self).is_valid(
            metadata, sample, timestamp
        )


class RandomSampling(ABC):
    def __init__(
        self,
        num_neg,
        metadata,
        feature_extractor,
        watch_len_thres,
        sample_start,
        sample_end,
        task=1,
        drop_rate=0,
        drop_days=0,
    ):
        self.num_neg = num_neg
        self.metadata = metadata
        self.extractor = feature_extractor
        self.watch_len_thres = watch_len_thres
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.candidates = {}
        for country in self.metadata.countries:
            self.candidates[country] = list(
                self.metadata.movies[country].items()
            ) + list(self.metadata.tv_shows[country].items())
        self.task = task
        self.drop_rate = drop_rate
        self.drop_days = drop_days

    def is_valid(self, metadata, sample, timestamp):
        return True

    def check_target_ts(self, ts):
        if self.sample_end and ts >= self.sample_end:
            return False
        if self.sample_start and ts < self.sample_start:
            return False
        return True

    def get_target_watches(self, sample):
        target_watch = None
        for w in reversed(sample.watches):
            if w.watch_len >= self.watch_len_thres and self.check_target_ts(
                w.first_watch
            ):
                target_watch = w
                break
        if target_watch:
            return [target_watch]
        else:
            return []

    def generate(self, sample):
        meta = ([self.task],)

        target_watches = self.get_target_watches(sample)
        if target_watches is None:
            return

        visited = set()
        for target_watch in target_watches:
            target_id = target_watch.content_id
            if (
                target_id not in self.metadata.tv_shows[sample.country]
                and target_id not in self.metadata.movies[sample.country]
            ):
                continue

            if not self.is_valid(self.metadata, sample, target_watch.first_watch):
                continue
            if target_id in visited:
                continue

            timestamp = target_watch.first_watch
            metadata = self.metadata

            watches = [w for w in sample.watches if w.first_watch < timestamp]
            ns_watches = [w for w in sample.ns_watches if w.first_watch < timestamp]
            device_info = sample.find_device_info(timestamp)

            (
                user_feature,
                sub_plans,
                aggregated_values,
            ) = self.extractor.generate_user_features(
                watches,
                ns_watches,
                sample.age,
                sample.gender,
                sample.joined_on,
                sample.sub_info,
                sample.installed_app,
                device_info,
                timestamp,
                metadata.wv,
                metadata.tv_shows[sample.country],
                metadata.movies[sample.country],
                metadata.sports[sample.country],
                sample.country,
            )

            target_feature = self.extractor.generate_target_feature(
                target_id,
                1.0,
                timestamp,
                metadata.tv_shows[sample.country],
                metadata.movies[sample.country],
                target_watch.language,
                drop_rate=self.drop_rate,
                drop_days=self.drop_days,
            )
            wide_feature = self.extractor.generate_wide_features(
                target_id,
                aggregated_values,
                sub_plans,
                timestamp,
                metadata.pv,
                metadata.tv_shows[sample.country],
                metadata.movies[sample.country],
                target_watch.language,
            )
            yield meta + user_feature + target_feature + wide_feature

            candidate_ids = self._generate_candidates(sample, timestamp, visited)

            for candidate_id in candidate_ids:
                if candidate_id in self.metadata.tv_shows[sample.country]:
                    content = self.metadata.tv_shows[sample.country][candidate_id]
                elif candidate_id in self.metadata.movies[sample.country]:
                    content = self.metadata.movies[sample.country][candidate_id]
                else:
                    continue
                visited.add(candidate_id)
                language_id = -1
                target_feature = self.extractor.generate_target_feature(
                    candidate_id,
                    0.0,
                    timestamp,
                    metadata.tv_shows[sample.country],
                    metadata.movies[sample.country],
                    language_id,
                    drop_rate=self.drop_rate,
                    drop_days=self.drop_days,
                )
                wide_feature = self.extractor.generate_wide_features(
                    candidate_id,
                    aggregated_values,
                    sub_plans,
                    timestamp,
                    metadata.pv,
                    metadata.tv_shows[sample.country],
                    metadata.movies[sample.country],
                    language_id,
                )
                yield meta + user_feature + target_feature + wide_feature

    def _generate_candidates(self, sample, timestamp, visited):
        candidate_ids = []
        for i in range(self.num_neg):
            candidate_id, content = random.choice(self.candidates[sample.country])
            if candidate_id in sample.touched_ids or candidate_id in visited:
                continue
            # FIXME: migration bug: cms3 no longer has Movie/Show entity class
            if isinstance(content, Movie) and content.start_dt > timestamp:
                continue
            if isinstance(content, Show) and content.episode_start_dt > timestamp:
                continue
            candidate_ids.append(candidate_id)
        return candidate_ids


class LastWatch(RandomSampling):
    def get_target_watches(self, sample):
        if len(sample.lastwatch_samples) > 0:
            for w in sample.lastwatch_samples[::-1]:
                if w.watch_time > self.watch_len_thres:
                    return [
                        Watch2(
                            content_id=w.content_id,
                            language=-1,
                            watch_len=w.watch_time,
                            first_watch=w.timestamp,
                            last_watch=w.timestamp,
                        )
                    ]
        return []


class AllLastWatch(RandomSampling):
    def get_target_watches(self, sample):
        l = []
        if len(sample.lastwatch_samples) > 0:
            for w in sample.lastwatch_samples[::-1]:
                if w.watch_time > self.watch_len_thres:
                    l.append(
                        Watch2(
                            content_id=w.content_id,
                            language=-1,
                            watch_len=w.watch_time,
                            first_watch=w.timestamp,
                            last_watch=w.timestamp,
                        )
                    )
        return l


class ImpressionClick(ABC):
    def __init__(
        self,
        metadata,
        feature_extractor,
        sample_start,
        sample_end,
        num_random_neg=0,
        drop_rate=0,
        drop_days=0,
        task=0,
    ):
        self.metadata = metadata
        self.extractor = feature_extractor
        self.sample_start = sample_start
        self.sample_end = sample_end
        self.drop_rate = drop_rate
        self.drop_days = drop_days
        self.task = task

        self.candidates = {}
        for country in self.metadata.countries:
            self.candidates[country] = list(
                self.metadata.movies[country].items()
            ) + list(self.metadata.tv_shows[country].items())
        self.num_random_neg = num_random_neg

    @abstractmethod
    def get_samples(self, sample):
        pass

    def is_valid(self, metadata, sample, timestamp):
        return True

    def check_target_ts(self, ts):
        if self.sample_end and ts >= self.sample_end:
            return False
        if self.sample_start and ts < self.sample_start:
            return False
        return True

    def generate(self, sample):
        meta = ([self.task],)
        metadata = self.metadata
        extractor = self.extractor

        last_ts = None
        last_user_feature_output = None

        samples = []
        labels = []
        # assume sorted
        random_visited = set()
        impr_samples = self.get_samples(sample)
        if len(impr_samples) > IMPR_SAMPLE_LIMIT:
            return []

        for impr_sample in impr_samples:
            target_id = impr_sample.content_id
            label = float(impr_sample.label)  # TODO: int
            timestamp = impr_sample.timestamp

            if not self.check_target_ts(timestamp):
                continue

            if target_id in self.metadata.tv_shows[sample.country]:
                content = self.metadata.tv_shows[sample.country][target_id]
            elif target_id in self.metadata.movies[sample.country]:
                content = self.metadata.movies[sample.country][target_id]
            else:
                # print("unknown target", target_id)
                continue

            language_id = -1
            if timestamp != last_ts:
                if not self.is_valid(metadata, sample, timestamp):
                    continue

                last_ts = timestamp

                watches = [w for w in sample.watches if w.first_watch < timestamp]
                ns_watches = [
                    w for w in sample.ns_watches if w.first_watch < timestamp
                ][:20]
                device_info = sample.find_device_info(timestamp)
                last_user_feature_output = extractor.generate_user_features(
                    watches,
                    ns_watches,
                    sample.age,
                    sample.gender,
                    sample.joined_on,
                    sample.sub_info,
                    sample.installed_app,
                    device_info,
                    timestamp,
                    metadata.wv,
                    metadata.tv_shows[sample.country],
                    metadata.movies[sample.country],
                    metadata.sports[sample.country],
                    sample.country,
                )
            user_feature, sub_plans, aggregated_values = last_user_feature_output

            target_feature = extractor.generate_target_feature(
                target_id,
                label,
                timestamp,
                metadata.tv_shows[sample.country],
                metadata.movies[sample.country],
                language_id,
                drop_rate=self.drop_rate,
                drop_days=self.drop_days,
            )
            wide_feature = extractor.generate_wide_features(
                target_id,
                aggregated_values,
                sub_plans,
                timestamp,
                metadata.pv,
                metadata.tv_shows[sample.country],
                metadata.movies[sample.country],
                language_id,
            )
            samples.append(meta + user_feature + target_feature + wide_feature)
            labels.append(label)

            if label > 0:
                if self.num_random_neg > 0:
                    for i in range(self.num_random_neg):
                        target_id, content = self._generate_candidate(
                            sample, timestamp, random_visited
                        )
                        if target_id is None:
                            continue
                        label = 0
                        language_id = -1

                        random_visited.add(target_id)
                        target_feature = extractor.generate_target_feature(
                            target_id,
                            label,
                            timestamp,
                            metadata.tv_shows[sample.country],
                            metadata.movies[sample.country],
                            language_id,
                            drop_rate=self.drop_rate,
                            drop_days=self.drop_days,
                        )
                        wide_feature = extractor.generate_wide_features(
                            target_id,
                            aggregated_values,
                            sub_plans,
                            timestamp,
                            metadata.pv,
                            metadata.tv_shows[sample.country],
                            metadata.movies[sample.country],
                            language_id,
                        )
                        samples.append(
                            meta + user_feature + target_feature + wide_feature
                        )
                        labels.append(0)
                        break

        sum_labels = sum(labels)
        if 0 < sum_labels < len(labels):
            yield from samples

    def _generate_candidate(self, sample, timestamp, visited):
        for i in range(10):
            candidate_id, content = random.choice(self.candidates[sample.country])
            if candidate_id in sample.touched_ids or candidate_id in visited:
                continue
            if isinstance(content, Movie) and content.start_dt > timestamp:
                continue
            if isinstance(content, Show) and content.episode_start_dt > timestamp:
                continue
            return candidate_id, content
        return None, None


class TpfyImpressionSampler(ImpressionClick):
    def get_samples(self, user):
        return user.tpfy_samples


class TpfyImpressionPaidSubsampleSampler(ImpressionClick):
    def get_samples(self, user):
        is_free = user.sub_info is not None
        if not is_free and random.random() < 0.1:
            return []
        return user.tpfy_samples


class TpfyImprClickNewItem(ImpressionClick):
    def get_samples(self, sample):
        if not self.sample_start:
            raise Exception("sample start is None")
        for s in sample.tpfy_samples:
            target_id = s.content_id
            ts = s.timestamp
            if not self.check_target_ts(ts):
                continue

            if target_id in self.metadata.tv_shows[sample.country]:
                content = self.metadata.tv_shows[sample.country][target_id]
                if (
                    content.start_dt >= self.sample_start
                    or content.episode_start_dt >= self.sample_start
                ):
                    return sample.tpfy_samples
            elif target_id in self.metadata.movies[sample.country]:
                content = self.metadata.movies[sample.country][target_id]
                if content.start_dt >= self.sample_start:
                    return sample.tpfy_samples
        return []


class HybridSampler:
    def __init__(self, weight_sampler):
        self.weight_sampler = weight_sampler

    def generate(self, sample):
        for weight, sampler in self.weight_sampler:
            if random.random() < weight:
                yield from sampler.generate(sample)
