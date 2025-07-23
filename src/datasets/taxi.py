import os
import numpy as np
import pandas as pd
from shapely import wkt
import torch

from src.estimation.empirical import EmpiricalEstimator
from src.chains.models import MarkovChainMatrix, MarkovChainTensor
from src.utils import (
    chain_mat_to_ten,
    mat_to_ten,
    ten_to_mat,
)


class TaxiDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.estimator = EmpiricalEstimator()

    def _get_filename(self):
        return f"taxi_{self.cfg.generation_type}.npy"

    def _get_full_path(self):
        return os.path.join(self.cfg.data_dir, self._get_filename())

    def get_data(self, use_tensor: bool = False):
        if os.path.exists(self._get_full_path()):
            return self._load_data(use_tensor)
        else:
            self._generate_chain()
            return self._generate_data(use_tensor)

    def _generate_chain(self):
        self._load_raw_data()
        self._preprocess_zones()
        self._preprocess_trips()
        self._filter_manhattan_trips()
        self._build_transition_tensors()
        print("Dataset preprocessed. Starting chain generation...")

    def _generate_data(self, use_tensor: bool):
        dims = torch.tensor(self._P_true_ten.shape[:2])
        I = torch.prod(dims).item()
        self.I = I
        self.Is = dims

        if self.cfg.generation_type == "matrix":
            mc = MarkovChainMatrix(P=self._P_true_mat, R=self._R_true_vec, Q=self._Q_true_mat)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_matrix_batch(trajectories, I)
            P_emp_mats, _ = zip(*estimates)
        else:
            mc = MarkovChainTensor(P=self._P_true_ten, R=self._R_true_ten, Q=self._Q_true_ten)
            trajectories = mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=self.cfg.trials,
                burn_in=self.cfg.burn_in,
            )
            estimates = self.estimator.estimate_tensor_batch(trajectories, dims)
            P_emp_tensors, _ = zip(*estimates)
            P_emp_mats = [ten_to_mat(P, I) for P in P_emp_tensors]
        self._P_emp_list_mat = list(P_emp_mats)

        mc_true = [MarkovChainMatrix(
            P=self._P_true_mat,
            R=self._R_true_vec,
            Q=self._Q_true_mat) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_mats]

        trajectories_true = [
            mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=1,
                burn_in=self.cfg.burn_in,
            )[0]
            for mc in mc_true
        ]
        trajectories_emp = [
            mc.simulate(
                num_steps=self.cfg.length,
                num_trajectories=1,
                burn_in=self.cfg.burn_in,
            )[0]
            for mc in mc_emp
        ]

        if use_tensor:
            trajectories_true = [chain_mat_to_ten(traj, dims) for traj in trajectories_true]
            trajectories_emp = [chain_mat_to_ten(traj, dims) for traj in trajectories_emp]

            P_emp_tensors = [mat_to_ten(P, dims) for P in P_emp_mats]
            mc_true = [MarkovChainTensor(
                P=self._P_true_ten,
                R=self._R_true_ten,
                Q=self._Q_true_ten) for _ in range(self.cfg.trials)]
            mc_emp = [MarkovChainTensor(P) for P in P_emp_tensors]
            trajectories_true = torch.stack([
                torch.stack([torch.tensor(x, dtype=torch.long) for x in traj])
                for traj in trajectories_true
            ])
            trajectories_emp = torch.stack([
                torch.stack([torch.tensor(x, dtype=torch.long) for x in traj])
                for traj in trajectories_emp
            ])
        else:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)

        self.mc_true = mc_true
        self.mc_emp = mc_emp
        self.trajectories_true = trajectories_true
        self.trajectories_emp = trajectories_emp

        self.save_data()
        return trajectories_true, trajectories_emp, mc_true, mc_emp

    def save_data(self):
        assert self._P_true_mat is not None and self._P_emp_list_mat is not None

        data_path = self._get_full_path()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(
            data_path,
            {
                "P_true": self._P_true_mat.numpy(),
                "P_emp": [P.numpy() for P in self._P_emp_list_mat],
                "trajectories_true": self.trajectories_true.numpy(),
                "trajectories_emp": self.trajectories_emp.numpy(),
                "dims": self.Is,
            },
        )

    def _load_data(self, use_tensor: bool):
        data_path = self._get_full_path()
        data = np.load(data_path, allow_pickle=True).item()

        P_true = torch.tensor(data["P_true"])
        P_emp_list = [torch.tensor(P) for P in data["P_emp"]]
        trajectories_true = data["trajectories_true"]
        trajectories_emp = data["trajectories_emp"]
        dims = torch.tensor(data.get("dims", [P_true.shape[0]]))
        I = torch.prod(dims).item()

        self.I = I
        self.Is = dims
        self.cfg.dims = dims.tolist()
        self.cfg.trials = len(P_emp_list)
        self.cfg.length = trajectories_true.shape[1]

        mc_true = [MarkovChainMatrix(P_true) for _ in range(self.cfg.trials)]
        mc_emp = [MarkovChainMatrix(P) for P in P_emp_list]

        if use_tensor:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)
            P_true_tensor = mat_to_ten(P_true, dims)
            P_emp_tensors = [mat_to_ten(P, dims) for P in P_emp_list]
            mc_true = [MarkovChainTensor(P_true_tensor) for _ in range(self.cfg.trials)]
            mc_emp = [MarkovChainTensor(P) for P in P_emp_tensors]
        else:
            trajectories_true = torch.tensor(trajectories_true, dtype=torch.long)
            trajectories_emp = torch.tensor(trajectories_emp, dtype=torch.long)

        self.mc_true = mc_true
        self.mc_emp = mc_emp
        self.trajectories_true = trajectories_true
        self.trajectories_emp = trajectories_emp

        return trajectories_true, trajectories_emp, mc_true, mc_emp

    def _load_raw_data(self):
        self.df_trips = pd.read_parquet(self.cfg.trips_path)
        self.df_zones = pd.read_csv(self.cfg.zones_path)

    @staticmethod
    def _get_longitude(polygon):
        multipolygon = wkt.loads(polygon)
        return multipolygon.centroid.x

    @staticmethod
    def _get_latitude(polygon):
        multipolygon = wkt.loads(polygon)
        return multipolygon.centroid.y

    def _preprocess_zones(self):
        df = self.df_zones
        df['lon'] = df['the_geom'].apply(self._get_longitude)
        df['lat'] = df['the_geom'].apply(self._get_latitude)
        self.df_zones = df

    @staticmethod
    def _categorize_time(hour):
        if 6 <= hour <= 11:
            return 0
        elif 12 <= hour <= 17:
            return 1
        else:
            return 2

    def _preprocess_trips(self):
        df = self.df_trips
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        df['PUhour'] = df['tpep_pickup_datetime'].dt.hour // 4
        df['DOhour'] = df['tpep_dropoff_datetime'].dt.hour // 4
        df['PUtime_period'] = df['PUhour'].apply(self._categorize_time)
        df['DOtime_period'] = df['DOhour'].apply(self._categorize_time)

        zone_cols = ['LocationID', 'borough', 'zone', 'lon', 'lat']
        df = df.merge(self.df_zones[zone_cols], left_on='PULocationID', right_on='LocationID', suffixes=('', '_PU'))
        df.rename(columns={'lon': 'PUlon', 'lat': 'PUlat', 'borough': 'PUborough', 'zone': 'PUzone'}, inplace=True)
        df = df.merge(self.df_zones[zone_cols], left_on='DOLocationID', right_on='LocationID', suffixes=('', '_DO'))
        df.rename(columns={'lon': 'DOlon', 'lat': 'DOlat', 'borough': 'DOborough', 'zone': 'DOzone'}, inplace=True)
        df.drop(columns=['LocationID', 'LocationID_DO'], inplace=True)
        self.df_trips = df

    def _filter_manhattan_trips(self):
        df = self.df_trips
        df = df[
            (df['PUborough'] == 'Manhattan') & 
            (df['DOborough'] == 'Manhattan')
        ].copy()

        unique_pu_ids = set(df['PULocationID'].unique())
        unique_do_ids = set(df['DOLocationID'].unique())
        common_location_ids = unique_pu_ids.intersection(unique_do_ids)

        df = df[
            (df['PULocationID'].isin(common_location_ids)) &
            (df['DOLocationID'].isin(common_location_ids))
        ].copy()

        df['PULocationID'], self._pu_id_mapping = pd.factorize(df['PULocationID'])
        df['DOLocationID'], self._do_id_mapping = pd.factorize(df['DOLocationID'])
        df['PULocationID'] += 1
        df['DOLocationID'] += 1

        self.df_trips_manhattan = df

    def _build_transition_tensors(self):
        df = self.df_trips_manhattan
        n_periods = 6  # 24h/4
        n_locations = df['PULocationID'].nunique()
        
        R_ten = np.zeros((n_periods, n_locations))
        Q_ten = np.zeros((n_periods, n_locations, n_periods, n_locations))

        for _, row in df.iterrows():
            pu_period_idx = int(row['PUhour'])
            pu_loc_idx = int(row['PULocationID']) - 1
            do_period_idx = int(row['DOhour'])
            do_loc_idx = int(row['DOLocationID']) - 1
            R_ten[pu_period_idx, pu_loc_idx] += 1
            Q_ten[pu_period_idx, pu_loc_idx, do_period_idx, do_loc_idx] += 1

        R_ten_sum = R_ten.sum()
        if R_ten_sum > 0:
            R_ten /= R_ten_sum
        R_vec = R_ten.flatten()

        Q_ten_sum = Q_ten.sum()
        if Q_ten_sum > 0:
            Q_ten /= Q_ten_sum
        P_sum = Q_ten.sum(axis=(2, 3), keepdims=True)
        P_sum[P_sum == 0] = 1
        P_ten = Q_ten / P_sum

        Q_mat = Q_ten.reshape(n_periods * n_locations, n_periods * n_locations)
        P_mat = P_ten.reshape(n_periods * n_locations, n_periods * n_locations)

        self._R_true_ten = torch.from_numpy(R_ten).float()
        self._R_true_vec = torch.from_numpy(R_vec).float()
        self._Q_true_ten = torch.from_numpy(Q_ten).float()
        self._Q_true_mat = torch.from_numpy(Q_mat).float()
        self._P_true_ten = torch.from_numpy(P_ten).float()
        self._P_true_mat = torch.from_numpy(P_mat).float()
