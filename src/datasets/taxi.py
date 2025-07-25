import os
import numpy as np
import pandas as pd
from shapely import wkt
import torch
from typing import Optional

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
        return f"taxi_data_preproc.npy"

    def _get_full_path(self):
        return os.path.join(self.cfg.data_dir, self._get_filename())

    def get_data(self, use_tensor: bool = False):
        if os.path.exists(self._get_full_path()):
            self._load_data()
        else:
            self._generate_chain()
            self._generate_data()
        return self._get_data(use_tensor)

    def _generate_chain(self):
        self._load_raw_data()
        self._preprocess_zones()
        self._preprocess_trips()
        self._filter_manhattan_trips()

        _, R_ten, R_vec, Q_ten, Q_mat, P_ten, P_mat = self._build_transition_tensors(df_raw=self.df_trips_manhattan)
        self._R_true_ten = R_ten
        self._R_true_vec = R_vec
        self._Q_true_ten = Q_ten
        self._Q_true_mat = Q_mat
        self._P_true_ten = P_ten
        self._P_true_mat = P_mat

    def _generate_data(self):
        self.R_ten_list, self.R_vec_list = [], []
        self.Q_ten_list, self.Q_mat_list = [], []
        self.P_ten_list, self.P_mat_list = [], []
        for trial in range(self.cfg.trials):
            R_ten_trial, R_vec_trial = {}, {} 
            Q_ten_trial, Q_mat_trial = {}, {}
            P_ten_trial, P_mat_trial = {}, {}
            df = self.df_trips_manhattan.copy(deep=True)
            for n in self.cfg.n_samples_all[::-1]:
                df, R_ten, R_vec, Q_ten, Q_mat, P_ten, P_mat = self._build_transition_tensors(df_raw=df, n=n)
                R_ten_trial[n], R_vec_trial[n] = R_ten, R_vec 
                Q_ten_trial[n], Q_mat_trial[n] = Q_ten, Q_mat 
                P_ten_trial[n], P_mat_trial[n] = P_ten, P_mat 
            self.R_ten_list.append(R_ten_trial), self.R_vec_list.append(R_vec_trial)
            self.Q_ten_list.append(Q_ten_trial), self.Q_mat_list.append(Q_mat_trial)
            self.P_ten_list.append(P_ten_trial), self.P_mat_list.append(P_mat_trial)
        self.save_data()

    def _get_data(self, use_tensor: bool):
        if not use_tensor:
            mc_true = [MarkovChainMatrix(
                P=self._P_true_mat,
                R=self._R_true_vec,
                Q=self._Q_true_mat) for _ in range(self.cfg.trials)]

            mc_emp = [MarkovChainMatrix(
                P=self.P_mat_list[trial][self.cfg.n_samples],
                R=self.R_vec_list[trial][self.cfg.n_samples],
                Q=self.Q_mat_list[trial][self.cfg.n_samples]) for trial in range(self.cfg.trials)]

        else:
            mc_true = [MarkovChainTensor(
                P=self._P_true_ten,
                R=self._R_true_ten,
                Q=self._Q_true_ten) for _ in range(self.cfg.trials)]

            mc_emp = [MarkovChainTensor(
                P=self.P_ten_list[trial][self.cfg.n_samples],
                R=self.R_ten_list[trial][self.cfg.n_samples],
                Q=self.Q_ten_list[trial][self.cfg.n_samples]) for trial in range(self.cfg.trials)]

        traj_true = [None for _ in range(self.cfg.trials)]
        traj_emp = [None for _ in range(self.cfg.trials)]

        return traj_true, traj_emp, mc_true, mc_emp

    def save_data(self):
        assert self._P_true_mat is not None and self.P_mat_list is not None

        data_path = self._get_full_path()
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        np.save(
            data_path,
            {
                "P_mat_true": self._P_true_mat.numpy(),
                "P_ten_true": self._P_true_ten.numpy(),
                "R_vec_true": self._R_true_vec.numpy(),
                "R_ten_true": self._R_true_ten.numpy(),
                "Q_mat_true": self._Q_true_mat.numpy(),
                "Q_ten_true": self._Q_true_ten.numpy(),
                "P_mat_emp": [{k: v.numpy() for k, v in d.items()} for d in self.P_mat_list],
                "P_ten_emp": [{k: v.numpy() for k, v in d.items()} for d in self.P_ten_list],
                "R_vec_emp": [{k: v.numpy() for k, v in d.items()} for d in self.R_vec_list],
                "R_ten_emp": [{k: v.numpy() for k, v in d.items()} for d in self.R_ten_list],
                "Q_mat_emp": [{k: v.numpy() for k, v in d.items()} for d in self.Q_mat_list],
                "Q_ten_emp": [{k: v.numpy() for k, v in d.items()} for d in self.Q_ten_list],
            },
        )

    def _load_data(self):
        data_path = self._get_full_path()
        data = np.load(data_path, allow_pickle=True).item()

        self._P_true_mat = torch.tensor(data["P_mat_true"])
        self._P_true_ten = torch.tensor(data["P_ten_true"])
        self._R_true_vec = torch.tensor(data["R_vec_true"])
        self._R_true_ten = torch.tensor(data["R_ten_true"])
        self._Q_true_mat = torch.tensor(data["Q_mat_true"])
        self._Q_true_ten = torch.tensor(data["Q_ten_true"])
        self.P_mat_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["P_mat_emp"]]
        self.P_ten_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["P_ten_emp"]]
        self.R_vec_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["R_vec_emp"]]
        self.R_ten_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["R_ten_emp"]]
        self.Q_mat_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["Q_mat_emp"]]
        self.Q_ten_list = [{k: torch.from_numpy(v).float() for k, v in d.items()} for d in data["Q_ten_emp"]] 

        self.Is = torch.tensor(self._P_true_ten.shape[:2])

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

        neighborhoods_from_central_park_to_south = [
            'Alphabet City', 'Battery Park', 'Battery Park City', 'Central Park',
            'Chinatown', 'Clinton East', 'Clinton West', 'East Chelsea', 
            'East Village', 'Financial District North', 'Financial District South', 
            'Flatiron', 'Hudson Sq', 'Garment District', 
            "Governor's Island/Ellis Island/Liberty Island", 'Gramercy', 
            'Greenwich Village North', 'Greenwich Village South', 'Kips Bay', 
            'Lenox Hill East', 'Lenox Hill West', 'Lincoln Square East', 
            'Lincoln Square West', 'Little Italy/NoLiTa', 'Lower East Side', 
            'Meatpacking/West Village West', 'Midtown Center', 'Midtown East', 
            'Midtown North', 'Midtown South', 'Murray Hill', 
            'Penn Station/Madison Sq West', 'Seaport', 'SoHo', 
            'Stuy Town/Peter Cooper Village', 'Sutton Place/Turtle Bay North', 
            'Times Sq/Theatre District', 'TriBeCa/Civic Center', 
            'Two Bridges/Seward Park', 'UN/Turtle Bay South', 'Union Sq', 
            'Upper East Side North', 'Upper East Side South', 'Upper West Side South', 
            'West Chelsea/Hudson Yards', 'West Village', 'World Trade Center', 
            'Yorkville East', 'Yorkville West'
        ]

        df = df[
            (df['PUborough'] == 'Manhattan') & 
            (df['DOborough'] == 'Manhattan') & 
            (df['PUzone'].isin(neighborhoods_from_central_park_to_south)) & 
            (df['DOzone'].isin(neighborhoods_from_central_park_to_south))
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

        self.n_periods = 6  # 24h/4
        self.n_locations = df['PULocationID'].nunique()
        self.df_trips_manhattan = df

    def _build_transition_tensors(self, df_raw, n: Optional[int]=None):
        if n is not None:
            df = df_raw.sample(n=n, replace=False)
        else:
            df = df_raw.copy(deep=True)

        R_ten = np.zeros((self.n_periods, self.n_locations))
        Q_ten = np.zeros((self.n_periods, self.n_locations, self.n_periods, self.n_locations))

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

        Q_mat = Q_ten.reshape(self.n_periods * self.n_locations, self.n_periods * self.n_locations)
        P_mat = P_ten.reshape(self.n_periods * self.n_locations, self.n_periods * self.n_locations)

        return (
            df,
            torch.from_numpy(R_ten).float(),
            torch.from_numpy(R_vec).float(),
            torch.from_numpy(Q_ten).float(),
            torch.from_numpy(Q_mat).float(),
            torch.from_numpy(P_ten).float(),
            torch.from_numpy(P_mat).float(),
        )
